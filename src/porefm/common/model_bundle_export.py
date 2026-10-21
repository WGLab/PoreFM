"""Export inference components without training state or runtime overrides."""

from dataclasses import asdict
import hashlib
from pathlib import Path
import tempfile

from porefm.common.model_bundle import create_model_bundle


def _digest(path):
    """Stream the complete checkpoint SHA-256, including on Python 3.10."""
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        while chunk := stream.read(8 * 1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def export_model_bundle(output, *, kind, components, settings=None):
    """Export only the checkpoint components used by the inference workflow.

    Canonical checkpoints retain their backbone, null and scale state/configs.
    Prototype checkpoints retain the same backbone subset that query inference
    already loads. Optimizer/scheduler state, pretraining heads, EMA copies and
    training history are omitted. Whitening/calibration and per-class prototype
    artifacts remain unchanged. Effect tensors remain unchanged; their copied
    metadata is linked to the compact checkpoint after verifying the original
    checkpoint hash. Runtime scoring overrides are never added.

    Parameters match :func:`create_model_bundle`. Sources are read only and an
    existing output is rejected. Temporary compact checkpoints are removed on
    exit; model execution and the existing inference loaders are not modified.
    """
    import torch
    from porefm.preprocessing import validate_artifact_preprocessing

    output = Path(output).expanduser()
    if output.exists():
        raise FileExistsError(f'Refusing to overwrite {output}')
    paths = {name: Path(path).expanduser().resolve(strict=True) for name, path in components.items()}
    name = 'canonical_null_model_path' if kind == 'zero-shot' else 'model_path'
    original = paths[name]
    original_digest = _digest(original)
    provenance = {name: {'source': str(original), 'source_sha256': original_digest}}
    with tempfile.TemporaryDirectory(prefix='porefm-export-') as directory:
        compact = Path(directory) / 'model.pt'
        if kind == 'zero-shot':
            from porefm.workflows.canonical_null.predictor import load_canonical_null_checkpoint
            model, checkpoint = load_canonical_null_checkpoint(str(original), 'cpu')
            del checkpoint
            payload = {
                'model_class': 'OntModelForCanonicalNull',
                'backbone_config': asdict(model.backbone.config),
                'backbone_state_dict': model.backbone.state_dict(),
                'null_config': asdict(model.null_predictor.config),
                'head_state_dicts': {'null_predictor': model.null_predictor.state_dict()},
            }
            if model.scale_model is not None:
                payload['scale_config'] = asdict(model.scale_model.config)
                payload['head_state_dicts']['scale_model'] = model.scale_model.state_dict()
        else:
            from porefm.models.loading import load_model
            model = load_model(original, model_type='backbone', add_signal_cross_attn=False, device='cpu')
            payload = {'model_class': 'ONTBackbone', 'backbone_config': asdict(model.config),
                       'backbone_state_dict': model.state_dict()}
        if model.preprocessing is not None:
            payload['preprocessing'] = model.preprocessing.to_dict()
        hidden_dim = (model.backbone.config if kind == 'zero-shot' else model.config).hidden_dim
        torch.save(payload, compact)
        del model, payload
        paths[name] = compact
        print(f'Inference checkpoint: {original.stat().st_size:,} -> {compact.stat().st_size:,} bytes', flush=True)

        if 'effect_model' in paths:
            source = paths['effect_model']
            from porefm.workflows.prototypes.effects import load_effect_model
            effect = load_effect_model(source)
            if effect['metadata']['checkpoint_sha256'] != original_digest:
                raise ValueError('query backbone checkpoint does not match fitted effect artifact')
            if effect['embedding_dim'] != hidden_dim:
                raise ValueError('effect embedding dimension does not match the query backbone')
            # Preserve all fitted tensors and existing artifact defaults. The
            # existing inference hash check must identify the exported checkpoint.
            preprocessing = validate_artifact_preprocessing(effect)
            effect['metadata'] = {'checkpoint_sha256': _digest(compact)}
            if preprocessing is not None:
                effect['preprocessing'] = preprocessing.to_dict()
            target = Path(directory) / 'effects.pt'
            torch.save(effect, target)
            del effect
            paths['effect_model'] = target
            provenance['effect_model'] = {'source': str(source), 'source_sha256': _digest(source)}
        return create_model_bundle(output, kind=kind, components=paths, settings=settings,
                                   source_provenance=provenance)
