"""Prepare shared POD5 inputs before model and DataLoader initialization.

These helpers change input access only. Scoring workflows keep their existing
complete-read batches, model execution, and output contracts.
"""

from datetime import timedelta
import json
import os
from pathlib import Path
from types import SimpleNamespace


def prepare_pod5_samples(samples, output_dir):
    """Validate one BAM/POD5 sample and attach a reusable, unfiltered index.

    Prefer an explicit ``bam_index``, then an existing ``<bam_path>.pfmi``.
    Otherwise build ``OUTPUT_DIR/BAM_FILENAME.pfmi`` or validate/reuse that
    file if present. The full BAM filename is retained, including ``.bam``.
    Explicit indexes must exist. Invalid/stale indexes fail without replacement
    or fallback. Call once in the owning process after reserving outputs and
    before starting DataLoader workers. The sample's BAM path is unchanged.

    Returns the index's absolute path, or ``None`` for ordinary SG samples.
    The sample is updated in memory; input JSON files are never rewritten.
    """
    from porefm.data.bam_index import BamReadIndex, build_bam_index, default_bam_index_path
    from porefm.data.pod5 import validate_pod5_samples

    if not validate_pod5_samples(samples):
        return None
    sample = samples[0]
    explicit = getattr(sample, 'bam_index', None)
    adjacent = default_bam_index_path(sample.bam_path)
    if explicit:
        path = Path(explicit)
    elif adjacent.exists() or adjacent.is_symlink():
        path = adjacent
    else:
        path = Path(output_dir) / adjacent.name
        if not path.exists() and not path.is_symlink():
            print(f'Building shared POD5 inference BAM index: {path}', flush=True)
            try:
                build_bam_index(sample.bam_path, path)
            except FileExistsError:
                # Atomic publication may lose to another invocation; only accept
                # its completed output after checking the BAM fingerprint below.
                pass
    with BamReadIndex(path, sample.bam_path):
        pass
    sample.bam_index = str(path.resolve())
    print(f'POD5 inference BAM index: {sample.bam_index}', flush=True)
    return sample.bam_index


def prepare_cli_pod5_input(args):
    """Resolve ``--pod5``/``--bam_index`` after a scoring run reserves outputs.

    Return keyword arguments for ``LabelledBamSample`` and record the resolved
    index on the namespace. SG invocations return an empty mapping. The same
    single-BAM-file and recursive-POD5 discovery contract applies to all scorers.
    """
    pod5 = getattr(args, 'pod5', None)
    index = getattr(args, 'bam_index', None)
    if not pod5:
        if index:
            raise ValueError('--bam_index requires --pod5')
        return {}
    sample = SimpleNamespace(bam_path=args.bam_path, pod5=pod5, bam_index=index)
    args.bam_index = prepare_pod5_samples([sample], args.output_dir)
    return dict(pod5=pod5, bam_index=args.bam_index)


def inference_worker_count(dataset, requested, rank=0, world_size=1):
    """Cap POD5 workers to disjoint files; preserve SG worker configuration."""
    from porefm.data.pod5 import pod5_worker_count

    files = getattr(dataset, 'pod5_files', None)
    return pod5_worker_count(files, requested, rank, world_size) if files else requested


def prepare_before_process_group(action):
    """Run startup I/O once, sharing success/failure before NCCL initialization.

    ``action`` takes no arguments and returns JSON-serializable input metadata.
    Under torchrun, its existing environment rendezvous provides a TCP store;
    rank zero alone runs the action, while peers wait without a GPU collective.
    A 24-hour store timeout permits large index builds. Ordinary action errors
    are published to every rank; hard process failures remain torchrun's job.

    Returns ``(result, store)``. Pass the returned store to
    ``init_process_group(store=..., rank=..., world_size=...)`` so startup uses
    the same rendezvous. Single-process runs return ``(action(), None)``.
    """
    if 'RANK' not in os.environ or 'WORLD_SIZE' not in os.environ:
        return action(), None
    import torch.distributed as dist

    store, rank, world_size = next(dist.rendezvous(
        'env://', timeout=timedelta(hours=24),
    ))
    prefix = 'porefm-pod5-input-' + os.environ.get('TORCHELASTIC_RESTART_COUNT', '0')
    startup = dist.PrefixStore(prefix, store)
    error = None
    if rank == 0:
        try:
            result = action()
            message = json.dumps({'result': result})
        except Exception as exc:
            error = exc
            message = json.dumps({'error': f'{type(exc).__name__}: {exc}'})
        startup.set('result', message)
    response = json.loads(startup.get('result'))
    if 'error' in response:
        startup.set(f'error-seen/{rank}', '1')
        if rank == 0:
            # A static torchrun rendezvous may host the store in this process.
            # Let peers receive the error before releasing its server handle.
            startup.wait([f'error-seen/{peer}' for peer in range(world_size)])
            raise error
        raise RuntimeError('POD5 input preparation failed: ' + response['error'])
    return response['result'], store
