EMBEDDING_KEY_CHOICES = (
    'contextual_base_seq',
    'fixed_sa_base',
    'prenorm_contextual_base_seq',
    'contextual_base_seq-fixed_sa_base',
    'base_to_signal_qkv',
    'base_to_signal_head0_ctx',
)


def select_base_embedding(model_output, embedding_key):
    """Select or derive a base-aligned embedding tensor from model output.

    ``embedding_key`` may name a tensor returned by the backbone or the special
    ``"contextual_base_seq-fixed_sa_base"`` residual. The selected value must
    have shape ``(batch, bases, embedding_dim)``; subtraction is performed in
    ``float32`` after verifying that both source shapes agree.
    """
    if embedding_key == 'contextual_base_seq-fixed_sa_base':
        contextual = model_output['contextual_base_seq']
        fixed = model_output['fixed_sa_base']
        if contextual.shape != fixed.shape:
            raise ValueError(
                "Cannot subtract contextual_base_seq and fixed_sa_base with "
                f"shapes {tuple(contextual.shape)} and {tuple(fixed.shape)}"
            )
        embedding = contextual.float() - fixed.float()
    else:
        if embedding_key not in model_output:
            raise KeyError(
                f"Model output does not contain embedding key {embedding_key!r}; "
                f"available keys are {sorted(model_output)}"
            )
        embedding = model_output[embedding_key]

    if embedding is None:
        raise ValueError(f"Model output {embedding_key!r} is None")
    if embedding.ndim != 3:
        raise ValueError(
            f"Embedding {embedding_key!r} must have shape [batch, bases, dim], "
            f"got {tuple(embedding.shape)}"
        )
    return embedding


def get_embedding_dimension(model_config, embedding_key):
    """Return the feature width produced by a supported embedding selection.

    Attention-head context exposes one head, so its width is
    ``hidden_dim // num_heads``. All other selections use ``hidden_dim``.
    """
    if embedding_key == 'base_to_signal_head0_ctx':
        return int(model_config.hidden_dim) // int(model_config.num_heads)
    return int(model_config.hidden_dim)
