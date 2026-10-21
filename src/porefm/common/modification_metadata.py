"""Resolve classifier modification identities without changing model execution."""


def resolve_modification_metadata(
    base, mod_code, *, metadata, num_classes, legacy_base=None, require_bam_tags=False
):
    """Return canonical base and ordered modification codes for a classifier.

    Explicit CLI values take precedence over checkpoint ``meta`` fields.
    ``legacy_base`` is the inference fallback for checkpoints without a base.
    Class zero is canonical; codes correspond to classes 1 through
    ``num_classes - 1``. Missing values remain ``None`` unless BAM output
    requires both. Supplied codes must be a nonempty sequence of nonempty
    strings with one entry per modified class; numeric codes stay strings.
    """
    if base is None:
        base = metadata.get('base')
    if base is None:
        base = legacy_base
    if mod_code is None:
        mod_code = metadata.get('mod_code')

    if base is not None and base not in ('A', 'C', 'G', 'T', 'U'):
        raise ValueError("--base (or checkpoint meta.base) must be one of A, C, G, T, U.")
    if mod_code is not None:
        if (
            not isinstance(mod_code, (list, tuple))
            or not mod_code
            or any(not isinstance(code, str) or not code.strip() for code in mod_code)
        ):
            raise ValueError("--mod_code (or checkpoint meta.mod_code) must be a nonempty list of code strings.")
        if len(mod_code) != num_classes - 1:
            raise ValueError(
                f"--mod_code must provide exactly {num_classes - 1} modification codes "
                f"for {num_classes} classes (class 0 is canonical); got {len(mod_code)}. "
                "Supply updated codes if extending a classifier checkpoint."
            )
        mod_code = list(mod_code)

    if require_bam_tags:
        missing = [flag for flag, value in (('--base', base), ('--mod_code', mod_code)) if value is None]
        if missing:
            raise ValueError(
                f"BAM output requires {' and '.join(missing)} because the checkpoint "
                "does not provide the corresponding modification metadata."
            )
    return base, mod_code
