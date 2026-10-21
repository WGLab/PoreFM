"""Data-preparation and configuration utilities under ``porefm utils``."""

from __future__ import annotations

from collections.abc import Sequence

from porefm.cli._commands import Command, dispatch_commands


COMMANDS = {
    "bundle-model": Command("porefm.tools.bundle_model", "Package zero-shot or prototype models into one inference file."),
    "migrate-preprocessing": Command("porefm.tools.migrate_preprocessing", "Add or correct preprocessing metadata in a new checkpoint."),
    "validate-config": Command("porefm.config.cli", "Validate and inspect versioned JSON configurations."),
    "pod5-to-bam": Command("porefm.tools.pod5_to_bam", "Attach POD5 signals and write one unsorted BAM."),
    "index-bam": Command("porefm.tools.index_bam", "Save an unfiltered BAM read-ID index for POD5 inference."),
    "shard-bam": Command("porefm.tools.shard_bam", "Split BAM inputs into coordinate shards."),
    "sample-kmers": Command("porefm.tools.sample_kmers", "Sample reads by reference k-mer composition."),
    "generate-motif-coordinates": Command("porefm.tools.generate_motif_coordinates", "Generate motif-coordinate tables from FASTA."),
}


def main(argv: Sequence[str] | None = None) -> int:
    """Dispatch a utility or list available utilities without starting work."""
    return dispatch_commands(
        COMMANDS,
        argv,
        program_name="porefm utils",
        description="PoreFM data-preparation and configuration utilities.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
