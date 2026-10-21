from porefm.cli._runner import run_module


def main() -> int:
    return run_module("porefm.tools.pairwise_parquet_to_bam")


if __name__ == "__main__":
    raise SystemExit(main())
