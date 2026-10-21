from porefm.cli._runner import run_module

def main() -> int:
    return run_module("porefm.tools.sample_kmers")

if __name__ == "__main__":
    raise SystemExit(main())
