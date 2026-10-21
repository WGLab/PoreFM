from porefm.cli._runner import run_module

def main() -> int:
    return run_module("porefm.workflows.read_tagging.sync_embeddings")

if __name__ == "__main__":
    raise SystemExit(main())
