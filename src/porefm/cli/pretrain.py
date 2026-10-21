from porefm.cli._runner import run_module

def main() -> int:
    return run_module("porefm.training.pretrain")

if __name__ == "__main__":
    raise SystemExit(main())
