"""Vendor the fixed ONT tables and their upstream notices into PoreFM.

Run under the project environment from any directory. Downloads always use
the immutable commit below; this is a maintainer operation, never runtime I/O.
"""

from pathlib import Path
import hashlib
import json
import struct
from urllib.request import urlopen


COMMIT = "4e56daed7fbb79b538f58e41262d5c54b07356ea"
TABLES = {
    "dna_r10.4.1_e8.2_400bps": ("dna", "r10", "dna_r10.4.1_e8.2_400bps/9mer_levels_v1.txt", 9),
    "rna004": ("rna", "rna004", "rna004/9mer_levels_v1.txt", 9),
    "rna002": ("rna", "rna002", "rna_r9.4_180mv_70bps/5mer_levels_v1.txt", 5),
}


def main():
    """Download pinned files, validate complete tables, and record digests."""
    root = Path(__file__).resolve().parents[1] / "src/porefm/resources/kmer_models"
    manifest = {"repository": "https://github.com/nanoporetech/kmer_models", "commit": COMMIT, "tables": {}}
    downloads = {}
    for path in [*(row[2] for row in TABLES.values()), "LICENSE.md", "COPYRIGHT", "README.md"]:
        url = f"https://raw.githubusercontent.com/nanoporetech/kmer_models/{COMMIT}/{path}"
        with urlopen(url, timeout=30) as response:
            downloads[path] = response.read()
    for chemistry, (seq_type, pore_type, path, k) in TABLES.items():
        levels = {}
        for line in downloads[path].decode().splitlines():
            kmer, value = line.split()
            if len(kmer) != k or set(kmer) - set("ACGT") or kmer in levels:
                raise ValueError(f"Invalid or duplicate k-mer in {path}: {kmer}")
            levels[kmer] = float(value)
        if len(levels) != 4 ** k:
            raise ValueError(f"Incomplete table: {path}")
        lookup = b"".join(struct.pack("<f", levels[kmer]) for kmer in sorted(levels))
        manifest["tables"][chemistry] = dict(
            seq_type=seq_type, pore_type=pore_type, path=path, kmer_size=k,
            file_sha256=hashlib.sha256(downloads[path]).hexdigest(),
            table_sha256=hashlib.sha256(lookup).hexdigest(),
        )
    # Finish every download/validation before modifying package resources.
    for path, content in downloads.items():
        target = root / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
    (root / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
