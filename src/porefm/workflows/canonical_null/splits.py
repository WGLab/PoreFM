"""Reproducible, source-stratified canonical-control BAM partitions.

Splits are disjoint by canonical path and inode, not guaranteed independent
by locus or sequencing run. Input BAMs are never rewritten. Byte size times
read-sampling probability is a workload proxy, not a token count.
"""

import argparse
import hashlib
import json
import math
import random
from collections import defaultdict
from pathlib import Path


def balance_worker_order(bam_list, workers, seed):
    """Balance expected bytes across strided workers, then shuffle each bin.

    Equal-count quotas preserve worker ownership when bins are interleaved.
    This is a scheduling heuristic; compressed bytes are not exact runtime.
    """
    if not bam_list:
        return []
    workers = min(max(int(workers), 1), len(bam_list))
    quotas = [len(bam_list) // workers + int(i < len(bam_list) % workers) for i in range(workers)]
    bins, loads = [[] for _ in range(workers)], [0.0] * workers
    costs = {p: Path(p).stat().st_size * prob for p, prob in bam_list}
    for row in sorted(bam_list, key=lambda row: (-costs[row[0]], row[0])):
        slot = min((i for i in range(workers) if len(bins[i]) < quotas[i]), key=lambda i: (loads[i], i))
        bins[slot].append(row)
        loads[slot] += costs[row[0]]
    rng = random.Random(seed)
    for group in bins:
        rng.shuffle(group)
    return [group[j] for j in range(max(quotas)) for group in bins if j < len(group)]


def split_bam_list(input_path, output_dir, fractions, seed, calibration_only=None):
    """Write new JSON splits and a manifest; fail on missing/duplicate BAMs.

    Fractions must be positive and sum to one. Allocation is source-stratified
    with largest-remainder file quotas and greedy effective-byte balancing.
    An optional separate calibration-only list is checked against every base
    input by path/inode and assigned wholly to calibration, outside the
    fractional allocation. Original read-sampling probabilities are preserved. Very small sources
    cannot appear everywhere and are reported. Existing outputs are refused.
    """
    if not fractions or any(v <= 0 for v in fractions.values()) or not math.isclose(sum(fractions.values()), 1):
        raise ValueError("Split fractions must be positive and sum to one")
    input_path, output_dir = Path(input_path), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f"Refusing to overwrite split directory: {output_dir}")
    source_text = input_path.read_text()
    groups, seen_paths, seen_inodes = defaultdict(list), set(), set()
    for line in source_text.splitlines():
        if not line.strip() or line.lstrip().startswith("#"):
            continue
        parts = line.strip().split("|")
        path = Path(parts[0]).resolve(strict=True)
        probability = float(parts[1]) if len(parts) > 1 and parts[1] else 1.0
        if not 0 < probability <= 1:
            raise ValueError(f"Invalid sampling probability for {path}")
        stat = path.stat()
        inode = (stat.st_dev, stat.st_ino)
        if str(path) in seen_paths or inode in seen_inodes:
            raise ValueError(f"Duplicate BAM path/inode: {path}")
        seen_paths.add(str(path))
        seen_inodes.add(inode)
        groups[str(path.parent)].append((str(path), probability, stat.st_size))
    names = list(fractions)
    assigned = {name: [] for name in names}
    manifest = {"input": str(input_path.resolve()), "input_sha256": hashlib.sha256(source_text.encode()).hexdigest(),
                "seed": seed, "fractions": fractions, "unit": "whole BAM path/inode",
                "limitations": ["No exhaustive read-ID audit; no locus/run disjointness claim.",
                                "Historical backbone pretraining exposure is not removed by these new splits."],
                "small_sources": [], "sources": {}, "splits": {}}
    rng = random.Random(seed)
    for source, rows in sorted(groups.items()):
        quotas = {n: int(len(rows) * fractions[n]) for n in names}
        remainders = sorted(names, key=lambda n: (-(len(rows) * fractions[n] - quotas[n]), names.index(n)))
        for n in remainders[:len(rows) - sum(quotas.values())]:
            quotas[n] += 1
        if len(rows) < len(names):
            manifest["small_sources"].append({"source": source, "files": len(rows), "allocation": quotas})
        loads, counts = {n: 0.0 for n in names}, {n: 0 for n in names}
        rng.shuffle(rows)
        rows.sort(key=lambda row: -row[1] * row[2])
        total = sum(p * size for _, p, size in rows)
        for row in rows:
            name = min((n for n in names if counts[n] < quotas[n]),
                       key=lambda n: loads[n] / max(total * fractions[n], 1))
            assigned[name].append(row)
            loads[name] += row[1] * row[2]
            counts[name] += 1
        manifest["sources"][source] = {"files": counts, "effective_bytes": loads}
    if calibration_only is not None:
        extra_path = Path(calibration_only)
        extra_text = extra_path.read_text()
        extra_count = 0
        if "calibration" not in assigned:
            raise ValueError("Calibration-only input requires a calibration split")
        for line in extra_text.splitlines():
            if not line.strip() or line.lstrip().startswith("#"):
                continue
            parts = line.strip().split("|")
            path = Path(parts[0]).resolve(strict=True)
            probability = float(parts[1]) if len(parts) > 1 and parts[1] else 1.0
            stat = path.stat()
            inode = (stat.st_dev, stat.st_ino)
            if str(path) in seen_paths or inode in seen_inodes:
                raise ValueError(f"Calibration-only BAM overlaps another input: {path}")
            if not 0 < probability <= 1:
                raise ValueError(f"Invalid sampling probability for {path}")
            seen_paths.add(str(path))
            seen_inodes.add(inode)
            assigned["calibration"].append((str(path), probability, stat.st_size))
            extra_count += 1
            source = manifest["sources"].setdefault(str(path.parent), {
                "files": dict.fromkeys(names, 0), "effective_bytes": dict.fromkeys(names, 0.0)})
            source["files"]["calibration"] += 1
            source["effective_bytes"]["calibration"] += probability * stat.st_size
        manifest["calibration_only"] = {
            "input": str(extra_path.resolve()), "files": extra_count,
            "input_sha256": hashlib.sha256(extra_text.encode()).hexdigest(),
            "exposure": "Absent from supplied main list; historical model exposure not audited.",
        }
    output_dir.mkdir(parents=True)
    for name, rows in assigned.items():
        rng.shuffle(rows)
        config = {"config_type": "pretraining", "schema_version": 1,
                  "samples": [{"bam_path": p, "sampling_probability": prob} for p, prob, _ in rows]}
        (output_dir / f"{name}.json").write_text(json.dumps(config, indent=2) + "\n")
        manifest["splits"][name] = {"files": len(rows), "bytes": sum(s for _, _, s in rows),
                                    "effective_bytes": sum(prob * s for _, prob, s in rows)}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def main():
    """Create purpose splits with all allocation decisions explicitly supplied."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--fractions", required=True, help="JSON object mapping purpose names to fractions")
    parser.add_argument("--calibration-only", help="Additional disjoint BAM list reserved entirely for calibration")
    args = parser.parse_args()
    result = split_bam_list(args.input, args.output_dir, json.loads(args.fractions), args.seed, args.calibration_only)
    print(json.dumps({"splits": result["splits"], "small_sources": result["small_sources"]}, indent=2))


if __name__ == "__main__":
    main()
