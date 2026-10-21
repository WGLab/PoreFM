#!/usr/bin/env python3
"""Run the same anomaly detection on stored embedding blocks.

This script consumes the embedding files written by dual_stream_metrics_optimized.py
when run with --store_embeddings. It does not read BAM files and does not run model
inference. The heavy per-block tensor math can run on CPU or CUDA depending on
--device.
"""

from __future__ import annotations

import argparse
import datetime
import os
from pathlib import Path
import re
import sys
import time
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, ThreadPoolExecutor, wait

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import torch

from porefm.cli._parser import PoreFMArgumentParser
from porefm.workflows.anomaly.detection import process_block_anomaly_detection_vectorized


def natural_key(path: Path):
    """Sort chr2 before chr10 while preserving normal lexical behavior elsewhere."""
    return [int(x) if x.isdigit() else x.lower() for x in re.split(r"(\d+)", str(path))]


def resolve_embeddings_dir(path: str | os.PathLike) -> Path:
    """Accept either <output> or <output>/embeddings."""
    p = Path(path)
    if (p / "embeddings").is_dir():
        child = p / "embeddings"
        if any(child.glob("*.pt")) or any(child.glob("*.npz")):
            return child
    return p


def iter_embedding_files(embeddings_dir: Path, fmt: str = "auto", recursive: bool = False):
    """Find stored embedding blocks in deterministic natural-sort order.

    Parameters
    ----------
    embeddings_dir : pathlib.Path
        Directory containing ``.pt`` or ``.npz`` blocks.
    fmt : {"auto", "pt", "npz"}, default="auto"
        File format to include. ``"auto"`` includes both supported formats.
    recursive : bool, default=False
        Search descendant directories instead of only ``embeddings_dir``.

    Returns
    -------
    list[pathlib.Path]
        Unique paths sorted so numeric chromosome components sort numerically.
    """
    if fmt not in {"auto", "pt", "npz"}:
        raise ValueError("fmt must be one of: auto, pt, npz")

    patterns = []
    if fmt in {"auto", "pt"}:
        patterns.append("*.pt")
    if fmt in {"auto", "npz"}:
        patterns.append("*.npz")

    files = []
    for pattern in patterns:
        files.extend(embeddings_dir.rglob(pattern) if recursive else embeddings_dir.glob(pattern))

    # Avoid duplicate Path entries when auto is used.
    return sorted(set(files), key=natural_key)


def torch_load_compat(path: Path):
    """Load a torch-saved block, including numpy readname arrays on newer PyTorch."""
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")


def _torch_dtype_from_string(dtype_string: str):
    if not dtype_string.startswith("torch."):
        raise ValueError(f"Unsupported stored tensor dtype metadata: {dtype_string}")
    dtype_name = dtype_string.split(".", 1)[1]
    if not hasattr(torch, dtype_name):
        raise ValueError(f"Unknown torch dtype in stored metadata: {dtype_string}")
    return getattr(torch, dtype_name)


def _load_npz_stream(data, prefix: str):
    positions = torch.from_numpy(np.asarray(data[f"{prefix}_position"]))
    strands = torch.from_numpy(np.asarray(data[f"{prefix}_strand"]))
    readnames = np.asarray(data[f"{prefix}_readname"]).astype(str, copy=False)

    emb_payload = np.asarray(data[f"{prefix}_embeddings"])
    emb_dtype = str(np.asarray(data[f"{prefix}_embedding_dtype"]).item())
    emb_shape = tuple(int(x) for x in np.asarray(data[f"{prefix}_embedding_shape"]))

    if emb_dtype == "torch.bfloat16_as_uint16":
        # npz stored bfloat16 as exact raw uint16 payload to avoid lossy conversion.
        embeddings = torch.from_numpy(np.ascontiguousarray(emb_payload)).view(torch.bfloat16).reshape(emb_shape)
    else:
        embeddings = torch.from_numpy(np.ascontiguousarray(emb_payload)).reshape(emb_shape)
        expected_dtype = _torch_dtype_from_string(emb_dtype)
        if embeddings.dtype != expected_dtype:
            embeddings = embeddings.to(expected_dtype)

    return {
        "positions": positions,
        "embeddings": embeddings,
        "strands": strands,
        "readnames": readnames,
    }


def load_embedding_block(path: str | os.PathLike):
    """Load one stored matched WT/control embedding block.

    Supports both formats produced by the writer:
      .pt  : --embedding_write_mode fast
      .npz : --embedding_write_mode compressed
    """
    path = Path(path)
    suffix = path.suffix.lower()

    if suffix == ".pt":
        block = torch_load_compat(path)
        required = {"chrom", "block_start", "wt", "control"}
        missing = required - set(block)
        if missing:
            raise ValueError(f"{path} is missing required keys: {sorted(missing)}")
        return block

    if suffix == ".npz":
        with np.load(path, allow_pickle=False) as data:
            return {
                "chrom": str(np.asarray(data["chrom"]).item()),
                "block_start": int(np.asarray(data["block_start"]).item()),
                "wt": _load_npz_stream(data, "wt"),
                "control": _load_npz_stream(data, "control"),
            }

    raise ValueError(f"Unsupported embedding block extension for {path}; expected .pt or .npz")


def safe_chrom_name(chrom) -> str:
    """Convert a chromosome identifier to a single safe path component."""
    return str(chrom).replace(os.sep, "_")


def output_path_for_block(output_dir: Path, chrom, block_start: int) -> Path:
    """Return the parquet path assigned to one chromosome block."""
    return output_dir / f"{safe_chrom_name(chrom)}_{int(block_start)}.parquet"


def process_embedding_file(
    embedding_path: str,
    output_dir: str,
    anomaly_kwargs: dict,
    overwrite: bool = False,
    skip_existing: bool = True,
):
    """Run anomaly detection for one stored WT/control embedding block.

    The input is loaded on CPU and passed to
    :func:`process_block_anomaly_detection_vectorized`. A non-empty result is
    written as a parquet table containing read name, genomic position,
    p-value, strand, and chromosome columns. Empty blocks do not create a
    parquet file.

    Parameters
    ----------
    embedding_path : str
        Path to a ``.pt`` or ``.npz`` embedding block.
    output_dir : str
        Directory for the per-block parquet output.
    anomaly_kwargs : dict
        Keyword arguments forwarded to the anomaly detector.
    overwrite : bool, default=False
        Recompute an existing output instead of preserving it.
    skip_existing : bool, default=True
        Return a ``"skipped_existing"`` result when the destination exists.
        This only applies when ``overwrite`` is false.

    Returns
    -------
    dict
        Processing metadata with input/output paths, block coordinates,
        status, and row count. The row count is ``None`` for skipped blocks.
    """
    embedding_path = Path(embedding_path)
    output_dir = Path(output_dir)

    block = load_embedding_block(embedding_path)
    chrom = block["chrom"]
    block_start = int(block["block_start"])
    out_path = output_path_for_block(output_dir, chrom, block_start)

    if out_path.exists() and skip_existing and not overwrite:
        return {
            "path": str(embedding_path),
            "output": str(out_path),
            "chrom": chrom,
            "block_start": block_start,
            "status": "skipped_existing",
            "n_rows": None,
        }

    result = process_block_anomaly_detection_vectorized(
        block["control"],
        block["wt"],
        **anomaly_kwargs,
    )

    n_rows = len(result["positions"])
    if n_rows > 0:
        table = pa.Table.from_pydict({
            "readname": result["readnames"],
            "position": result["positions"],
            "p_val": result["p_vals"],
            "strand": result["strands"],
            "chrom": [chrom] * n_rows,
        })
        pq.write_table(table, out_path)
        status = "written"
    else:
        status = "empty"

    return {
        "path": str(embedding_path),
        "output": str(out_path),
        "chrom": chrom,
        "block_start": block_start,
        "status": status,
        "n_rows": n_rows,
    }


def choose_device(device_arg: str) -> str:
    """Resolve ``"auto"`` and reject unavailable CUDA requests."""
    if device_arg == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device_arg.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(f"{device_arg!r} was requested, but CUDA is not available")
    return device_arg


def drain_completed(pending, block: bool = False):
    """Collect completed futures and remove them from a mutable pending set.

    Parameters
    ----------
    pending : set[concurrent.futures.Future]
        Futures awaiting collection. Completed entries are removed in place.
    block : bool, default=False
        Wait for all pending futures when true; otherwise return only futures
        that are ready without waiting.
    """
    if not pending:
        return []
    if block:
        done, _ = wait(pending)
    else:
        done, _ = wait(pending, timeout=0, return_when=FIRST_COMPLETED)
    results = []
    for future in done:
        pending.remove(future)
        results.append(future.result())
    return results


def print_result(result):
    """Print a one-line summary returned by :func:`process_embedding_file`."""
    n_rows = result["n_rows"]
    rows_text = "NA" if n_rows is None else str(n_rows)
    print(
        f"{result['status']}: {result['chrom']}:{result['block_start']} rows={rows_text} -> {result['output']}",
        flush=True,
    )


def run(args):
    """Process all selected embedding blocks according to parsed CLI options.

    This function creates the output directory, optionally seeds NumPy and
    PyTorch, and applies bounded task submission. CPU jobs use worker
    processes, whereas CUDA jobs use threads so workers share one CUDA
    context.

    Parameters
    ----------
    args : argparse.Namespace
        Arguments produced by :func:`build_arg_parser`.

    Raises
    ------
    FileNotFoundError
        If no matching embedding blocks are found.
    ValueError
        If fewer than one worker is requested.
    """
    embeddings_dir = resolve_embeddings_dir(args.embeddings_dir)
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = choose_device(args.device)
    if args.random_seed is not None:
        torch.manual_seed(args.random_seed)
        np.random.seed(args.random_seed)

    files = iter_embedding_files(embeddings_dir, fmt=args.format, recursive=args.recursive)
    if args.limit is not None:
        files = files[: args.limit]

    if not files:
        raise FileNotFoundError(f"No embedding block files found in {embeddings_dir}")

    if args.workers < 1:
        raise ValueError("--workers must be >= 1")

    anomaly_kwargs = dict(
        min_control_cov=args.min_control_cov,
        min_test_cov=args.min_test_cov,
        max_fit_cov=args.max_fit_cov,
        n_comp=args.n_comp,
        batch_size=args.anomaly_batch_size,
        pvalue_method=args.pvalue_method,
        calibration_fraction=args.calibration_fraction,
        device=device,
    )

    print("\n%s:\nCommand: python %s\n" % (str(datetime.datetime.now()), " ".join(sys.argv)), flush=True)
    print(f"Embedding blocks: {len(files)}", flush=True)
    print(f"Input dir: {embeddings_dir}", flush=True)
    print(f"Output dir: {output_dir}", flush=True)
    print(f"Anomaly device: {device}", flush=True)

    start = time.time()

    # CUDA work is run through threads when workers > 1 to avoid repeatedly
    # initializing independent CUDA contexts in child processes. CPU work uses
    # processes to bypass the GIL and parallelize linear algebra over blocks.
    executor_cls = ThreadPoolExecutor if str(device).startswith("cuda") else ProcessPoolExecutor
    max_pending = args.max_pending_tasks or max(2 * args.workers, args.workers + 1)

    if args.workers == 1:
        for path in files:
            result = process_embedding_file(
                str(path),
                str(output_dir),
                anomaly_kwargs,
                overwrite=args.overwrite,
                skip_existing=not args.overwrite,
            )
            print_result(result)
    else:
        pending = set()
        with executor_cls(max_workers=args.workers) as executor:
            for path in files:
                fut = executor.submit(
                    process_embedding_file,
                    str(path),
                    str(output_dir),
                    anomaly_kwargs,
                    args.overwrite,
                    not args.overwrite,
                )
                fut.block_path = str(path)
                pending.add(fut)

                if len(pending) >= max_pending:
                    for result in drain_completed(pending, block=True):
                        print_result(result)
                else:
                    for result in drain_completed(pending, block=False):
                        print_result(result)

            for result in drain_completed(pending, block=True):
                print_result(result)

    print(f"Total Time = {time.time() - start:.2f}s", flush=True)


def build_arg_parser():
    """Build the command-line parser for stored-embedding anomaly detection."""
    parser = PoreFMArgumentParser(
        description="Run anomaly detection from stored WT/control embedding blocks without reading BAMs or running model inference."
    )

    parser.add_argument("--embeddings_dir", type=str, required=True, help="Directory containing .pt/.npz embedding blocks, or parent output dir containing embeddings/.")
    parser.add_argument("--output", type=str, required=True, help="Output directory for per-block parquet anomaly results.")
    parser.add_argument("--format", choices=["auto", "pt", "npz"], default="auto", help="Stored embedding file format to read.")
    parser.add_argument("--recursive", default=False, action="store_true", help="Search recursively under --embeddings_dir.")
    parser.add_argument("--limit", type=int, default=None, help="Process only the first N embedding files; useful for smoke tests.")

    parser.add_argument("--device", type=str, default="auto", help="Device for heavy anomaly tensor math: auto, cpu, cuda, cuda:0, etc.")
    parser.add_argument("--workers", type=int, default=1, help="Parallel block workers. CPU uses processes; CUDA uses threads.")
    parser.add_argument("--max_pending_tasks", type=int, default=None, help="Maximum queued block tasks before backpressure. Default is about 2x workers.")
    parser.add_argument("--overwrite", default=False, action="store_true", help="Recompute and overwrite existing parquet outputs. By default, existing outputs are skipped.")
    parser.add_argument("--random_seed", type=int, default=None, help="Optional CPU/GPU RNG seed before processing. The anomaly algorithm uses random reservoir ordering.")

    parser.add_argument("--min_control_cov", type=int, default=30)
    parser.add_argument("--min_test_cov", type=int, default=30)
    parser.add_argument("--max_fit_cov", type=int, default=256)
    parser.add_argument("--n_comp", type=int, default=2)
    parser.add_argument("--anomaly_batch_size", type=int, default=2000, help="Number of genomic positions per anomaly batch.")
    parser.add_argument(
        "--pvalue_method",
        choices=["chi2", "empirical"],
        default="chi2",
        help="Use chi-square or held-out empirical-control p-values.",
    )
    parser.add_argument(
        "--calibration_fraction",
        type=float,
        default=0.2,
        help=(
            "Per-locus control fraction held out for empirical p-values. The "
            "remaining fit-control coverage must satisfy --min_control_cov."
        ),
    )

    return parser


def main():
    """Parse command-line arguments and run the anomaly workflow."""
    parser = build_arg_parser()
    args = parser.parse_args()
    run(args)


if __name__ == "__main__":
    main()
