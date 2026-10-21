# Run output safety

By default, the following commands require an absent or empty output directory,
or an absent output file. Hidden files, logs, and manifests count as existing results.
Checks run before model loading and before opening logs or result writers.

| Command | Protected destination |
| --- | --- |
| `pretrain` (including evaluation) | `--model_save_path` |
| `finetune-modification` (alias `supervised-finetune`) | `--model_save_path` |
| `detect-modification` (alias `infer`) | `--output` |
| `zero-shot train` / `zero-shot train-scale` | `--model_save_path` when training |
| `zero-shot calibrate` / `zero-shot fit-whitening` | `--calibration_output` / `--whitening_output` |
| `zero-shot detect` | `--output_dir` |
| `prototype extract-kmer-prototypes` (including union mode) | `--output`, when supplied |
| `prototype extract-effect-neighborhoods`, `prototype fit-effect-prototypes` | `--output` |
| `prototype classify` | `--output_dir` and the fitted classifier file; includes residual-effects scoring |
| `python -m porefm.cli.single_read_tag` | Active Parquet `--output` and/or BAM `--bam_output` |

Each output directory gets `run_manifest.json`. A standalone artifact also gets
`<filename>.run_manifest.json`, allowing other artifacts to share its parent
directory. Manifests record the invocation arguments, command, UTC time, host,
working directory, canonical output paths, Git revision and dirty flags, input
file sizes/mtimes, and JSON config snapshots with SHA-256 hashes. Large BAMs and
checkpoints are not hashed. These are startup records, not completion markers;
an interrupted run retains its reservation. Forced runs replace the manifests.

Supervised inference with direct sample options also records the normalized
sample configuration, including defaults, in `arguments.resolved_input`.
JSON input is captured as config snapshots. Both input routes
retain the model release identity when using `--model`.

Distributed training/inference prepares outputs once on rank zero and
broadcasts success or the error to every rank before writers or models start.
The checks also apply when the canonical-null pipeline invokes its training
and fitting stages directly. Its `--force-overwrite` flag is forwarded to
executed stages. The pipeline's `--skip_existing` stage reuse takes
precedence for completed stages; those stages are not executed or overwritten.

## Continuing training

Load a saved checkpoint using the command's loading/resume
options, and choose a new `--model_save_path`, or explicitly pass
`--force-overwrite` to reuse the existing directory. A checkpoint continuation
still counts as a new invocation. Pretraining's `--resume` and
`--resume_interrupted_log` retain their training/data-state semantics; they do
not grant permission to reuse an occupied output directory.

Keep redirected stdout/stderr logs outside the run directory. For example,
use a sibling `run-name.log` instead of `run-name/train.log`; the shell opens
the log before Python checks the output directory.

## Explicit replacement

With `--force-overwrite`, the occupancy check still runs. If any destinations
are occupied, a warning on stderr lists their paths before execution continues
and permits matching files and run manifests to be overwritten directly.
Fresh destinations produce no overwrite warning. Distributed runs print the
warning once on rank zero. No backup or directory cleanup is performed.
Unrelated files, stale result parts, and existing BAM indexes may remain; with
this flag, managing those files and avoiding concurrent writers is the user's
responsibility. The flag does not change the command's checkpoint-loading,
resume, or training behavior.

The flag does not bypass checks for output symlinks, file/directory collisions,
or destinations containing the working directory or repository. Direct output
file aliases of discovered inputs are also rejected. Loading
`old-run/model.100` while writing to `old-run` is permitted with the flag:
directory preparation leaves the checkpoint in place. Every destination is
checked before any run manifests are written.

## Other workflows

`pairwise-comparison` uses manifest-checked `--resume` behavior.
Data conversion tools, plotting commands, and analysis scripts have their own
output rules; the run-directory checks listed above do not apply to them.
Check the relevant command or API before reusing an output path.
