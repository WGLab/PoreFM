# Signal dropout during training

These options apply when training models. Running a released inference model
does not require changing its checkpoint configuration. For input JSON, see
[configuration formats](../configuration.md#pretraining-and-canonical-null-json).

## Signal-modality dropout during pretraining

Signal-modality dropout trains the model with one signal representation
occasionally withheld. During pretraining, `--signal_dropout_prob P`
independently chooses one action for each batch item: drop percentile features
with probability `P/2`, drop raw-signal features with probability `P/2`, or
keep both with probability `1-P`. The choice is broadcast across that batch
item. Inference and supervised classification use zero signal dropout.

Choose how the EMA target encoder handles dropout with
`--ema_signal_dropout_mode`:

- `independent`: sample separate batch-item choices for the student and EMA
  target;
- `shared`: reuse each student's choice for the corresponding EMA batch item;
- `disabled` (default): apply configured dropout to the student while keeping EMA target
  modality dropout off.

Checkpoints save both settings. `--use_loaded_config` and `--resume` inherit
them when the corresponding CLI options are omitted. Resume requires the
saved settings; a warm start without `--resume` may override them. Evaluation
uses zero signal dropout.

## Auxiliary pretraining during fine-tuning

With `--train_pretrain_tasks`, auxiliary batches may use an explicit value with
`--pretrain_signal_dropout_prob`, or use the checkpoint value with
`--use_loaded_pretrain_signal_dropout_prob`. The EMA mode inherits from the
checkpoint unless `--pretrain_ema_signal_dropout_mode` overrides it. These
options are rejected when auxiliary pretraining is disabled.
