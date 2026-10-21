# Supervised classifier alignment and confidence

## Reuse the pretrained alignment

For a new cross-attention classifier initialized with `--pretrain_load_path`,
add:

```bash
--classifier_type transformer_cross \
--classifier_center_source backbone
```

The default `--classifier_center_source head` learns a new center predictor
during supervised fine-tuning.
`backbone` consumes the `u_center` already returned by the final base-to-signal
cross-attention layer and creates no classifier center predictor. The centers
have shape `(batch, base_tokens)` and use signal-token coordinates, not raw
sample or reference-genome coordinates. A backbone with only one base-to-signal
layer returns static length-ratio centers; reuse uses those same returned values.

Both `--classifier_signal_source fixed_sa_ts` and `contextual_ts` preserve the
signal-token axis and support reuse. Centers supply the distance penalty in
attention scores; they do not force attention to one signal position or reuse
the backbone's attention weights. The classifier still learns its query/key/
value projections, feed-forward network, optional mixer, and output layers.

The center input is detached. This prevents an additional classifier gradient
through the reused center tensor, while ordinary gradients through base/signal
embeddings follow the selected training policy. Use `--freeze_pretrain` to hold
the entire backbone fixed; otherwise its centers can evolve as its weights
change during fine-tuning or auxiliary pretraining.

The policy is recorded in `classifier_config.center_source` and the execution summary.
Inference restores it automatically. Continuation with `--model_load_path`
preserves it and rejects explicit policy changes. The option applies only
to `transformer_cross`.

## Classifier precision and logit regularization

See [classifier numerical stability](classifier-numerical-stability.md) for
the float32 final projections and losses, raw-logit penalty, and checkpoint
metadata that records these policies.

## Interpret focal-loss probabilities

`porefm.common.utils.focal_loss_fn` first smooths targets toward uniform and
then applies the per-class focal factor and alpha weight:

```text
q_k = (1 - epsilon) * target_k + epsilon / K
loss = -sum_k alpha_k * q_k * (1 - p_k)^gamma * log(p_k)
```

Unlike ordinary cross entropy, this does not generally minimize at `p=q`.
Even a small nonzero target on the incorrect class remains influential because
its focal factor is large, while the correct class term is strongly downweighted
as its predicted probability rises. Thus smoothing epsilon does not define the
expected distance of a focal-loss confidence peak from zero or one. This is
consistent with the probability-estimation analysis in
[Charoenphakdee et al., CVPR 2021](https://arxiv.org/abs/2011.09172).

For a controlled probability comparison, keep the same data, backbone training
policy, and head while comparing cross entropy against focal loss. With focal
loss, explicitly test zero smoothing for all classification sources. JSON sample
settings override config settings, which override `--label_smoothing`, so a
zero CLI fallback alone does not disable JSON overrides. Nonzero soft targets
in label files still matter when the resolved epsilon is zero. See
[label-smoothing settings](reference/training-inputs.md#classification-label-smoothing).
Assess held-out log loss, Brier
score, reliability by confidence bin, and discrimination. Fit any post-training
calibration on a separate representative validation set.

## Assess head capacity

The cross head adds an attention block and inherits the backbone hidden width,
attention-head count, and feed-forward width. `--fc_dim_ff 32` only limits the
final classification MLP; it does not shrink the attention block. Supervised
training creates the head with zero dropout. Alignment reuse removes one
predictor but leaves most of the head's capacity intact.

Compare CNN, cross attention with its own centers, and cross attention with
reused centers using identical training data and loss settings. Start with a
frozen backbone when isolating head capacity. Compare training and held-out
performance, and hold out biological sources or sequence contexts where
generalization is important. Overlapping chunks from the same reads/loci must
not cross the split. Compressed probability peaks alone do not establish
overfitting; a growing training/validation gap is more informative.
