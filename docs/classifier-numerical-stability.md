# Classifier precision and logit regularization

Final classifier projections run in float32 in both training and inference,
even inside bfloat16 autocast. Transformer heads use float32 for their final
two-layer GELU projection; CNN heads use float32 for the final linear layer.
The backbone and the attention/convolutional head body use the configured
autocast policy.

Supervised cross-entropy, focal, exclusion, proportion, and logit-penalty
calculations use float32. `--logit_reg_lambda` weights
`mean(logits.square())` over all retained bases and classes. The raw penalty equals the centered penalty plus the
mean squared per-base logit offset. It therefore constrains growth in a shared
offset that leaves ideal softmax probabilities unchanged but can erase class
differences when low-precision logits become large. Changing the penalty
coefficient is not a substitute for computing the projection and loss with
sufficient precision.

Supervised checkpoints record `logit_regularization: raw_mean_square`,
`classifier_projection_dtype: float32`, and `supervised_loss_dtype: float32`
in their training metadata. A coefficient of zero disables the penalty.
Assess numerical stability on representative data when choosing training settings.
