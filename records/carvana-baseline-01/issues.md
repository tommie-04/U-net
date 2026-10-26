# Non-finite Loss in Initial AMP Attempt

Source: user-provided cloud training log excerpt; not direct process inspection.

The original training entry ran with five planned epochs, batch size 1,
learning rate 1e-5, scale 0.5, validation 10%, and AMP. It reported 4,580
training and 508 validation samples. W&B offline run ID: 84hmb4ge.

The displayed batch loss was NaN at 20%, 40%, 60%, 80%, and completion of
epoch 1. Each displayed validation Dice was approximately 8.0642567e-12.
Epoch 1 displayed 06:17 elapsed, followed by a checkpoint-saved message.
The first point of numerical failure is unknown from this excerpt.
A saved checkpoint does not establish finite weights or valid learning.

Status: abnormal attempt; user instructed to stop the training child process
and retain logs/checkpoints. Actual stop and exit status remain unconfirmed.
No checkpoint from this attempt is accepted as a successful baseline.

Diagnosis pending: inspect checkpoint finiteness and numerical behavior.
AMP is a candidate to test, not a confirmed cause. Do not load this checkpoint
as the initial state of a clean retry or overwrite this attempt's evidence.

## Subsequent diagnostic evidence

The user reported 100 non-finite floating-point tensors in the epoch-1
checkpoint, including convolution weights and BatchNorm parameters/buffers.
The checkpoint is not accepted for inference or continuation.

A fresh FP32 model completed 10 updates on the first 10 sorted training images
at scale 0.5, using the original preprocessing, loss, and RMSprop settings.
Loss decreased from 1.157753 to 0.382148. Per-step checks found finite loss,
gradient norm, and model floating-point state; no out-of-memory error was reported.
This is a small diagnostic, not full training or a matched AMP comparison.
The root cause remains unconfirmed.

The user authorized preparing a new five-epoch FP32 run. The next command
archives the failed checkpoints and starts from random initialization, omitting
`--amp`; all other CLI settings remain unchanged. Actual launch is pending
user execution. The failed attempt remains separate from the retry.
