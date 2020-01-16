
This directory provides wrapper for
warp-ctc from https://github.com/baidu-research/warp-ctc.

warp-ctc uses the same license as Kaldi, Apache License 2.0.

Although warp-ctc has not been updated for a long time, it
is still widely used. For example, espnet is still using
it (https://github.com/espnet/espnet/issues/1434).

When it comes to PyTorch, we may switch to its built-in
`torch.nn.CTCLoss` (https://pytorch.org/docs/stable/nn.html#torch.nn.CTCLoss)
if we find it faster than warp-ctc. This needs some benchmarks.

Note that this wrapper has no dependencies on PyTorch.
