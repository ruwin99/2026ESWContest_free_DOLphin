from __future__ import annotations

import torch

from build_gpu_postprocess_wrapper import CandidatePostprocess


def test_candidate_wrapper_is_exact_and_ignores_crack_rows_above_112() -> None:
    logits = torch.randn(1, 5, 240, 1280)
    model = CandidatePostprocess()
    rust, crack_count = model(logits)
    assert torch.equal(rust, logits[:, :4].argmax(1))
    assert torch.equal(crack_count, (logits[:, 4, 112:240] >= 0).sum((1, 2)))
    changed = logits.clone()
    changed[:, 4, :112] += 1000
    assert torch.equal(model(changed)[1], crack_count)
