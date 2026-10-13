"""Pure metric checks; no model download or Chroma index required."""

from eval import em_metric, f1_metric, hallucination_metric, source_hit_metric


def test_simple_evaluation_helpers():
    assert em_metric(" 정답 ", ["정답"]) == 1
    assert f1_metric("a b", ["a b"]) == 1.0
    assert source_hit_metric("근거 p.3", "2,3") == 1
    assert hallucination_metric(0, 0.0, 0) == 1
