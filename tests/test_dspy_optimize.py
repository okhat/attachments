"""Tests for the DSPy wrapper around optimizable context pipelines.

These exercise the structural integration with DSPy (module, prediction,
teleprompter-style optimizer) without making any LM calls, so they run with
no API key. Skipped entirely if dspy is not installed.
"""

import pytest

dspy = pytest.importorskip("dspy")

from attachments.data import get_sample_path
from attachments.dspy import (
    AttachmentContext,
    ContextOptimizer,
    keyword_coverage_metric,
)

SAMPLE = get_sample_path("sample.txt")


def test_attachment_context_is_a_dspy_module():
    ctx = AttachmentContext(SAMPLE)
    assert isinstance(ctx, dspy.Module)
    pred = ctx()  # forward, no LM involved
    assert isinstance(pred, dspy.Prediction)
    assert isinstance(pred.context, str) and pred.context
    assert pred.tokens > 0


def test_config_changes_the_rendered_context():
    big = AttachmentContext(SAMPLE, config={})()
    small = AttachmentContext(SAMPLE, config={"budget": 80})()
    assert small.tokens < big.tokens


def test_context_optimizer_tunes_the_config():
    student = AttachmentContext(
        SAMPLE, search_space={"format": ["plain", "markdown"], "budget": [None, 120]}
    )
    metric = keyword_coverage_metric(
        keywords=["attachments", "library", "text"], max_tokens=140
    )
    tuned = ContextOptimizer(metric=metric).compile(student)

    # Returns a tuned copy with a config drawn from the search space.
    assert isinstance(tuned, AttachmentContext)
    assert tuned.config in student.candidates()
    assert hasattr(tuned, "_compiled_score")
    # The tuned module should be at least as good as the empty baseline.
    baseline = metric(AttachmentContext(SAMPLE, config={})().context)
    assert tuned._compiled_score >= baseline


def test_composes_as_submodule_of_a_dspy_program():
    """AttachmentContext slots into a larger program; MIPROv2 would optimize
    the predictor while ContextOptimizer optimizes the ingestion. We only check
    structural composition here (no LM call)."""

    class RAG(dspy.Module):
        def __init__(self):
            super().__init__()
            self.ingest = AttachmentContext(SAMPLE, config={"budget": 100})
            self.answer = dspy.ChainOfThought("context, question -> answer")

        def forward(self, question):
            ctx = self.ingest().context
            return self.answer(context=ctx, question=question)

    prog = RAG()
    # The predictor is discoverable by DSPy optimizers...
    predictor_names = [name for name, _ in prog.named_predictors()]
    assert any("answer" in n for n in predictor_names)
    # ...and the ingestion module produces context without an LM.
    assert prog.ingest().context
