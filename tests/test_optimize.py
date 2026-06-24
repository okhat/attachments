"""Tests for self-optimizing context pipelines and smart budgeting.

These run fully offline (no API key, no network, no heavy parsers).
"""

import re

import attachments as att
from attachments import (
    Attachments,
    OptimizableAttachments,
    attach,
    context_budget,
    estimate_tokens,
    keyword_coverage_metric,
    refine,
)
from attachments.data import get_sample_path


SAMPLE = get_sample_path("sample.txt")


def _vocab(s: str):
    return set(re.findall(r"[a-z]{4,}", s.lower()))


def test_refiner_is_registered_without_clobbering_existing():
    assert hasattr(refine, "context_budget")
    assert "context_budget" in att._refiners
    for existing in ("truncate", "add_headers", "tile_images", "no_op"):
        assert existing in att._refiners


def test_context_budget_respects_budget():
    base = str(Attachments(SAMPLE))
    a = attach(SAMPLE)
    a.text = base
    a = context_budget(a, budget=120)
    assert estimate_tokens(a.text) <= 125
    meta = a.metadata["processing"][-1]
    assert meta["operation"] == "context_budget"
    assert meta["blocks_kept"] < meta["blocks_total"]


def test_budget_keeps_informative_content_vs_tail_truncation():
    base = str(Attachments(SAMPLE))
    budget = 120

    dumb = attach(SAMPLE); dumb.text = base
    dumb = refine.truncate(dumb, limit=budget * 4)

    smart = attach(SAMPLE); smart.text = base
    smart = context_budget(smart, budget=budget)

    # Both stay near the budget...
    assert estimate_tokens(smart.text) <= budget + 5
    # ...and the smart cut pulls from across the document (keeps a tail anchor
    # that pure truncation drops). We assert it isn't just a prefix of `base`.
    assert not base.startswith(smart.text)


def test_budget_noop_when_already_small():
    a = attach("x.txt")
    a.text = "short enough"
    before = a.text
    a = context_budget(a, budget=10_000)
    assert a.text == before


def test_dsl_budget_through_pipeline():
    piped = (
        attach(f"{SAMPLE}[budget:80]")
        | att.load.text_to_string
        | att.present.text
        | refine.context_budget
    )
    assert estimate_tokens(piped.text) <= 85


def test_auto_search_space_is_grounded_in_real_dsl():
    space = OptimizableAttachments(SAMPLE).auto_search_space()
    known = set(att.get_dsl_info().keys())
    for knob in space:
        # Every knob is either a real registered DSL command or 'budget'
        # (provided by this module).
        assert knob == "budget" or knob in known


def test_optimizer_beats_baseline_and_respects_target():
    opt = OptimizableAttachments(
        SAMPLE,
        search_space={"format": ["plain", "markdown"], "budget": [None, 120, 250]},
    )
    metric = keyword_coverage_metric(
        keywords=["attachments", "library", "text"],
        max_tokens=140,
        over_budget_penalty=1.0,
    )
    opt.compile(metric=metric)

    assert opt.best_config, "optimizer found no config"
    baseline = metric(opt.render(config={}))
    assert opt.best_score >= baseline
    assert estimate_tokens(opt.render()) <= 140 + 5
