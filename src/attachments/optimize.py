"""
Self-optimizing context pipelines + smart context budgeting.
====================================================================

Two **additive** features. Both run fully offline (no API key, no extra deps):

1. ``refine.context_budget``  —  DSL: ``[budget:N]``
   Importance-aware trimming. Where ``refine.truncate`` chops the tail, this
   scores the blocks of a document and keeps the most informative ones that fit
   within a token budget, re-emitting them in their original order. "Smart
   trimming instead of dumb trimming."

2. ``OptimizableAttachments``
   Treats the Attachments DSL as a *search space* and tunes the
   context-construction choices (format, trimming, budget, ...) against a
   user-supplied metric — DSPy-style "let the machine pick the settings, not
   the human." The search space is discovered from the library's own DSL
   introspection (:func:`attachments.dsl_info.get_dsl_info`), so it stays
   grounded in the real, registered commands.

Nothing in this module mutates existing behaviour; it only registers one new
refiner and adds one new class.
"""

from __future__ import annotations

import itertools
import math
import re
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple

from .core import Attachment, attach, refiner


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

# A tiny stop-word list so block scoring favours content-bearing words.
_STOPWORDS = {
    "the", "a", "an", "and", "or", "but", "if", "then", "else", "for", "of",
    "to", "in", "on", "at", "by", "is", "are", "was", "were", "be", "been",
    "being", "it", "its", "this", "that", "these", "those", "as", "with",
    "from", "into", "about", "your", "you", "we", "our", "they", "their",
    "can", "will", "would", "could", "should", "may", "might", "has", "have",
    "had", "do", "does", "did", "not", "no", "yes", "so", "such", "than",
    "there", "here", "how", "what", "which", "who", "when", "where", "why",
}

_WORD_RE = re.compile(r"[A-Za-z0-9][A-Za-z0-9'_-]*")


def estimate_tokens(text: str) -> int:
    """Cheap, model-agnostic token estimate (~4 chars/token).

    Deliberately dependency-free so budgeting works without a tokenizer.
    """
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def _content_words(text: str) -> List[str]:
    return [
        w.lower()
        for w in _WORD_RE.findall(text)
        if len(w) > 2 and w.lower() not in _STOPWORDS
    ]


def _split_blocks(text: str) -> List[str]:
    """Split text into paragraph-like blocks on blank lines."""
    blocks = re.split(r"\n\s*\n", text.strip())
    return [b.strip() for b in blocks if b.strip()]


def _is_header(block: str) -> bool:
    first_line = block.lstrip().split("\n", 1)[0]
    return first_line.startswith("#") or bool(re.match(r"^[A-Z][^\n]{0,60}:$", first_line))


def _score_block(block: str, index: int, doc_term_freq: Dict[str, int]) -> float:
    """Importance score for a block. Higher = more worth keeping.

    Combines: information density (rare-ish content words), presence of
    numbers/facts, a small intro bonus for the first block, and a strong
    structural bonus for headers. Normalised by sqrt(length) so we prefer
    dense short blocks over long filler.
    """
    words = _content_words(block)
    if not words:
        return 0.0

    # Information content: words weighted by how often they recur in the doc
    # (recurring terms signal the document's main subject).
    info = sum(1.0 + math.log1p(doc_term_freq.get(w, 1)) for w in set(words))

    digit_bonus = 1.5 * len(re.findall(r"\d", block)) ** 0.5
    intro_bonus = 2.0 if index == 0 else 0.0
    header_bonus = 5.0 if _is_header(block) else 0.0

    length_norm = math.sqrt(max(1, len(words)))
    return (info + digit_bonus) / length_norm + intro_bonus + header_bonus


def _parse_budget(raw: Any) -> Tuple[int, str]:
    """Parse a budget value -> (amount, unit). Units: 'tok' (default) or 'char'."""
    if raw is None:
        return 0, "tok"
    s = str(raw).strip().lower()
    unit = "tok"
    if s.endswith("char") or s.endswith("chars"):
        unit = "char"
        s = re.sub(r"chars?$", "", s)
    elif s.endswith("tok") or s.endswith("tokens"):
        unit = "tok"
        s = re.sub(r"tok(ens)?$", "", s)
    s = s.strip()
    try:
        return max(0, int(float(s))), unit
    except (ValueError, TypeError):
        return 0, "tok"


def _cost(text: str, unit: str) -> int:
    return len(text) if unit == "char" else estimate_tokens(text)


# ---------------------------------------------------------------------------
# Feature 3: smart context budgeting (registered refiner + DSL [budget:N])
# ---------------------------------------------------------------------------

@refiner
def context_budget(att: Attachment, budget: Any = None) -> Attachment:
    """Keep the most informative blocks of text that fit within a budget.

    DSL: ``file.txt[budget:500]`` (~500 tokens) or ``[budget:2000char]``.

    Unlike ``refine.truncate`` (which cuts the end), this ranks blocks by an
    offline importance heuristic, keeps the best ones that fit, and re-emits
    them in their original order with a marker where content was dropped.
    """
    if budget is None:
        budget = att.commands.get("budget")
    amount, unit = _parse_budget(budget)
    if amount <= 0 or not att.text:
        return att

    full_cost = _cost(att.text, unit)
    if full_cost <= amount:
        return att  # already fits — nothing to do

    blocks = _split_blocks(att.text)
    if len(blocks) <= 1:
        # Can't rank a single block; fall back to a clean character cut.
        if unit == "char":
            att.text = att.text[:amount].rstrip() + " […]"
        else:
            att.text = att.text[: amount * 4].rstrip() + " […]"
        return att

    # Document-level term frequencies for scoring.
    doc_tf: Dict[str, int] = {}
    for b in blocks:
        for w in _content_words(b):
            doc_tf[w] = doc_tf.get(w, 0) + 1

    scored = [
        (i, _score_block(b, i, doc_tf), _cost(b, unit))
        for i, b in enumerate(blocks)
    ]

    # Greedy knapsack-ish: take highest score-per-cost first until budget is hit.
    order = sorted(scored, key=lambda t: t[1] / max(1, t[2]), reverse=True)
    kept: set = set()
    used = 0
    for i, _score, cost in order:
        if used + cost <= amount:
            kept.add(i)
            used += cost
    if not kept:  # budget smaller than any block: keep the single best block
        kept.add(max(scored, key=lambda t: t[1])[0])

    # Re-emit in original order, marking gaps where blocks were dropped.
    pieces: List[str] = []
    prev_kept = True
    for i, b in enumerate(blocks):
        if i in kept:
            if not prev_kept:
                pieces.append("[…]")
            pieces.append(b)
            prev_kept = True
        else:
            prev_kept = False
    if not prev_kept:
        pieces.append("[…]")

    new_text = "\n\n".join(pieces)
    att.metadata.setdefault("processing", []).append(
        {
            "operation": "context_budget",
            "unit": unit,
            "budget": amount,
            "blocks_total": len(blocks),
            "blocks_kept": len(kept),
            "cost_before": full_cost,
            "cost_after": _cost(new_text, unit),
        }
    )
    att.text = new_text
    return att


# ---------------------------------------------------------------------------
# Feature 1: self-optimizing context pipeline
# ---------------------------------------------------------------------------

# Curated candidate values for text-relevant DSL knobs. These names are
# validated against the library's live DSL introspection at runtime, so the
# search space can only ever contain commands the library actually supports.
_DEFAULT_CANDIDATES: Dict[str, List[Any]] = {
    "format": ["plain", "markdown"],
    "images": ["false"],
    "truncate": [None, 1500],
    "budget": [None, 200, 500],
}


def _known_dsl_commands() -> set:
    try:
        from .dsl_info import get_dsl_info

        return set(get_dsl_info().keys())
    except Exception:
        return set()


class OptimizableAttachments:
    """Tune Attachments' context-construction choices against a metric.

    Example (fully offline)::

        opt = OptimizableAttachments("report.txt")
        opt.compile(metric=my_metric)     # tries configs, keeps the best
        context = opt.render()            # best context as a string

    The metric is any callable ``metric(context_text) -> float`` (higher is
    better). Because the metric is just a function, the optimiser runs with no
    API key; plug in a model-backed metric when you have one.
    """

    def __init__(
        self,
        *paths: str,
        search_space: Optional[Dict[str, List[Any]]] = None,
    ) -> None:
        if not paths:
            raise ValueError("OptimizableAttachments needs at least one path.")
        self.paths: Tuple[str, ...] = paths
        self.search_space: Dict[str, List[Any]] = (
            search_space if search_space is not None else self.auto_search_space()
        )
        self.best_config: Dict[str, Any] = {}
        self.best_score: float = float("-inf")
        self.trials: List[Dict[str, Any]] = []

    # -- search space ----------------------------------------------------
    def auto_search_space(self) -> Dict[str, List[Any]]:
        """Build a search space from curated knobs, grounded in the real DSL.

        Any knob the installed library doesn't actually expose is dropped, so
        the space is always valid. ``budget`` is always allowed (it's provided
        by this module).
        """
        known = _known_dsl_commands()
        space: Dict[str, List[Any]] = {}
        for knob, values in _DEFAULT_CANDIDATES.items():
            if knob == "budget" or not known or knob in known:
                space[knob] = list(values)
        return space

    def candidates(self, max_trials: int = 32) -> List[Dict[str, Any]]:
        """Enumerate configs (cartesian product of the search space), capped."""
        keys = list(self.search_space.keys())
        value_lists = [self.search_space[k] for k in keys]
        combos = []
        for values in itertools.product(*value_lists):
            combos.append({k: v for k, v in zip(keys, values) if v is not None})
        # De-duplicate (dropping None can collapse configs) and cap.
        seen, unique = set(), []
        for c in combos:
            key = tuple(sorted(c.items()))
            if key not in seen:
                seen.add(key)
                unique.append(c)
        return unique[:max_trials]

    # -- rendering -------------------------------------------------------
    @staticmethod
    def _dsl_for(path: str, config: Dict[str, Any]) -> str:
        suffix = "".join(f"[{k}:{v}]" for k, v in config.items() if k != "budget")
        return f"{path}{suffix}"

    def _render_config(self, config: Dict[str, Any]) -> str:
        """Build the context string for one config, fully offline."""
        from .highest_level_api import Attachments  # local import: avoid cycles

        parts: List[str] = []
        for path in self.paths:
            ctx = Attachments(self._dsl_for(path, config))
            text = str(ctx)
            budget = config.get("budget")
            if budget is not None:
                holder = attach(path)
                holder.text = text
                holder = context_budget(holder, budget=budget)
                text = holder.text
            parts.append(text)
        return "\n\n".join(parts)

    # -- optimisation ----------------------------------------------------
    def compile(
        self,
        metric: Callable[[str], float],
        max_trials: int = 32,
        verbose: bool = False,
    ) -> "OptimizableAttachments":
        """Search the space, scoring each config's context with ``metric``."""
        self.trials = []
        self.best_score = float("-inf")
        self.best_config = {}
        for config in self.candidates(max_trials=max_trials):
            try:
                context = self._render_config(config)
                score = float(metric(context))
            except Exception as exc:  # a bad config shouldn't kill the search
                self.trials.append({"config": config, "error": str(exc)})
                continue
            trial = {
                "config": config,
                "score": score,
                "tokens": estimate_tokens(context),
            }
            self.trials.append(trial)
            if verbose:
                print(f"  trial {config} -> score={score:.4f} "
                      f"tokens={trial['tokens']}")
            if score > self.best_score:
                self.best_score = score
                self.best_config = config
        return self

    def render(self, config: Optional[Dict[str, Any]] = None) -> str:
        """Render the context using the best config (or a provided one)."""
        return self._render_config(config if config is not None else self.best_config)

    def __call__(self) -> str:
        return self.render()

    def report(self) -> str:
        """Human-readable summary of the search."""
        lines = [
            f"OptimizableAttachments over {list(self.paths)}",
            f"  search space: {self.search_space}",
            f"  trials run:   {len([t for t in self.trials if 'score' in t])}",
            f"  best config:  {self.best_config or '(none)'}",
            f"  best score:   {self.best_score:.4f}"
            if self.best_score != float("-inf")
            else "  best score:   (not compiled)",
        ]
        ranked = sorted(
            [t for t in self.trials if "score" in t],
            key=lambda t: t["score"],
            reverse=True,
        )
        if ranked:
            lines.append("  ranking:")
            for t in ranked:
                lines.append(
                    f"    score={t['score']:.4f}  tokens={t['tokens']:<5}  {t['config']}"
                )
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Convenience metric factory (handy for offline demos / tests)
# ---------------------------------------------------------------------------

def keyword_coverage_metric(
    keywords: Iterable[str],
    max_tokens: Optional[int] = None,
    over_budget_penalty: float = 1.0,
) -> Callable[[str], float]:
    """A model-free metric: reward covering key terms, penalise overspending.

    Returns the fraction of ``keywords`` present in the context, minus a
    penalty proportional to how far the context exceeds ``max_tokens``. This
    rewards configs that pack the important content into a small budget —
    exactly the "context engineering as optimisation" behaviour.
    """
    kws = [k.lower() for k in keywords]

    def metric(context: str) -> float:
        low = context.lower()
        covered = sum(1 for k in kws if k in low) / max(1, len(kws))
        penalty = 0.0
        if max_tokens is not None:
            tokens = estimate_tokens(context)
            if tokens > max_tokens:
                penalty = over_budget_penalty * (tokens - max_tokens) / max_tokens
        return covered - penalty

    return metric
