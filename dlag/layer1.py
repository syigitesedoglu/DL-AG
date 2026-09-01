"""
DL-AG Layer 1: Levenshtein + language-model purification
Implements Algorithm (paper): lambda=0.7, delta=2, skip rules, confidence tau.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Dict, Iterable, List, Optional, Set

try:
    import editdistance
except ImportError:  # pragma: no cover
    editdistance = None

try:
    from wordfreq import zipf_frequency
except ImportError:  # pragma: no cover
    zipf_frequency = None

try:
    from spellchecker import SpellChecker
except ImportError:  # pragma: no cover
    SpellChecker = None


_URL_RE = re.compile(
    r"^(https?://|www\.)|"
    r"[a-z0-9.-]+\.(com|org|net|edu|gov|io|ai)(/.*)?$",
    re.I,
)

_DEFAULT_PROT = {
    "COVID-19",
    "Covid-19",
    "ChatGPT",
    "GPT-4",
    "BERT",
    "RoBERTa",
    "DistilBERT",
    "AdvGLUE",
    "TextFooler",
}


def _lev(a: str, b: str) -> int:
    if editdistance is not None:
        return int(editdistance.eval(a, b))
    # pure-python fallback
    if a == b:
        return 0
    la, lb = len(a), len(b)
    if la == 0:
        return lb
    if lb == 0:
        return la
    prev = list(range(lb + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            ins, delete, sub = cur[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)
            cur.append(min(ins, delete, sub))
        prev = cur
    return prev[-1]


def _lm_logprob(word: str) -> float:
    """Unigram LM proxy: higher zipf -> higher log P. Returns log P approx."""
    w = word.lower()
    if zipf_frequency is not None:
        z = zipf_frequency(w, "en")
        # zipf is log10-ish frequency score; convert to soft log-prob
        return math.log(max(10 ** (z - 9.0), 1e-12))
    # crude fallback: prefer shorter alphabetic tokens
    if w.isalpha():
        return -math.log(max(len(w), 1))
    return -20.0


class Layer1Purifier:
    """Paper Layer~1: discrete repair with Levenshtein radius and LM term."""

    def __init__(
        self,
        delta: int = 2,
        lam: float = 0.7,
        tau: float = 1e-6,
        protected: Optional[Iterable[str]] = None,
        max_candidates: int = 40,
    ):
        self.delta = delta
        self.lam = lam
        self.tau = tau
        self.max_candidates = max_candidates
        self.protected: Set[str] = set(_DEFAULT_PROT)
        if protected:
            self.protected.update(protected)

        self.lexicon: Set[str] = set()
        self._by_len: Dict[int, List[str]] = defaultdict(list)
        self._build_lexicon()

    def _build_lexicon(self) -> None:
        words: Set[str] = set()
        if SpellChecker is not None:
            sc = SpellChecker(language="en")
            words.update(sc.word_frequency.keys())
        # Always keep a small core so the class still works offline
        words.update(
            {
                "the", "a", "an", "and", "or", "of", "to", "in", "on", "for",
                "is", "are", "was", "were", "be", "been", "good", "bad", "movie",
                "film", "great", "terrible", "love", "hate", "best", "worst",
                "not", "very", "this", "that", "with", "from", "as", "it",
            }
        )
        # Cap for memory / speed on CPU full runs
        if len(words) > 80_000:
            # keep higher-frequency-looking tokens (shorter + alpha)
            ranked = sorted(words, key=lambda w: (0 if w.isalpha() else 1, len(w), w))
            words = set(ranked[:80_000])
        self.lexicon = {w.lower() for w in words if w and w.isascii()}
        for w in self.lexicon:
            self._by_len[len(w)].append(w)

    def _is_url(self, tok: str) -> bool:
        return bool(_URL_RE.search(tok))

    def _candidates(self, w_adv: str) -> List[str]:
        w = w_adv.lower()
        cands: List[str] = []
        for L in range(max(1, len(w) - self.delta), len(w) + self.delta + 1):
            for w2 in self._by_len.get(L, []):
                if _lev(w, w2) <= self.delta:
                    cands.append(w2)
                    if len(cands) >= self.max_candidates * 5:
                        break
            if len(cands) >= self.max_candidates * 5:
                break
        # rank by morph+lm quickly then truncate
        scored = []
        for w_cand in cands:
            morph = _lev(w, w_cand)
            sem = -_lm_logprob(w_cand)
            score = self.lam * morph + (1.0 - self.lam) * sem
            scored.append((score, w_cand))
        scored.sort(key=lambda x: x[0])
        return [w for _, w in scored[: self.max_candidates]]

    def purify_token(self, w_adv: str, context: str = "") -> str:
        if not w_adv:
            return w_adv
        if self._is_url(w_adv) or w_adv in self.protected:
            return w_adv
        # keep punctuation-only / numbers
        core = re.sub(r"^[^A-Za-z]+|[^A-Za-z]+$", "", w_adv)
        if not core:
            return w_adv
        if core.lower() in self.lexicon:
            return w_adv

        cands = self._candidates(core)
        if not cands:
            return w_adv

        best = core
        best_score = float("inf")
        best_p = 0.0
        for w_cand in cands:
            morph = _lev(core.lower(), w_cand)
            logp = _lm_logprob(w_cand)
            sem = -logp
            score = self.lam * morph + (1.0 - self.lam) * sem
            if score < best_score:
                best_score = score
                best = w_cand
                best_p = math.exp(logp)

        if best_p < self.tau:
            return w_adv

        # restore original capitalization pattern lightly
        if core.isupper():
            best = best.upper()
        elif core[:1].isupper():
            best = best[:1].upper() + best[1:]
        # reattach edge punctuation
        prefix = w_adv[: len(w_adv) - len(w_adv.lstrip(".,!?;:'\""))]
        # simpler: if w_adv had non-alpha edges, keep them
        m = re.match(r"^([^A-Za-z]*)(.*?)([^A-Za-z]*)$", w_adv)
        if m:
            return f"{m.group(1)}{best}{m.group(3)}"
        return best

    def purify(self, text: str) -> str:
        if not text or not text.strip():
            return text
        tokens = text.split()
        out = [self.purify_token(tok, context=text) for tok in tokens]
        return " ".join(out)
