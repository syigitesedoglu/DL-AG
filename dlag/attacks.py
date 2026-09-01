"""
Attack wrappers for evaluator.
Prefer TextAttack recipes when installed; otherwise fail with a clear message
(paper full-run needs real attacks, not stubs).
"""
from __future__ import annotations

from typing import Any, Dict, Optional


class AttackNotAvailable(RuntimeError):
    pass


def build_attacks(model, tokenizer, device, recipes=("textfooler", "deepwordbug", "pruthi")) -> Dict[str, Any]:
    try:
        from textattack.attack_recipes import (
            DeepWordBugGao2018,
            Pruthi2019,
            TextFoolerJin2019,
        )
        from textattack.models.wrappers import HuggingFaceModelWrapper
    except ImportError as e:
        raise AttackNotAvailable(
            "textattack is required for paper-protocol evaluator attacks. "
            "Install: pip install textattack"
        ) from e

    wrapper = HuggingFaceModelWrapper(model, tokenizer)
    mapping = {
        "textfooler": TextFoolerJin2019,
        "deepwordbug": DeepWordBugGao2018,
        "pruthi": Pruthi2019,
    }
    attacks = {}
    for name in recipes:
        cls = mapping[name]
        attacks[name] = cls.build(wrapper)
    return attacks


def generate_adversarial(attack, text: str, label: int) -> str:
    """Run one TextAttack recipe; return perturbed text or original on skip/fail."""
    from textattack.shared import AttackedText

    attacked = AttackedText(text)
    result = attack.attack(attacked, label)
    # result can be SuccessfulAttackResult / FailedAttackResult / SkippedAttackResult
    try:
        pert = result.perturbed_result.attacked_text.text
        return pert
    except Exception:
        return text
