from __future__ import annotations

import hashlib
import random

import regex


def stable_perturbation_seed(row_id: str, family: str, severity: float, seed: int) -> int:
    payload = f"{row_id}|{family}|{severity:.6f}|{seed}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest()[:16], 16)


def perturb_text(text: str, *, family: str, severity: float, seed: int) -> str:
    """Apply a deterministic grapheme-aware input corruption."""

    if not 0.0 <= severity <= 1.0:
        raise ValueError("severity must be in [0, 1]")
    if severity == 0.0 or not text:
        return text
    rng = random.Random(seed)
    graphemes = regex.findall(r"\X", text)
    lexical = [index for index, value in enumerate(graphemes) if regex.search(r"[\p{L}\p{M}]", value)]
    if not lexical:
        return text
    operations = max(1, round(len(lexical) * severity))

    if family == "grapheme_deletion":
        selected = set(rng.sample(lexical, k=min(operations, len(lexical))))
        result = [value for index, value in enumerate(graphemes) if index not in selected]
    elif family == "adjacent_swap":
        result = list(graphemes)
        candidates = [index for index in lexical if index + 1 < len(result) and index + 1 in lexical]
        rng.shuffle(candidates)
        occupied: set[int] = set()
        applied = 0
        for index in candidates:
            if index in occupied or index + 1 in occupied:
                continue
            result[index], result[index + 1] = result[index + 1], result[index]
            occupied.update((index, index + 1))
            applied += 1
            if applied >= operations:
                break
    elif family == "whitespace_noise":
        result = list(graphemes)
        candidates = list(range(1, len(result)))
        rng.shuffle(candidates)
        selected = sorted(candidates[: min(operations, len(candidates))], reverse=True)
        for index in selected:
            if result[index - 1].isspace():
                del result[index - 1]
            elif result[index].isspace():
                del result[index]
            else:
                result.insert(index, " ")
    else:
        raise ValueError(f"Unknown perturbation family: {family}")
    return "".join(result).strip()
