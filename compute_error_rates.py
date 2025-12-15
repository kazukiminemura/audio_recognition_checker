"""Utility to compute WER and CER for speech recognition outputs.

Usage:
    python compute_error_rates.py "<reference text>" "<hypothesis text>"
    python compute_error_rates.py @reference.txt @hypothesis.txt

Prefix an argument with ``@`` to read the text from a file.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Callable, Iterable, List, Sequence, Tuple


@dataclass
class EditCounts:
    substitutions: int
    insertions: int
    deletions: int

    @property
    def total_errors(self) -> int:
        return self.substitutions + self.insertions + self.deletions


@dataclass
class ErrorRateResult:
    rate: float
    counts: EditCounts
    reference_tokens: int
    hypothesis_tokens: int


def _add_counts(counts: EditCounts, update: Tuple[int, int, int]) -> EditCounts:
    sub, ins, delete = update
    return EditCounts(
        substitutions=counts.substitutions + sub,
        insertions=counts.insertions + ins,
        deletions=counts.deletions + delete,
    )


def _edit_distance(reference: Sequence[str], hypothesis: Sequence[str]) -> EditCounts:
    """Return edit counts (S, I, D) between two token sequences."""

    ref_len, hyp_len = len(reference), len(hypothesis)
    dp: List[List[EditCounts]] = [
        [EditCounts(0, j, 0) for j in range(hyp_len + 1)]
    ]
    for i in range(1, ref_len + 1):
        row = [EditCounts(0, 0, i)]
        row.extend(EditCounts(0, 0, i) for _ in range(hyp_len))
        dp.append(row)

    for i in range(1, ref_len + 1):
        for j in range(1, hyp_len + 1):
            if reference[i - 1] == hypothesis[j - 1]:
                dp[i][j] = dp[i - 1][j - 1]
                continue

            substitution = _add_counts(dp[i - 1][j - 1], (1, 0, 0))
            insertion = _add_counts(dp[i][j - 1], (0, 1, 0))
            deletion = _add_counts(dp[i - 1][j], (0, 0, 1))

            candidates = [substitution, insertion, deletion]
            dp[i][j] = min(
                candidates,
                key=lambda c: (c.total_errors, c.substitutions, c.insertions, c.deletions),
            )

    return dp[ref_len][hyp_len]


def _calculate_error_rate_for_lines(
    references: Sequence[str],
    hypotheses: Sequence[str],
    tokenizer: Callable[[str], Sequence[str]],
) -> ErrorRateResult:
    total_counts = EditCounts(0, 0, 0)
    total_reference_tokens = 0
    total_hypothesis_tokens = 0

    for reference, hypothesis in zip(references, hypotheses):
        reference_tokens = tokenizer(reference)
        hypothesis_tokens = tokenizer(hypothesis)
        counts = _edit_distance(reference_tokens, hypothesis_tokens)
        total_counts = _add_counts(
            total_counts,
            (counts.substitutions, counts.insertions, counts.deletions),
        )
        total_reference_tokens += len(reference_tokens)
        total_hypothesis_tokens += len(hypothesis_tokens)

    if total_reference_tokens == 0:
        rate = 0.0 if total_hypothesis_tokens == 0 else 1.0
    else:
        rate = total_counts.total_errors / total_reference_tokens
    return ErrorRateResult(
        rate=rate,
        counts=total_counts,
        reference_tokens=total_reference_tokens,
        hypothesis_tokens=total_hypothesis_tokens,
    )


def calculate_wer(reference: str, hypothesis: str) -> Tuple[float, EditCounts]:
    """Compute WER and edit counts for word-level tokens."""
    result = _calculate_error_rate_for_lines([reference], [hypothesis], lambda s: s.split())
    return result.rate, result.counts


def calculate_cer(reference: str, hypothesis: str) -> Tuple[float, EditCounts]:
    """Compute CER and edit counts for character-level tokens."""
    result = _calculate_error_rate_for_lines([reference], [hypothesis], list)
    return result.rate, result.counts


def _format_formula(label: str, result: ErrorRateResult) -> str:
    if result.reference_tokens == 0:
        return (
            f"{label} formula: reference has zero tokens, so the error rate is "
            f"{result.rate:.3f} by definition."
        )
    return (
        f"{label} = (S + D + I) / N = ("
        f"{result.counts.substitutions} + "
        f"{result.counts.deletions} + "
        f"{result.counts.insertions}) / "
        f"{result.reference_tokens} = {result.rate:.3f}"
    )


def _read_lines(value: str) -> List[str]:
    if value.startswith("@"):
        with open(value[1:], "r", encoding="utf-8") as handle:
            return handle.read().splitlines()
    return [value]


def main(args: Iterable[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Compute WER and CER.")
    parser.add_argument(
        "reference",
        help="Reference text or @path to file containing the text.",
    )
    parser.add_argument(
        "hypothesis",
        help="Hypothesis text or @path to file containing the text.",
    )
    parser.add_argument(
        "--show-formula",
        action="store_true",
        help="Show the error-rate formula using the computed counts.",
    )
    parsed = parser.parse_args(args=args)

    reference_lines = _read_lines(parsed.reference)
    hypothesis_lines = _read_lines(parsed.hypothesis)

    if len(reference_lines) != len(hypothesis_lines):
        raise ValueError(
            "Reference and hypothesis must contain the same number of lines"
        )

    wer_result = _calculate_error_rate_for_lines(
        reference_lines, hypothesis_lines, lambda s: s.split()
    )
    cer_result = _calculate_error_rate_for_lines(reference_lines, hypothesis_lines, list)

    reference_text = "\n".join(reference_lines)
    hypothesis_text = "\n".join(hypothesis_lines)

    print("Reference:", reference_text)
    print("Hypothesis:", hypothesis_text)
    print("\nWER: {:.3f}".format(wer_result.rate))
    print("  Substitutions: {}".format(wer_result.counts.substitutions))
    print("  Deletions: {}".format(wer_result.counts.deletions))
    print("  Insertions: {}".format(wer_result.counts.insertions))
    if parsed.show_formula:
        print("  " + _format_formula("WER", wer_result))

    print("\nCER: {:.3f}".format(cer_result.rate))
    print("  Substitutions: {}".format(cer_result.counts.substitutions))
    print("  Deletions: {}".format(cer_result.counts.deletions))
    print("  Insertions: {}".format(cer_result.counts.insertions))
    if parsed.show_formula:
        print("  " + _format_formula("CER", cer_result))


if __name__ == "__main__":
    main()
