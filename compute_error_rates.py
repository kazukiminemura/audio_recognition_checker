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
) -> Tuple[float, EditCounts]:
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
    return rate, total_counts


def calculate_wer(reference: str, hypothesis: str) -> Tuple[float, EditCounts]:
    """Compute WER and edit counts for word-level tokens."""
    return _calculate_error_rate_for_lines([reference], [hypothesis], lambda s: s.split())


def calculate_cer(reference: str, hypothesis: str) -> Tuple[float, EditCounts]:
    """Compute CER and edit counts for character-level tokens."""
    return _calculate_error_rate_for_lines([reference], [hypothesis], list)


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
    parsed = parser.parse_args(args=args)

    reference_lines = _read_lines(parsed.reference)
    hypothesis_lines = _read_lines(parsed.hypothesis)

    if len(reference_lines) != len(hypothesis_lines):
        raise ValueError(
            "Reference and hypothesis must contain the same number of lines"
        )

    wer, wer_counts = _calculate_error_rate_for_lines(
        reference_lines, hypothesis_lines, lambda s: s.split()
    )
    cer, cer_counts = _calculate_error_rate_for_lines(reference_lines, hypothesis_lines, list)

    reference_text = "\n".join(reference_lines)
    hypothesis_text = "\n".join(hypothesis_lines)

    print("Reference:", reference_text)
    print("Hypothesis:", hypothesis_text)
    print("\nWER: {:.3f}".format(wer))
    print("  Substitutions: {}".format(wer_counts.substitutions))
    print("  Deletions: {}".format(wer_counts.deletions))
    print("  Insertions: {}".format(wer_counts.insertions))

    print("\nCER: {:.3f}".format(cer))
    print("  Substitutions: {}".format(cer_counts.substitutions))
    print("  Deletions: {}".format(cer_counts.deletions))
    print("  Insertions: {}".format(cer_counts.insertions))


if __name__ == "__main__":
    main()
