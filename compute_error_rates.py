"""Utility to compute WER and CER for speech recognition outputs.

Usage:
    python compute_error_rates.py "<reference text>" "<hypothesis text>"
    python compute_error_rates.py @reference.txt @hypothesis.txt

Prefix an argument with ``@`` to read the text from a file.
"""
from __future__ import annotations

import argparse
from dataclasses import dataclass
from typing import Iterable, List, Sequence, Tuple

from jiwer import cer, measures as jiwer_measures, wer


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


def _calculate_error_rate_for_lines(
    references: Sequence[str],
    hypotheses: Sequence[str],
    mode: str,
) -> ErrorRateResult:
    if mode == "word":
        rate = wer(references, hypotheses)
        output = jiwer_measures.process_words(references, hypotheses)
    elif mode == "char":
        rate = cer(references, hypotheses)
        output = jiwer_measures.process_characters(references, hypotheses)
    else:
        raise ValueError(f"Unsupported mode: {mode}")

    reference_tokens = output.hits + output.substitutions + output.deletions
    hypothesis_tokens = output.hits + output.substitutions + output.insertions
    counts = EditCounts(
        substitutions=output.substitutions,
        insertions=output.insertions,
        deletions=output.deletions,
    )

    return ErrorRateResult(
        rate=rate,
        counts=counts,
        reference_tokens=reference_tokens,
        hypothesis_tokens=hypothesis_tokens,
    )


def calculate_wer(reference: str, hypothesis: str) -> Tuple[float, EditCounts]:
    """Compute WER and edit counts using jiwer at the word level."""
    result = _calculate_error_rate_for_lines([reference], [hypothesis], "word")
    return result.rate, result.counts


def calculate_cer(reference: str, hypothesis: str) -> Tuple[float, EditCounts]:
    """Compute CER and edit counts using jiwer at the character level."""
    result = _calculate_error_rate_for_lines([reference], [hypothesis], "char")
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
        reference_lines, hypothesis_lines, "word"
    )
    cer_result = _calculate_error_rate_for_lines(reference_lines, hypothesis_lines, "char")

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
