# Audio Recognition Error Rates

This repository provides a simple Python script to compute Word Error Rate (WER) and Character Error Rate (CER) at the same time. You can pass raw text or point to files that contain the reference and hypothesis strings.

## Usage

Run the script with two arguments: the reference text and the hypothesis text. Prefix an argument with `@` to load the text from a file. When files are used, each line is paired by position so multi-line inputs are fully evaluated.

```bash
python compute_error_rates.py "これは ペン です" "これは ペン"
python compute_error_rates.py @reference.txt @hypothesis.txt
python compute_error_rates.py "今日は いい 天気 ですね" "今日は いい 天気 です" --show-formula
```

The output includes both WER and CER along with the counts for substitutions, deletions, and insertions.
Add the `--show-formula` flag to print the calculation steps, which is useful if you want to verify a
specific formula such as ``WER = (S + I + D) / N``. For the Japanese example above, the command will
report that substituting “ですね” with “です” gives a single substitution over four reference words,
leading to a WER of 0.25 (25%).
