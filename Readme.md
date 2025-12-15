# Audio Recognition Error Rates

This repository provides a simple Python script to compute Word Error Rate (WER) and Character Error Rate (CER) at the same time. You can pass raw text or point to files that contain the reference and hypothesis strings.

## Usage

Run the script with two arguments: the reference text and the hypothesis text. Prefix an argument with `@` to load the text from a file.

```bash
python compute_error_rates.py "これは ペン です" "これは ペン"
python compute_error_rates.py @reference.txt @hypothesis.txt
```

The output includes both WER and CER along with the counts for substitutions, deletions, and insertions.
