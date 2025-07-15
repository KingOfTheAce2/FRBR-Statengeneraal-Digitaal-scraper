# Statengeneraal Digitaal scraper

This repository contains a crawler that collects OCR XML text from the
[Statengeneraal Digitaal](https://repository.overheid.nl/frbr/sgd) portal and
pushes the plain text to a Hugging Face dataset.

## Usage

The script is intended to run from GitHub Actions. Configure the following
environment variables (or pass their values via the `--hf-repo` and
`--hf-token` command line arguments):

- `HF_TOKEN` – Hugging Face token with write access.
- `HF_DATASET_REPO` – destination dataset repository (e.g. `my-org/sgd-ocr`).
- `HF_PRIVATE` – optional `true`/`false` to create a private dataset.

Both `HF_TOKEN` and `HF_DATASET_REPO` must be set for the crawler to upload
batches to the hub. If either is missing the script exits with an error.

Run the crawler locally or inside the workflow:

```bash
python scripts/sgd_crawler.py --max-items 1000 --delay 0.5 --years 5 --resume
```

Use `--max-items`, `--delay` and `--years` to control crawl size and
politeness. The optional `--workers` argument processes multiple works in
parallel. The `--resume` flag continues from a previous run using
`visited.txt`. Specify `--hf-repo` and `--hf-token` to override the
corresponding environment variables when uploading to Hugging Face.
Progress is tracked in this file, which is cached between runs and excluded
from version control.

To start a fresh crawl from the earliest available year (1814) remove the
`visited.txt` cache and pass `--years 0`.

Year directories on the portal do not always use a simple four digit format.
Some are ranges such as `18141815` or include suffixes like `1815I`.
The crawler therefore accepts any directory that starts with a four digit year.

By default the crawler processes at most 500 XML files with a 0.2 second delay
between requests.

The crawler writes newline-delimited JSON shards to the `data/` directory,
producing files like `sgd_shard_1.jsonl`. The next shard number is persisted
in `state.json` so new runs append files instead of overwriting existing ones.

When both `HF_TOKEN` and `HF_DATASET_REPO` are set, newly created batches are
uploaded to the specified Hugging Face dataset repository. The script will
exit with an error if either variable is missing when an upload is attempted.

Each HTTP request has a 15 second timeout and is retried twice. This prevents
the workflow from hanging indefinitely when the server is unresponsive.


OCR data is retrieved via the service's ZIP archives when possible to minimise
the number of HTTP requests.

The included GitHub workflow runs daily and keeps track of processed files
using a cached `visited.txt`. Results are uploaded to the public dataset
`vGassen/Dutch-Statengeneraal-Digitaal-Historical`.

See `.github/workflows/sgd-crawler.yml` for a complete example.

The workflow relies on the default `GITHUB_TOKEN` to push new batches back to
this repository, so it must have write permissions.

## Dependencies

The workflow installs these Python packages: `requests`, `beautifulsoup4`,
`lxml`, `datasets` and `tqdm`.
