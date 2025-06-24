# Statengeneraal Digitaal scraper

This repository contains a crawler that collects OCR XML text from the
[Statengeneraal Digitaal](https://repository.overheid.nl/frbr/sgd) portal and
pushes the plain text to a Hugging Face dataset.

## Usage

The script is intended to run from GitHub Actions. Configure the following
environment variables:

- `HF_TOKEN` – Hugging Face token with write access.
- `HF_DATASET_REPO` – destination dataset repository (e.g. `my-org/sgd-ocr`).
- `HF_PRIVATE` – optional `true`/`false` to create a private dataset.

Run the crawler locally or inside the workflow:

```bash
python crawler_for_sgd.py --max-items 1000 --delay 0.5 --resume
```

Use `--max-items` and `--delay` to control crawl size and politeness. The
`--resume` flag continues from a previous run using `visited.txt`.

By default the crawler processes at most 500 XML files with a 0.2 second delay
between requests.


OCR data is retrieved via the service's ZIP archives when possible to minimise
the number of HTTP requests.

The included GitHub workflow runs every 30 minutes and keeps track of processed
files using a cached `visited.txt`. Results are uploaded to the public dataset
`vGassen/Dutch-Statengeneraal-Digitaal-Historical`.

See `.github/workflows/sgd.yml` for a complete example.

## Dependencies

The workflow installs these Python packages: `requests`, `beautifulsoup4`,
`lxml`, `datasets` and `tqdm`.
