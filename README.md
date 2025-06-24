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
python crawler_for_sgd.py
```

See `.github/workflows/sgd.yml` for a complete example.

## Dependencies

The workflow installs these Python packages: `requests`, `beautifulsoup4`,
`lxml`, `datasets` and `tqdm`.
