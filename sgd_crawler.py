#!/usr/bin/env python3
"""
Crawler for the full Statengeneraal Digitaal (SGD) collection.
Traverses all listing pages and downloads the XML manifests in memory
without saving the raw files. Output is a JSONL file with url, plain text
content and source.
"""
import json
from pathlib import Path
import requests
from bs4 import BeautifulSoup

HF_REPO = "vGassen/Dutch-Statengeneraal-Digitaal-Historical"
OUT_PATH = Path("data/statengeneraal_digitaal.jsonl")
BASE_URL = "https://repository.overheid.nl"
LIST_PATH = "/frbr/sgd"
TOTAL_PAGES = 22

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "sgd-crawler"
SESSION.timeout = (10, 30)


def plain_text_from_xml(xml_bytes: bytes) -> str:
    """Extract plain text from an OCR XML file."""
    soup = BeautifulSoup(xml_bytes, "lxml-xml")
    for tag in soup.select("meta, head, style, script"):
        tag.decompose()
    return "\n".join(s.strip() for s in soup.stripped_strings)


def iter_manifest_urls():
    """Yield manifest XML URLs from all listing pages."""
    for page in range(1, TOTAL_PAGES + 1):
        url = f"{BASE_URL}{LIST_PATH}?page={page}"
        r = SESSION.get(url)
        r.raise_for_status()
        soup = BeautifulSoup(r.text, "html.parser")
        for a in soup.find_all("a", href=True):
            href = a["href"]
            if href.startswith("/frbr/sgd/") and href.endswith("/ocr"):
                yield f"{BASE_URL}{href}"


def main():
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for manifest_url in iter_manifest_urls():
            try:
                xml_bytes = SESSION.get(manifest_url).content
                text = plain_text_from_xml(xml_bytes)
                record = {
                    "url": manifest_url,
                    "content": text,
                    "source": "Statengeneraal Digitaal",
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            except Exception as e:
                print(f"Failed on {manifest_url}: {e}")
    print(f"SGD crawler wrote {written} items")


if __name__ == "__main__":
    main()
