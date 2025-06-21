#!/usr/bin/env python3
"""Simplified scraper for the Statengeneraal Digitaal collection.

The script traverses listing pages on repository.overheid.nl and downloads
all OCR XML files. For each XML file the plain text content and XML tag
names are extracted. Output is written to ``data/statengeneraal_digitaal.jsonl``
as one JSON record per file.

This implementation uses only the Python standard library so it can run in
restricted environments.
"""

import json
from pathlib import Path
from html.parser import HTMLParser
from urllib.request import Request, urlopen
import xml.etree.ElementTree as ET

BASE_URL = "https://repository.overheid.nl"
LIST_PATH = "/frbr/sgd"
TOTAL_PAGES = 22
OUT_PATH = Path("data/statengeneraal_digitaal.jsonl")
USER_AGENT = "sgd-scraper"

def fetch_url(url: str) -> bytes:
    """Retrieve a URL and return the raw bytes."""
    req = Request(url, headers={"User-Agent": USER_AGENT})
    with urlopen(req, timeout=30) as resp:
        return resp.read()

class LinkParser(HTMLParser):
    """Collect all ``href`` values from anchor tags."""

    def __init__(self):
        super().__init__()
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str]]):
        if tag == "a":
            for name, value in attrs:
                if name == "href" and value:
                    self.links.append(value)


def parse_links(html: bytes) -> list[str]:
    parser = LinkParser()
    parser.feed(html.decode(errors="ignore"))
    return parser.links


def plain_text_from_xml(xml_bytes: bytes) -> tuple[str, list[str]]:
    """Return plain text and list of tag names from an XML document."""
    try:
        root = ET.fromstring(xml_bytes)
    except ET.ParseError:
        return "", []

    texts = []
    tags: set[str] = set()
    for el in root.iter():
        tags.add(el.tag)
        if el.text and el.text.strip():
            texts.append(el.text.strip())
    return "\n".join(texts), sorted(tags)


def iter_xml_urls() -> list[str]:
    """Yield OCR XML file URLs from all listing pages."""
    for page in range(1, TOTAL_PAGES + 1):
        list_url = f"{BASE_URL}{LIST_PATH}?page={page}"
        try:
            html = fetch_url(list_url)
        except Exception as exc:
            print(f"Failed to fetch {list_url}: {exc}")
            continue
        for href in parse_links(html):
            if href.startswith("/frbr/sgd/") and href.endswith("/ocr"):
                ocr_url = f"{BASE_URL}{href}"
                try:
                    ocr_html = fetch_url(ocr_url)
                except Exception as exc:
                    print(f"Failed to fetch {ocr_url}: {exc}")
                    continue
                for link in parse_links(ocr_html):
                    if link.endswith(".xml") or link.endswith(".xmlxml"):
                        yield f"{BASE_URL}{link}"


def main() -> None:
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    with OUT_PATH.open("w", encoding="utf-8") as fh:
        for xml_url in iter_xml_urls():
            try:
                xml_bytes = fetch_url(xml_url)
                text, tags = plain_text_from_xml(xml_bytes)
                record = {
                    "url": xml_url,
                    "content": text,
                    "tags": tags,
                    "source": "Statengeneraal Digitaal",
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            except Exception as exc:
                print(f"Failed on {xml_url}: {exc}")
    print(f"SGD: wrote {written} items")


if __name__ == "__main__":
    main()
