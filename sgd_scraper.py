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


def iter_xml_urls(start_path: str = LIST_PATH) -> list[str]:
    """Breadth-first crawl under ``start_path`` and yield OCR XML URLs."""
    seen: set[str] = set()
    queue: list[str] = [f"{BASE_URL}{start_path}"]
    while queue:
        url = queue.pop(0)
        if url in seen:
            continue
        seen.add(url)
        try:
            html = fetch_url(url)
        except Exception as exc:
            print(f"Failed to fetch {url}: {exc}")
            continue
        for href in parse_links(html):
            # Make absolute URL relative to the current page
            if href.startswith("http://") or href.startswith("https://"):
                abs_url = href
            else:
                abs_url = f"{url.rstrip('/')}/{href.lstrip('/')}"

            if abs_url.endswith(".xml") or abs_url.endswith(".xmlxml"):
                yield abs_url
            elif abs_url.startswith(f"{BASE_URL}{LIST_PATH}") and abs_url not in seen:
                # continue crawling within the collection
                queue.append(abs_url)


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