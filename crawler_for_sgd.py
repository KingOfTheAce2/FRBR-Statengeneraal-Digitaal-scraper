#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""crawler_for_sgd.py

Crawl Statengeneraal Digitaal (https://repository.overheid.nl/frbr/sgd)
and push OCR XML text to a Hugging Face dataset.

Designed to run *only* inside GitHub Actions. Requires the following
environment variables:

- HF_TOKEN          – Hugging Face token with write access
- HF_DATASET_REPO   – e.g. "my-org/sgd-ocr"
- HF_PRIVATE        – "true" or "false" (optional; default false)

Dependencies are installed in the workflow:
  requests, beautifulsoup4, lxml, datasets, tqdm
"""

import os
import time
import logging
from pathlib import PurePosixPath
from typing import List, Dict, Iterator
from io import BytesIO
from zipfile import ZipFile

import requests
from bs4 import BeautifulSoup
from lxml import etree
from datasets import Dataset, Features, Value
from tqdm import tqdm
from requests.adapters import HTTPAdapter, Retry

BASE_URL = "https://repository.overheid.nl"
ROOT_PATH = "/frbr/sgd"
HEADERS = {
    "User-Agent": "SDG-GitHubActions-OCR-Scraper"
}
SLEEP = 0.2  # polite crawl delay (seconds)
PAGE_RETRIES = Retry(
    total=5,
    backoff_factor=1.5,
    status_forcelist=[500, 502, 503, 504],
)
MAX_ITEMS = 500  # cap the crawl to avoid CI timeouts

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
)
session = requests.Session()
session.headers.update(HEADERS)
session.mount("https://", HTTPAdapter(max_retries=PAGE_RETRIES))


def fetch_soup(path: str) -> BeautifulSoup:
    """GET a path relative to BASE_URL and return BeautifulSoup parser."""
    url = f"{BASE_URL}{path}"
    for _ in range(3):
        resp = session.get(url, timeout=30)
        if resp.ok:
            return BeautifulSoup(resp.text, "lxml")
        logging.warning("Retry %s for %s (%s)", _, url, resp.status_code)
        time.sleep(SLEEP * 2)
    resp.raise_for_status()  # final error


def strip_xml(xml_bytes: bytes) -> str:
    """Remove all tags from XML and return plain UTF-8 text."""
    parser = etree.XMLParser(recover=True, encoding="utf-8")
    tree = etree.fromstring(xml_bytes, parser=parser)
    text_iter = tree.itertext()
    return " ".join(chunk.strip() for chunk in text_iter if chunk.strip())


def iter_subarea_paths() -> Iterator[str]:
    """Yield /frbr/sgd/<subarea> paths as they are discovered."""
    seen: set[str] = set()
    page = 0
    while True:
        path = f"{ROOT_PATH}?start={page*11}" if page else ROOT_PATH
        soup = fetch_soup(path)
        links = soup.select("a[href^='/frbr/sgd/']")
        if not links:
            break
        for a in links:
            href = a["href"].split("?")[0].rstrip("/")
            if href.count("/") == 3 and href not in seen:
                seen.add(href)
                yield href
        page += 1
        time.sleep(SLEEP)


def iter_document_paths(subarea_path: str) -> Iterator[str]:
    """Yield /frbr/sgd/<subarea>/<docid> paths within a subarea."""
    seen: set[str] = set()
    start = 0
    while True:
        page_path = f"{subarea_path}?start={start}" if start else subarea_path
        soup = fetch_soup(page_path)
        doc_links = soup.select("a[href^='/frbr/sgd/'][href$='']")  # basic heuristic
        if not doc_links:
            break
        for a in doc_links:
            href = a["href"].split("?")[0].rstrip("/")
            if href.count("/") == 4 and href not in seen:
                seen.add(href)
                yield href
        start += 11
        time.sleep(SLEEP)


def iter_ocr_xml(doc_path: str) -> Iterator[tuple[str, bytes]]:
    """Yield ``(url, xml_bytes)`` tuples for OCR XML files of a document."""
    expr_zip = f"{doc_path}/1?format=zip"
    try:
        resp = session.get(f"{BASE_URL}{expr_zip}", timeout=60)
        resp.raise_for_status()
        with ZipFile(BytesIO(resp.content)) as z:
            for name in z.namelist():
                if name.lower().endswith(".xml"):
                    yield (f"{BASE_URL}{expr_zip}#{name}", z.read(name))
        return
    except Exception as exc:
        logging.warning("ZIP failed for %s: %s", expr_zip, exc)

    expr_path = f"{doc_path}/1"
    soup = fetch_soup(expr_path)
    ocr_link = soup.find("a", href=lambda h: h and h.endswith("/ocr"))
    if not ocr_link:
        return
    soup_ocr = fetch_soup(ocr_link["href"])
    xml_links = soup_ocr.select("a[href$='.xml']")
    for a in xml_links:
        url = f"{BASE_URL}{a['href']}"
        try:
            resp = session.get(url, timeout=60)
            resp.raise_for_status()
            yield (url, resp.content)
        except Exception as exc:
            logging.error("Failed %s: %s", url, exc)
        finally:
            time.sleep(SLEEP)


def records_stream(limit: int | None = None) -> Iterator[Dict[str, str]]:
    """Yield dataset records matching the required schema."""
    grabbed = 0
    for sub_path in iter_subarea_paths():
        for doc_path in iter_document_paths(sub_path):
            for url, xml_bytes in iter_ocr_xml(doc_path):
                if limit is not None and grabbed >= limit:
                    return
                content = strip_xml(xml_bytes)
                yield {
                    "url": url,
                    "content": content,
                    "source": "Statengeneraal Digitaal",
                }
                grabbed += 1
                time.sleep(SLEEP)


def push_dataset():
    hf_repo = os.environ["HF_DATASET_REPO"]
    token = os.environ["HF_TOKEN"]
    private = os.getenv("HF_PRIVATE", "false").lower() == "true"

    features = Features(
        {
            "url": Value("string"),
            "content": Value("string"),
            "source": Value("string"),
        }
    )

    chunk, chunk_size = [], 1000
    total = 0
    for rec in tqdm(records_stream(limit=MAX_ITEMS), desc="Processing"):
        chunk.append(rec)
        if len(chunk) >= chunk_size:
            _push_chunk(chunk, features, hf_repo, token, private)
            total += len(chunk)
            chunk.clear()
    if chunk:
        _push_chunk(chunk, features, hf_repo, token, private)
        total += len(chunk)
    logging.info("Upload complete: %d records", total)


def _push_chunk(
    data: List[Dict[str, str]],
    features: Features,
    repo: str,
    token: str,
    private: bool,
):
    """Create a temporary Dataset from <data> and push to hub (append)."""
    ds = Dataset.from_list(data, features=features)
    ds.push_to_hub(
        repo_id=repo,
        token=token,
        split="train",
        private=private,
        max_shard_size="500MB",
    )
    logging.info("Pushed %d rows to %s", len(ds), repo)


if __name__ == "__main__":
    try:
        push_dataset()
    except KeyError as missing:
        logging.critical("Missing env var: %s", missing)
        exit(1)
