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
from typing import List, Dict

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


def discover_subarea_paths() -> List[str]:
    """Return list of /frbr/sgd/<subarea> paths."""
    paths: List[str] = []
    page = 0
    while True:
        path = f"{ROOT_PATH}?start={page*11}" if page else ROOT_PATH
        soup = fetch_soup(path)
        links = soup.select("a[href^='/frbr/sgd/']")
        if not links:
            break
        for a in links:
            href = a["href"].split("?")[0].rstrip("/")
            if href.count("/") == 3 and href not in paths:
                paths.append(href)
        page += 1
        time.sleep(SLEEP)
    logging.info("Discovered %d subareas", len(paths))
    return paths


def discover_document_paths(subarea_path: str) -> List[str]:
    """Return list of /frbr/sgd/<subarea>/<docid> paths within a subarea."""
    paths: List[str] = []
    start = 0
    while True:
        page_path = f"{subarea_path}?start={start}" if start else subarea_path
        soup = fetch_soup(page_path)
        doc_links = soup.select("a[href^='/frbr/sgd/'][href$='']")  # basic heuristic
        if not doc_links:
            break
        for a in doc_links:
            href = a["href"].split("?")[0].rstrip("/")
            # Ensure it is a document page (four segments)
            if href.count("/") == 4 and href not in paths:
                paths.append(href)
        start += 11
        time.sleep(SLEEP)
    logging.info("%s → %d docs", subarea_path, len(paths))
    return paths


def discover_ocr_xml_urls(doc_path: str) -> List[str]:
    """Return list of full URLs to OCR XML files for one document."""
    expr_path = f"{doc_path}/1"
    soup = fetch_soup(expr_path)
    ocr_link = soup.find("a", href=lambda h: h and h.endswith("/ocr"))
    if not ocr_link:
        return []
    soup_ocr = fetch_soup(ocr_link["href"])
    xml_links = soup_ocr.select("a[href$='.xml']")
    urls = [f"{BASE_URL}{a['href']}" for a in xml_links]
    return urls


def records_stream(limit: int | None = None) -> Dict[str, str]:
    """Generator yielding dataset records matching the required schema."""
    subareas = discover_subarea_paths()
    grabbed = 0
    for sub_path in subareas:
        for doc_path in discover_document_paths(sub_path):
            xml_urls = discover_ocr_xml_urls(doc_path)
            for url in xml_urls:
                if limit is not None and grabbed >= limit:
                    return
                try:
                    resp = session.get(url, timeout=60)
                    resp.raise_for_status()
                    content = strip_xml(resp.content)
                    yield {
                        "url": url,
                        "content": content,
                        "source": "Statengeneraal Digitaal",
                    }
                    grabbed += 1
                except Exception as exc:
                    logging.error("Failed %s: %s", url, exc)
                finally:
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
