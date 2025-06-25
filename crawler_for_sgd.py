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
from typing import List, Dict, Iterator, Tuple, Set
from io import BytesIO
from zipfile import ZipFile
from concurrent.futures import ThreadPoolExecutor
import argparse

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
DEFAULT_DELAY = 0.2  # polite crawl delay (seconds)
PAGE_RETRIES = Retry(
    total=2,
    backoff_factor=1.0,
    status_forcelist=[500, 502, 503, 504],
)
REQUEST_TIMEOUT = 15  # seconds per request
DEFAULT_MAX_ITEMS = 500  # cap the crawl to avoid CI timeouts
VISITED_FILE = "visited.txt"

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(message)s", level=logging.INFO
)


class BaseCrawler:
    """Common HTTP helpers for crawlers."""

    def __init__(self, delay: float = DEFAULT_DELAY) -> None:
        self.delay = delay
        self.session = requests.Session()
        self.session.headers.update(HEADERS)
        self.session.mount("https://", HTTPAdapter(max_retries=PAGE_RETRIES))

    def fetch_soup(self, path: str) -> BeautifulSoup:
        """GET a path relative to BASE_URL and return BeautifulSoup."""
        url = f"{BASE_URL}{path}"
        for attempt in range(3):
            resp = self.session.get(url, timeout=REQUEST_TIMEOUT)
            if resp.ok:
                return BeautifulSoup(resp.text, "lxml")
            logging.warning("Retry %s for %s (%s)", attempt, url, resp.status_code)
            time.sleep(self.delay * 2)
        resp.raise_for_status()

    @staticmethod
    def strip_xml(xml_bytes: bytes) -> str:
        """Remove tags from XML and return plain UTF-8 text."""
        parser = etree.XMLParser(recover=True, encoding="utf-8")
        tree = etree.fromstring(xml_bytes, parser=parser)
        text_iter = tree.itertext()
        return " ".join(chunk.strip() for chunk in text_iter if chunk.strip())

    @staticmethod
    def push_chunk(
        data: List[Dict[str, str]],
        features: Features,
        repo: str,
        token: str,
        private: bool,
    ) -> None:
        """Create a temporary Dataset from <data> and push to hub."""
        ds = Dataset.from_list(data, features=features)
        ds.push_to_hub(
            repo_id=repo,
            token=token,
            split="train",
            private=private,
            max_shard_size="500MB",
        )
        logging.info("Pushed %d rows to %s", len(ds), repo)


class SGDCrawler(BaseCrawler):
    """Crawler for the Statengeneraal Digitaal repository."""

    ROOT_PATH = ROOT_PATH

    def iter_subarea_paths(self) -> Iterator[str]:
        """Yield /frbr/sgd/<subarea> paths as they are discovered."""
        seen: Set[str] = set()
        offset = 0
        while True:
            path = f"{self.ROOT_PATH}?start={offset}" if offset else self.ROOT_PATH
            soup = self.fetch_soup(path)
            links = soup.select("a[href^='/frbr/sgd/']")
            if not links:
                break
            for a in links:
                href = a["href"].split("?")[0].rstrip("/")
                if href.count("/") == 3 and href not in seen:
                    seen.add(href)
                    yield href
            offset += 11
            time.sleep(self.delay)


    def iter_document_paths(self, subarea_path: str) -> Iterator[str]:
        """Yield /frbr/sgd/<subarea>/<docid> paths within a subarea."""
        seen: Set[str] = set()
        offset = 0
        while True:
            page_path = f"{subarea_path}?start={offset}" if offset else subarea_path
            soup = self.fetch_soup(page_path)
            doc_links = soup.select("a[href^='/frbr/sgd/'][href$='']")  # basic heuristic
            if not doc_links:
                break
            for a in doc_links:
                href = a["href"].split("?")[0].rstrip("/")
                if href.count("/") == 4 and href not in seen:
                    seen.add(href)
                    yield href
            offset += 11
            time.sleep(self.delay)


    def iter_ocr_xml(self, doc_path: str) -> Iterator[Tuple[str, bytes]]:
        """Yield ``(url, xml_bytes)`` tuples for OCR XML files of a document."""
        expr_zip = f"{doc_path}/1?format=zip"
        try:
            resp = self.session.get(f"{BASE_URL}{expr_zip}", timeout=60)
            resp.raise_for_status()
            with ZipFile(BytesIO(resp.content)) as z:
                for name in z.namelist():
                    lname = name.lower()
                    if lname.endswith(".xml") and "metadata" not in lname and not lname.endswith("manifest.xml") and not lname.endswith("didl.xml"):
                        yield (f"{BASE_URL}{expr_zip}#{name}", z.read(name))
            return
        except Exception as exc:
            logging.warning("ZIP failed for %s: %s", expr_zip, exc)

        expr_path = f"{doc_path}/1"
        soup = self.fetch_soup(expr_path)
        ocr_link = soup.find("a", href=lambda h: h and h.endswith("/ocr"))
        if not ocr_link:
            return
        soup_ocr = self.fetch_soup(ocr_link["href"])
        xml_links = soup_ocr.select("a[href$='.xml']")

        def fetch_one(url: str) -> Tuple[str, bytes] | None:
            try:
                resp = self.session.get(url, timeout=60)
                resp.raise_for_status()
                return url, resp.content
            except Exception as exc:
                logging.error("Failed %s: %s", url, exc)
                return None

        urls = [f"{BASE_URL}{a['href']}" for a in xml_links]
        with ThreadPoolExecutor(max_workers=4) as pool:
            for result in pool.map(fetch_one, urls):
                if result:
                    url, content = result
                    lname = PurePosixPath(url).name.lower()
                    if "metadata" in lname or lname.endswith("manifest.xml") or lname.endswith("didl.xml"):
                        continue
                    yield url, content
                time.sleep(self.delay)


    def records_stream(
        self,
        limit: int | None = None,
        resume: bool = False,
    ) -> Iterator[Dict[str, str]]:
        """Yield dataset records matching the required schema."""
        grabbed = 0
        seen: Set[str] = set()
        fh = None
        if resume and os.path.exists(VISITED_FILE):
            with open(VISITED_FILE, "r", encoding="utf-8") as fh_in:
                seen.update(line.strip() for line in fh_in)
        if resume:
            fh = open(VISITED_FILE, "a", encoding="utf-8")

        for sub_path in tqdm(self.iter_subarea_paths(), desc="Subareas"):
            for doc_path in self.iter_document_paths(sub_path):
                for url, xml_bytes in tqdm(
                    self.iter_ocr_xml(doc_path), desc="XML", leave=False
                ):
                    if url in seen:
                        continue
                    if limit is not None and grabbed >= limit:
                        if fh:
                            fh.close()
                        return
                    content = self.strip_xml(xml_bytes)
                    yield {
                        "url": url,
                        "content": content,
                        "source": "Statengeneraal Digitaal",
                    }
                    grabbed += 1
                    if fh:
                        fh.write(url + "\n")
                        fh.flush()
                    time.sleep(self.delay)
        if fh:
            fh.close()


def push_dataset(args: argparse.Namespace) -> None:
    hf_repo = os.environ["HF_DATASET_REPO"]
    token = os.environ["HF_TOKEN"]
    private = os.getenv("HF_PRIVATE", "false").lower() == "true"

    crawler = SGDCrawler(delay=args.delay)
    features = Features(
        {
            "url": Value("string"),
            "content": Value("string"),
            "source": Value("string"),
        }
    )

    chunk: List[Dict[str, str]] = []
    chunk_size = 1000
    total = 0
    for rec in crawler.records_stream(limit=args.max_items, resume=args.resume):
        chunk.append(rec)
        if len(chunk) >= chunk_size:
            crawler.push_chunk(chunk, features, hf_repo, token, private)
            total += len(chunk)
            chunk.clear()
    if chunk:
        crawler.push_chunk(chunk, features, hf_repo, token, private)
        total += len(chunk)
    logging.info("Upload complete: %d records", total)




def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Scrape Statengeneraal Digitaal")
    parser.add_argument("--max-items", type=int, default=DEFAULT_MAX_ITEMS)
    parser.add_argument("--delay", type=float, default=DEFAULT_DELAY)
    parser.add_argument("--resume", action="store_true", help="Resume from last run")
    return parser.parse_args()


if __name__ == "__main__":
    try:
        args = parse_args()
        push_dataset(args)
    except KeyError as missing:
        logging.critical("Missing env var: %s", missing)
        exit(1)
