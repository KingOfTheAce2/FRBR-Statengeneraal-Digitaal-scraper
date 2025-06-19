#!/usr/bin/env python3
"""
Scraper for Statengeneraal Digitaal collection (SGD).
Output: JSONL file with url, plain-text content, and fixed source.
"""
import os, json, datetime as dt
from pathlib import Path
import requests
from lxml import etree
from bs4 import BeautifulSoup
from huggingface_hub import HfApi

HF_REPO = "vGassen/Dutch-Statengeneraal-Digitaal-Historical"
OUT_PATH = Path("data/statengeneraal_digitaal.jsonl")
COLLECTION_ID = "sgd"
SRU_URL = "https://repository.overheid.nl/sru"

SESSION = requests.Session()
SESSION.headers["User-Agent"] = "sgd-scraper"
SESSION.timeout = (10, 30)

def search_identifiers(date: str) -> list[str]:
    query = f'c.product-area=={COLLECTION_ID} AND dt.modified=="{date}"'
    params = {
        "operation": "searchRetrieve",
        "version": "2.0",
        "maximumRecords": "1000",
        "recordSchema": "gzd",
        "query": query
    }
    r = SESSION.get(SRU_URL, params=params)
    r.raise_for_status()
    root = etree.fromstring(r.content)
    return [el.text for el in root.findall(".//{*}identifier") if el.text]

def fetch_sru(identifier: str) -> etree._Element:
    params = {
        "operation": "searchRetrieve",
        "version": "2.0",
        "maximumRecords": "1",
        "recordSchema": "gzd",
        "query": f'dt.identifier="{identifier}"'
    }
    r = SESSION.get(SRU_URL, params=params)
    r.raise_for_status()
    return etree.fromstring(r.content)

def extract_manifest_url(root: etree._Element) -> str:
    for el in root.iterfind(".//{*}itemUrl"):
        if el.get("manifestation") == "xml":
            return el.text
    raise ValueError("No XML manifestation found")

def extract_preferred_url(root: etree._Element) -> str | None:
    el = root.find(".//{*}prefferedUrl")
    return el.text if el is not None else None

def plain_text_from_xml(xml_bytes: bytes) -> str:
    soup = BeautifulSoup(xml_bytes, "lxml-xml")
    for tag in soup.select("meta, head, style, script"):
        tag.decompose()
    return "\n".join(s.strip() for s in soup.stripped_strings)

def main():
    scrape_date = os.getenv("SCRAPE_DATE", dt.date.today().isoformat())
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    written = 0

    try:
        identifiers = search_identifiers(scrape_date)
    except Exception as e:
        print(f"Failed to query SGD for {scrape_date}: {e}")
        return

    with OUT_PATH.open("a", encoding="utf-8") as fh:
        for identifier in identifiers:
            try:
                root = fetch_sru(identifier)
                manifest_url = extract_manifest_url(root)
                manifest_xml = SESSION.get(manifest_url).content
                text = plain_text_from_xml(manifest_xml)
                url = extract_preferred_url(root) or manifest_url

                record = {
                    "url": url,
                    "content": text,
                    "source": "Statengeneraal Digitaal"
                }
                fh.write(json.dumps(record, ensure_ascii=False) + "\n")
                written += 1
            except Exception as e:
                print(f"Failed on {identifier}: {e}")

    print(f"SGD: wrote {written} items for {scrape_date}")

    if os.getenv("HF_TOKEN"):
        HfApi().upload_file(
            path_or_fileobj=str(OUT_PATH),
            path_in_repo=OUT_PATH.name,
            repo_id=HF_REPO,
            repo_type="dataset",
            token=os.getenv("HF_TOKEN"),
            commit_message=f"Update {scrape_date} ({written} items)"
        )
        print("Uploaded to Hugging Face.")

if __name__ == "__main__":
    main()
