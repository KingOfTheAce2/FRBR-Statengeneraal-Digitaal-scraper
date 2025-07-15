import os
import json
import logging
import requests
from urllib.parse import urljoin
from lxml import etree, html
from huggingface_hub import HfApi

# Configuration (can be overridden via environment variables)
SRU_URL               = os.getenv("SRU_URL", "https://repository.overheid.nl/sru")
CQL_QUERY             = os.getenv("CQL_QUERY", "c.product-area==sgd")
SRU_VERSION           = os.getenv("SRU_VERSION", "2.0")
SRU_BATCH_SIZE        = int(os.getenv("SRU_BATCH_SIZE", "100"))
MAX_RECORDS_PER_RUN   = int(os.getenv("MAX_RECORDS_PER_RUN", "5000"))
SHARD_SIZE            = int(os.getenv("SHARD_SIZE", "250"))
STATE_FILE            = os.getenv("STATE_FILE", "state.json")
DATA_DIR              = os.getenv("DATA_DIR", "data")
HF_DATASET_REPO       = os.getenv("HF_DATASET_REPO")  # Hugging Face dataset repo ID
HF_TOKEN              = os.getenv("HF_TOKEN")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)

def load_state():
    if os.path.exists(STATE_FILE):
        try:
            state = json.load(open(STATE_FILE, encoding="utf-8"))
        except Exception:
            logging.warning("Could not read state file; starting from scratch.")
            state = {}
    else:
        state = {}

    # Defaults for new keys
    if "start" not in state:
        state["start"] = 1
    if "next_shard" not in state:
        try:
            existing = [
                int(f.split("_")[-1].split(".")[0])
                for f in os.listdir(DATA_DIR)
                if f.startswith("sgd_shard_") and f.endswith(".jsonl")
            ]
            state["next_shard"] = max(existing) + 1 if existing else 1
        except FileNotFoundError:
            state["next_shard"] = 1
    return state


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f)


def strip_text(raw):
    """Extract visible text from HTML or XML input."""
    if isinstance(raw, (bytes, bytearray)):
        # Parse XML bytes directly, avoiding unicode+declaration issues
        tree = etree.fromstring(raw)
        text = ''.join(tree.itertext())
        return text.strip()
    else:
        # HTML input as str
        tree = html.fromstring(raw)
        for bad in tree.xpath('//script|//style'):
            bad.drop_tree()
        return tree.text_content().strip()


def _download_from_ocr_page(ocr_page: str) -> bytes:
    resp = requests.get(ocr_page, timeout=30)
    resp.raise_for_status()
    doc = html.fromstring(resp.content)
    # find XML download link
    link = doc.xpath("//ul[contains(@class,'list--sources')]//a[contains(@class,'button--primary')]/@href")
    if not link:
        raise ValueError(f"No OCR XML link found on {ocr_page}")
    xml_url = urljoin(ocr_page, link[0])
    logging.info(f"Downloading XML: {xml_url}")
    xresp = requests.get(xml_url, timeout=30)
    xresp.raise_for_status()
    return xresp.content


def fetch_ocr_xml(item_url: str) -> bytes:
    """Given an item URL ending in '/1', fetch its OCR XML as bytes, with PDF fallback."""
    ocr_page = item_url.rstrip('/') + '/ocr'
    try:
        return _download_from_ocr_page(ocr_page)
    except requests.HTTPError as he:
        if he.response.status_code != 404:
            raise
        logging.info(f"Direct OCR not found for {item_url}, trying PDF fallback.")
    except ValueError:
        logging.info(f"Direct OCR link missing on {item_url}, trying PDF fallback.")

    # Fallback: parse PDF link from item landing page
    resp = requests.get(item_url, timeout=30)
    resp.raise_for_status()
    page = html.fromstring(resp.content)
    pdf_link = page.xpath("//div[contains(@class,'alert__inner')]//a[contains(@href,'/frbr/sgd/') and contains(@href,'/pdf/')]/@href")
    if not pdf_link:
        raise ValueError(f"No PDF link found on landing page {item_url}")
    pdf_url = urljoin(item_url, pdf_link[0])
    base = pdf_url.split('/pdf/')[0]
    fallback_ocr = base + '/ocr'
    logging.info(f"Fallback OCR page: {fallback_ocr}")
    return _download_from_ocr_page(fallback_ocr)


def fetch_and_process(state):
    os.makedirs(DATA_DIR, exist_ok=True)
    start = state.get("start", 1)
    total = None
    processed = 0
    new_records = []

    while processed < MAX_RECORDS_PER_RUN:
        params = {
            "version": SRU_VERSION,
            "operation": "searchRetrieve",
            "query": CQL_QUERY,
            "startRecord": start,
            "maximumRecords": SRU_BATCH_SIZE,
        }
        logging.info(f"Request SRU batch: start={start}")
        resp = requests.get(SRU_URL, params=params, timeout=30)
        resp.raise_for_status()
        root = etree.fromstring(resp.content)

        if total is None:
            sub = root.find('.//{http://www.w3.org/2005/Atom}subtitle')
            if sub is not None and ':' in sub.text:
                total = int(sub.text.split(':', 1)[1])
                logging.info(f"Total records to fetch: {total}")

        records = root.findall('.//{*}gzd')
        if not records:
            logging.info("No more records returned; exiting.")
            break

        for rec in records:
            if processed >= MAX_RECORDS_PER_RUN:
                break
            urls = rec.xpath(".//*[local-name()='preferredUrl']/text()") or rec.xpath(".//*[local-name()='itemUrl']/text()")
            if not urls:
                continue
            item_url = urls[0].strip()
            try:
                xml_bytes = fetch_ocr_xml(item_url)
                text = strip_text(xml_bytes)
                if not text:
                    logging.info(f"Empty text for {item_url}; skipping.")
                    continue
                new_records.append({
                    "URL": item_url + "/ocr",
                    "Content": text,
                    "Source": "Staten-Generaal Digitaal"
                })
                processed += 1
                logging.info(f"Processed record {processed}/{MAX_RECORDS_PER_RUN}")
            except Exception as e:
                logging.warning(f"Skipping {item_url}: {e}")

        start += len(records)
        state['start'] = start
        save_state(state)
        if total and start > total:
            logging.info("Reached total count; done.")
            break

    logging.info(f"Finished fetching {processed} records this run.")
    return new_records


def write_shards(records, state):
    shard_files = []
    for idx in range(0, len(records), SHARD_SIZE):
        shard = records[idx: idx + SHARD_SIZE]
        shard_num = state["next_shard"] + idx // SHARD_SIZE
        fname = os.path.join(DATA_DIR, f"sgd_shard_{shard_num}.jsonl")
        with open(fname, "w", encoding="utf-8") as f:
            for obj in shard:
                f.write(json.dumps(obj, ensure_ascii=False) + "\n")
        shard_files.append(fname)
    state["next_shard"] += len(shard_files)
    save_state(state)
    return shard_files


def upload_shards():
    if not HF_DATASET_REPO or not HF_TOKEN:
        logging.warning("HF_DATASET_REPO or HF_TOKEN not set; skipping upload.")
        return
    api = HfApi()
    api.create_repo(HF_DATASET_REPO, repo_type="dataset", token=HF_TOKEN, exist_ok=True)
    shard_files = sorted(
        [
            os.path.join(DATA_DIR, f)
            for f in os.listdir(DATA_DIR)
            if f.startswith("sgd_shard_") and f.endswith(".jsonl")
        ]
    )
    for shard in shard_files:
        logging.info(f"Uploading shard {shard}")
        api.upload_file(
            path_or_fileobj=shard,
            path_in_repo=os.path.basename(shard),
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN
        )


if __name__ == '__main__':
    logging.info("Starting SGD crawler.")
    state = load_state()
    records = fetch_and_process(state)
    new_shards = []
    if records:
        new_shards = write_shards(records, state)
    upload_shards()
    if new_shards:
        logging.info(f"Uploaded {len(new_shards)} new shard(s) to Hugging Face.")
    else:
        logging.info("No new records to process this run.")
