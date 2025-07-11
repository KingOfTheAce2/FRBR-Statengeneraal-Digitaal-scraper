import os
import json
import logging
import requests
from lxml import etree, html
from huggingface_hub import HfApi

# Configuration (can be overridden via environment variables)
SRU_URL            = os.getenv("SRU_URL", "https://repository.overheid.nl/sru")
CQL_QUERY          = os.getenv("CQL_QUERY", "c.product-area==sgd")
SRU_VERSION        = os.getenv("SRU_VERSION", "2.0")
BATCH_SIZE         = int(os.getenv("BATCH_SIZE", "100"))
STATE_FILE         = os.getenv("STATE_FILE", "state.json")
OUTPUT_FILE        = os.getenv("OUTPUT_FILE", "sgd.jsonl")
HF_DATASET_REPO    = os.getenv("HF_DATASET_REPO")  # Hugging Face dataset repo ID
HF_TOKEN           = os.getenv("HF_TOKEN")

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s: %(message)s"
)


def load_state():
    if os.path.exists(STATE_FILE):
        try:
            return json.load(open(STATE_FILE, encoding="utf-8"))
        except Exception:
            logging.warning("Could not read state file; starting from scratch.")
    return {"start": 1}


def save_state(state):
    with open(STATE_FILE, "w", encoding="utf-8") as f:
        json.dump(state, f)


def strip_html(raw_html: str) -> str:
    """
    Strip HTML tags, scripts, styles to extract visible text.
    """
    doc = html.fromstring(raw_html)
    for bad in doc.xpath('//script|//style'):
        bad.drop_tree()
    return doc.text_content().strip()


def fetch_and_process():
    state = load_state()
    start = state.get("start", 1)
    total = None

    with open(OUTPUT_FILE, "a", encoding="utf-8") as out_f:
        while True:
            params = {
                "version": SRU_VERSION,
                "operation": "searchRetrieve",
                "query": CQL_QUERY,
                "startRecord": start,
                "maximumRecords": BATCH_SIZE,
            }
            logging.info(f"Request SRU batch: start={start}")
            try:
                resp = requests.get(SRU_URL, params=params, timeout=30)
                resp.raise_for_status()
            except Exception as e:
                logging.error(f"Failed SRU request: {e}")
                break

            root = etree.fromstring(resp.content)

            # On first batch, read total count if available
            if total is None:
                sub = root.find('.//{http://www.w3.org/2005/Atom}subtitle')
                if sub is not None and ':' in sub.text:
                    total = int(sub.text.split(':', 1)[1])
                    logging.info(f"Total records to fetch: {total}")

            records = root.findall('.//{*}gzd')
            if not records:
                logging.info("No more records returned; exiting loop.")
                break

            for rec in records:
                # Use full XPath for element functions
                urls = rec.xpath(".//*[local-name()='preferredUrl']") or rec.xpath(".//*[local-name()='itemUrl']")
                for u in urls:
                    url = u.text.strip() if u.text else None
                    if not url:
                        continue
                    logging.info(f"Fetching content: {url}")
                    try:
                        dr = requests.get(url, timeout=30)
                        dr.raise_for_status()
                        content_type = dr.headers.get('Content-Type', '')

                        if 'html' in content_type or 'xml' in content_type:
                            text = strip_html(dr.text)
                        else:
                            logging.info(f"Skipping unsupported content-type: {content_type}")
                            continue

                        if not text:
                            logging.info("Empty content after stripping; skipping.")
                            continue

                        obj = {"URL": url, "Content": text, "Source": "SGD"}
                        out_f.write(json.dumps(obj, ensure_ascii=False) + "\n")
                    except Exception as e:
                        logging.warning(f"Failed to fetch/process {url}: {e}")
                        continue

            # Advance state
            start += len(records)
            state['start'] = start
            save_state(state)
            logging.info(f"Batch complete; next start={start}")

            if total and start > total:
                logging.info("Reached or exceeded total count; done.")
                break

    return OUTPUT_FILE


def upload_to_hf(filepath: str):
    if not HF_DATASET_REPO or not HF_TOKEN:
        logging.warning("HF_DATASET_REPO or HF_TOKEN not set; skipping upload.")
        return
    api = HfApi()
    try:
        api.create_repo(HF_DATASET_REPO, repo_type="dataset", token=HF_TOKEN, exist_ok=True)
        api.upload_file(
            path_or_fileobj=filepath,
            path_in_repo=os.path.basename(filepath),
            repo_id=HF_DATASET_REPO,
            repo_type="dataset",
            token=HF_TOKEN
        )
        logging.info(f"Uploaded {filepath} to Hugging Face dataset {HF_DATASET_REPO}")
    except Exception as e:
        logging.error(f"Failed to upload to HF: {e}")

if __name__ == '__main__':
    logging.info("Starting SGD crawler.")
    out_path = fetch_and_process()
    logging.info(f"Data saved to {out_path}")
    upload_to_hf(out_path)
