# scripts/sgd_crawler.py

import os
import requests
import zipfile
import io
from lxml import etree
import pandas as pd
from tqdm import tqdm
import time
from urllib.parse import urljoin

# Base URL for the repository
BASE_URL = "https://repository.overheid.nl/frbr/sgd"
# The source name for the data
SOURCE_NAME = "Statengeneraal Digitaal"
# Directory to save the data
DATA_DIR = "data"
# Number of documents to save per file
BATCH_SIZE = 500
# User-Agent to identify the crawler
HEADERS = {
    'User-Agent': 'vGassen/Dutch-Statengeneraal-Digitaal-Historical Crawler'
}

def get_year_links():
    """
    Fetches the links to the year directories from the base URL.
    This is a simple parsing of the main directory page.
    """
    try:
        response = requests.get(BASE_URL, headers=HEADERS)
        response.raise_for_status()
        # A simple way to find links that look like years
        links = etree.HTML(response.content).xpath('//a/@href')
        year_links = []
        for link in links:
            # Assuming year directories are numeric or in a specific format
            year = link.strip('/').split('/')[-1]
            if year.isdigit() and len(year) == 4:
                 year_links.append(f"{BASE_URL}/{year}/")
        return sorted(list(set(year_links)))
    except requests.exceptions.RequestException as e:
        print(f"Error fetching year links: {e}")
        return []

def get_work_links(year_url):
    """
    Fetches links to the individual 'works' for a given year.
    """
    try:
        response = requests.get(year_url, headers=HEADERS)
        response.raise_for_status()
        links = etree.HTML(response.content).xpath('//a/@href')
        work_links = []
        for link in links:
            full_link = urljoin(year_url, link)
            if not full_link.startswith(year_url):
                continue
            suffix = full_link[len(year_url):].strip('/')
            if suffix.isdigit():
                if not full_link.endswith('/'):
                    full_link += '/'
                work_links.append(full_link)
        return sorted(list(set(work_links)))
    except requests.exceptions.RequestException as e:
        print(f"Error fetching work links from {year_url}: {e}")
        return []

def process_work(work_url):
    """
    Downloads the zip archive for a work, extracts OCR XML files,
    and returns the formatted data.
    """
    zip_url = work_url.rstrip('/') + '/?format=zip'
    extracted_data = []

    try:
        response = requests.get(zip_url, headers=HEADERS, stream=True)
        response.raise_for_status()

        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            for filename in z.namelist():
                if filename.startswith('ocr/') and filename.endswith('.xml'):
                    with z.open(filename) as xml_file:
                        try:
                            # Parse the XML content
                            tree = etree.parse(xml_file)
                            # A simple approach to get all text content
                            text_content = " ".join(tree.xpath('//text()')).strip()
                            text_content = " ".join(text_content.split()) # Normalize whitespace

                            if text_content:
                                file_url = f"{zip_url}#{filename}"
                                extracted_data.append({
                                    "URL": file_url,
                                    "content": text_content,
                                    "source": SOURCE_NAME
                                })
                        except etree.XMLSyntaxError as e:
                            print(f"Error parsing XML file {filename} in {zip_url}: {e}")

    except requests.exceptions.RequestException as e:
        print(f"Error downloading or processing zip from {zip_url}: {e}")
    except zipfile.BadZipFile:
        print(f"Error: Bad zip file at {zip_url}")

    return extracted_data


def main():
    """
    Main function to orchestrate the crawling process.
    """
    if not os.path.exists(DATA_DIR):
        os.makedirs(DATA_DIR)

    print("Starting crawl of Statengeneraal Digitaal...")
    all_docs = []
    batch_counter = 1

    year_urls = get_year_links()
    if not year_urls:
        print("Could not find any years to process. Exiting.")
        return

    # For demonstration, let's limit to a few years.
    # Remove this slicing to crawl everything.
    year_urls = year_urls[-2:] # Example: crawl the last 2 years found

    for year_url in tqdm(year_urls, desc="Processing Years"):
        work_links = get_work_links(year_url)
        for work_url in tqdm(work_links, desc=f"Processing Works in {year_url.split('/')[-2]}", leave=False):
            docs = process_work(work_url)
            all_docs.extend(docs)

            if len(all_docs) >= BATCH_SIZE:
                df = pd.DataFrame(all_docs)
                filename = os.path.join(DATA_DIR, f"sgd_batch_{batch_counter:03d}.csv")
                df.to_csv(filename, index=False)
                print(f"Saved batch to {filename}")
                all_docs = []
                batch_counter += 1
            
            # Being a good web citizen
            time.sleep(0.1) 

    # Save any remaining documents
    if all_docs:
        df = pd.DataFrame(all_docs)
        filename = os.path.join(DATA_DIR, f"sgd_batch_{batch_counter:03d}.csv")
        df.to_csv(filename, index=False)
        print(f"Saved final batch to {filename}")

    print("Crawling finished.")

if __name__ == "__main__":
    main()
