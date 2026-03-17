import os
import json
import time
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://www.who.int"
FACT_SHEETS_INDEX = "https://www.who.int/news-room/fact-sheets"
SAVE_DIR = "rag/data/who_guidelines"

os.makedirs(SAVE_DIR, exist_ok=True)

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    )
}


def get_fact_sheet_urls():
    """Scrape the WHO fact-sheets index page and return all detail page URLs."""
    print(f"Fetching index: {FACT_SHEETS_INDEX}")
    resp = requests.get(FACT_SHEETS_INDEX, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    urls = []
    for a in soup.find_all("a", href=True):
        href = a["href"]
        if "/news-room/fact-sheets/detail/" in href:
            full_url = urljoin(BASE_URL, href)
            urls.append(full_url)

    return list(dict.fromkeys(urls))  # deduplicate, preserve order


def parse_and_save(url):
    """Fetch a single fact sheet, extract structured content, and save as JSON."""
    time.sleep(0.8)  # be polite to the server
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    soup = BeautifulSoup(resp.text, "html.parser")

    title_tag = soup.find("h1")
    if not title_tag:
        print(f"  Skipping (no <h1>): {url}")
        return

    data = {
        "url": url,
        "title": title_tag.get_text(strip=True),
        "sections": [],
    }

    # WHO wraps body content in this div; fall back to full page if missing
    body = (
        soup.find("div", {"class": "sf-detail-body-wrapper"})
        or soup.find("div", {"id": "PageContent"})
        or soup
    )

    current_section = {"heading": "Introduction", "content": ""}

    for tag in body.find_all(["h2", "h3", "p", "li"]):
        if tag.name in ("h2", "h3"):
            if current_section["content"].strip():
                data["sections"].append(current_section)
            current_section = {"heading": tag.get_text(strip=True), "content": ""}
        elif tag.name == "p":
            text = tag.get_text(strip=True)
            if text:
                current_section["content"] += text + "\n"
        elif tag.name == "li":
            text = tag.get_text(strip=True)
            if text:
                current_section["content"] += "- " + text + "\n"

    if current_section["content"].strip():
        data["sections"].append(current_section)

    filename = url.rstrip("/").split("/")[-1] + ".json"
    filepath = os.path.join(SAVE_DIR, filename)
    with open(filepath, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

    print(f"  Saved: {filename}")


def main():
    urls = get_fact_sheet_urls()
    print(f"Found {len(urls)} fact sheet URLs\n")

    for i, url in enumerate(urls, 1):
        print(f"[{i}/{len(urls)}] {url}")
        try:
            parse_and_save(url)
        except Exception as e:
            print(f"  Error: {e}")


if __name__ == "__main__":
    main()
