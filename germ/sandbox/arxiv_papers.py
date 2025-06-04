import arxiv
import json
import os
import re
import time

non_alphanum = re.compile(r"[^a-zA-Z0-9]")

client = arxiv.Client(
    num_retries=10
)
download_dir = 'data/arxiv-papers'

# Search for papers
search = arxiv.Search(
    query="language model",
    max_results=150,
    sort_by=arxiv.SortCriterion.SubmittedDate
)

# Download pdf files
for result in client.results(search):
    paper_id = result.pdf_url.split("/")[-1]  # Includes version
    file_basename = f"{paper_id}.{non_alphanum.sub('_', result.title)}"
    meta_filename = f"{file_basename}.json"
    pdf_filename = f"{file_basename}.pdf"

    backoff = 1
    while not os.path.exists(f"{download_dir}/{pdf_filename}"):
        print(f"Downloading: {result.published} {pdf_filename}")
        try:
            result.download_pdf(dirpath='data/arxiv-papers', filename=pdf_filename)
            with open(f"{download_dir}/{meta_filename}", "w") as f:
                json.dump({
                    "authors": [str(a) for a in result.authors],
                    "categories": result.categories,
                    "comment": result.comment,
                    "pdf_url": result.pdf_url,
                    "primary_category": result.primary_category,
                    "published": str(result.published),
                    "summary": result.summary,
                    "title": result.title,
                }, f, indent=4)
                backoff = 1
        except Exception as e:
            print(f"Download failed: {e}")
            print(f"Retrying after: {backoff}s")
            time.sleep(backoff)
            if backoff < 10:
                backoff += 1
