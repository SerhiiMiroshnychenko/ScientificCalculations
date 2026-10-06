from base_scraper import BaseDocxScraper

def main():
    start_url = "https://owl-docs.vercel.app/"
    scraper = BaseDocxScraper(
        start_url=start_url,
        base_path_prefix=None,  # Crawl the whole owl-docs domain
        output_filename="owl_docs.docx",
        delay=0.8,
        retry=2,
        timeout=30
    )
    scraper.scrape()
    scraper.save()

if __name__ == "__main__":
    main()
