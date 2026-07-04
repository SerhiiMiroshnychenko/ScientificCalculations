from base_scraper import BaseDocxScraper

def main():
    start_url = "https://www.odoo.com/documentation/19.0/contributing.html"
    scraper = BaseDocxScraper(
        start_url=start_url,
        base_path_prefix="/documentation/19.0/contributing",
        output_filename="odoo_19_contributing_docs.docx",
        delay=0.8,
        retry=2,
        timeout=30
    )
    scraper.scrape()
    scraper.save()

if __name__ == "__main__":
    main()
