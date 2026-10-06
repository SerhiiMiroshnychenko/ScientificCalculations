from base_scraper import BaseDocxScraper

def main():
    start_url = "https://www.odoo.com/documentation/20.0/administration.html"
    scraper = BaseDocxScraper(
        start_url=start_url,
        base_path_prefix="/documentation/20.0/administration",
        output_filename="odoo_20_administration_docs.docx",
        delay=0.8,
        retry=2,
        timeout=30
    )
    scraper.scrape()
    scraper.save()

if __name__ == "__main__":
    main()
