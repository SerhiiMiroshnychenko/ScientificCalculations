from base_scraper import BaseDocxScraper

def main():
    start_url = "https://www.odoo.com/documentation/20.0/applications.html"
    scraper = BaseDocxScraper(
        start_url=start_url,
        base_path_prefix="/documentation/20.0/applications",
        output_filename="odoo_20_applications_docs.docx",
        delay=0.8,
        retry=2,
        timeout=30,
        split_by_segment_index=3  # Splits files semantically by path category (e.g. finance, sales, hr)
    )
    scraper.scrape()

if __name__ == "__main__":
    main()
