import requests
from bs4 import BeautifulSoup
import time
import json
from urllib.parse import urljoin, urlparse
import logging
from collections import deque
import mimetypes
import io
from PyPDF2 import PdfReader
import pdfplumber

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class WebScraper:
    def __init__(self, base_url, delay=1):
        """
        Initialize the scraper with a base URL and delay between requests

        Args:
            base_url (str): The base URL to start scraping from
            delay (int): Delay between requests in seconds to be polite
        """
        self.base_url = base_url.rstrip("/")  # Remove trailing slash
        self.base_parsed = urlparse(base_url)
        self.delay = delay
        self.visited_urls = set()
        self.data = []
        self.url_queue = deque([base_url])
        self.final_text = ""

    def is_pdf_url(self, url):
        """Check if the URL points to a PDF file"""
        # Check URL extension
        if url.lower().endswith(".pdf"):
            return True

        # Check content type from URL
        parsed = urlparse(url)
        path = parsed.path.lower()
        return path.endswith(".pdf")

    def extract_pdf_text(self, pdf_content):
        """Extract text from PDF content"""
        try:
            pdf_file = io.BytesIO(pdf_content)
            pdf_reader = PdfReader(pdf_file)
            text = ""

            for page in pdf_reader.pages:
                text += page.extract_text() + "\n"

            return text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""

    def format_table_as_markdown(self, table, headers=None):
        """Format a table as markdown string"""
        if not table:
            return ""

        # Replace None with empty strings
        cleaned_table = [
            [cell.strip() if cell else "" for cell in row] for row in table
        ]

        # Use first row as header if not provided
        if not headers:
            headers = cleaned_table[0]
            rows = cleaned_table[1:]
        else:
            rows = cleaned_table

        # Normalize multi-line headers and cells
        headers = [" ".join(col.split()) for col in headers]
        rows = [[" ".join(cell.split()) for cell in row] for row in rows]

        # Build Markdown table
        markdown = "| " + " | ".join(headers) + " |\n"
        markdown += "| " + " | ".join(["---"] * len(headers)) + " |\n"
        for row in rows:
            # pad row to match header length
            padded_row = row + [""] * (len(headers) - len(row))
            markdown += "| " + " | ".join(padded_row) + " |\n"
        return markdown

    def extract_pdf_text_with_tables(self, pdf_content):
        """Extract text and tables from PDF content using pdfplumber"""
        try:
            pdf_file = io.BytesIO(pdf_content)
            processed_tables = set()  # Keep track of processed tables
            all_content = []

            with pdfplumber.open(pdf_file) as pdf:
                for page in pdf.pages:
                    page_content = []

                    # Extract tables first
                    tables = page.extract_tables()
                    for table in tables:
                        if not table:  # Skip empty tables
                            continue

                        # Create a string representation of the table for deduplication
                        table_str = str(table)
                        if table_str not in processed_tables:
                            processed_tables.add(table_str)
                            markdown_table = self.format_table_as_markdown(table)
                            if markdown_table.strip():  # Only add non-empty tables
                                page_content.append(("table", markdown_table))

                    # Extract text after tables
                    page_text = page.extract_text() or ""
                    if page_text.strip():
                        page_content.append(("text", page_text))

                    # Sort content to maintain original order (tables followed by text)
                    page_content.sort(key=lambda x: x[0] != "table")

                    # Add page content to main content list
                    all_content.extend([content for _, content in page_content])

            # Join all content with appropriate spacing
            final_text = "\n".join(all_content)
            logger.info("Successfully extracted text and tables from PDF")
            return final_text.strip()
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""

    def get_page(self, url):
        """Fetch a page and return its content"""
        try:
            headers = {
                "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36"
            }
            response = requests.get(url, headers=headers)
            response.raise_for_status()

            # Check if it's a PDF
            content_type = response.headers.get("content-type", "").lower()
            if "application/pdf" in content_type or self.is_pdf_url(url):
                logger.info(f"Processing PDF file: {url}")
                return self.extract_pdf_text_with_tables(response.content)

            return response.text
        except requests.RequestException as e:
            logger.error(f"Error fetching {url}: {e}")
            return None

    def is_valid_url(self, url):
        """Check if URL should be followed"""
        if not url:
            return False

        # Skip non-HTTP(S) URLs
        if not url.startswith(("http://", "https://")):
            return False

        # Skip URLs with fragments
        if "#" in url:
            url = url.split("#")[0]

        # Skip already visited URLs
        if url in self.visited_urls:
            return False

        # Parse the URL
        parsed = urlparse(url)

        # Check if it's the same domain
        if parsed.netloc != self.base_parsed.netloc:
            return False

        # Check if it's a subpath or query parameter of the base URL
        if not parsed.path.startswith(self.base_parsed.path):
            return False

        return True

    def extract_links(self, html, current_url):
        """Extract all links from the page"""
        if not html:
            return []

        soup = BeautifulSoup(html, "html.parser")
        links = []

        for link in soup.find_all("a", href=True):
            url = urljoin(current_url, link["href"])
            if self.is_valid_url(url):
                links.append(url)

        return links

    def parse_page(self, html, url):
        """Parse the HTML content and extract data"""
        if not html:
            return {}

        # If the content is from a PDF, handle it differently
        if self.is_pdf_url(url):
            return {
                "url": url,
                "title": url.split("/")[-1],  # Use filename as title
                "text": html,  # html variable contains the extracted PDF text
                "links": [],
                "type": "pdf",
            }

        soup = BeautifulSoup(html, "html.parser")

        # Try to find the main article content
        # Common article container classes/ids
        article_containers = soup.find_all(
            ["article", "div"], class_=["article", "story", "content", "main-content"]
        )

        # If no specific article container found, try to find the main content area
        if not article_containers:
            article_containers = [
                soup.find("main") or soup.find("div", class_="main") or soup
            ]

        # Extract text from the first article container found
        main_content = article_containers[0] if article_containers else soup

        # Extract paragraphs and headings, excluding navigation and footer
        content_elements = main_content.find_all(
            ["p", "h1", "h2", "h3", "h4", "h5", "h6"]
        )

        # Clean and join the text
        cleaned_text = " ".join(
            [
                element.get_text(strip=True)
                for element in content_elements
                if element.get_text(strip=True)  # Only include non-empty text
            ]
        )

        # Extract page data
        data = {
            "url": url,
            "title": soup.title.string.strip() if soup.title else "",
            "text": cleaned_text,
            "links": [],
        }

        # Extract all links
        for link in soup.find_all("a", href=True):
            link_url = urljoin(url, link["href"])
            if self.is_valid_url(link_url):
                link_text = link.get_text(strip=True)
                if link_text:  # Only include links with text
                    data["links"].append({"text": link_text, "url": link_url})

        return data

    def scrape(self):
        """Main scraping method that follows links containing the pattern"""
        while self.url_queue:
            current_url = self.url_queue.popleft()

            # Skip if already visited
            if current_url in self.visited_urls:
                logger.info(f"Skipping already visited URL: {current_url}")
                continue

            logger.info(f"Scraping: {current_url}")

            # Add delay between requests
            time.sleep(self.delay)

            # Get and parse the page
            html = self.get_page(current_url)

            # Mark URL as visited regardless of success or failure
            self.visited_urls.add(current_url)

            if html:
                # Extract data from the page
                page_data = self.parse_page(html, current_url)
                self.data.append(page_data)

                # Extract and queue new links
                new_links = self.extract_links(html, current_url)
                for link in new_links:
                    if link not in self.visited_urls:  # Only queue unvisited URLs
                        self.url_queue.append(link)

                logger.info(f"Found {len(new_links)} new links to follow")
            else:
                logger.warning(f"Failed to scrape {current_url} - marked as visited")

        return self.data

    def save_data(self, filename="scraped_data.json"):
        """Save the scraped data to a JSON file"""
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(self.data, f, ensure_ascii=False, indent=2)
        logger.info(f"Data saved to {filename}")

    def _text_generator(self, obj):
        if "title" not in obj or "text" not in obj:
            return
        self.final_text += obj["title"] + "\n"
        self.final_text += obj["text"] + "\n"

        for link in obj["links"]:
            self._text_generator(link)

    def get_final_text(self):
        for obj in self.data:
            self._text_generator(obj)
        return self.final_text


# Example usage
if __name__ == "__main__":
    # Replace with the website you want to scrape
    target_url = "https://example.com"

    scraper = WebScraper(target_url, delay=2)  # 2 second delay between requests
    data = scraper.scrape()
    scraper.save_data()
