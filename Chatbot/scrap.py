"""
Stock Data Scraper for Screener.in
Fetches comprehensive financial data for Indian stocks (NSE/BSE)
"""
import requests
from bs4 import BeautifulSoup

# User agent to make our requests look like a browser
USER_AGENT = {"User-Agent": "Mozilla/5.0"}


def fetch_stock_page(symbol: str):
    """
    Fetches the HTML page for a given stock symbol from Screener.in
    Returns a BeautifulSoup object for parsing
    """
    url = f"https://www.screener.in/company/{symbol}/consolidated/"
    response = requests.get(url, headers=USER_AGENT, timeout=20)
    response.raise_for_status()  # Raise error if request fails
    return BeautifulSoup(response.text, "html.parser")


def clean_text(element):
    """
    Extracts and cleans text from an HTML element.
    Returns None if element doesn't exist.
    """
    return element.get_text(strip=True) if element else None


def parse_html_table(table):
    """
    Converts an HTML table into a structured dictionary format.
    Returns columns (headers) and rows (data).
    """
    # Extract headers
    headers = [clean_text(th) for th in table.select("thead th")]
    
    # Extract data rows
    rows = []
    for tr in table.select("tbody tr"):
        row_data = [clean_text(td) for td in tr.find_all(["td", "th"])]
        rows.append(row_data)
    
    return {"columns": headers, "rows": rows}


def extract_section_by_id(soup, section_id: str):
    """
    Extracts a specific section from the page by its ID.
    Returns parsed table data or None if not found.
    """
    section = soup.select_one(f"section#{section_id} table")
    if not section:
        return None
    return parse_html_table(section)


def extract_shareholding_data(soup):
    """
    Extracts shareholding pattern data (both quarterly and yearly).
    Returns a dictionary with quarterly and yearly shareholding tables.
    """
    result = {"quarterly": [], "yearly": []}
    
    # Quarterly shareholding
    quarterly_table = soup.select_one("#quarterly-shp table")
    if quarterly_table:
        result["quarterly"].append(parse_html_table(quarterly_table))
    
    # Yearly shareholding
    yearly_table = soup.select_one("#yearly-shp table")
    if yearly_table:
        result["yearly"].append(parse_html_table(yearly_table))
    
    return result


def parse_growth_ranges(soup):
    """
    Parses the growth ranges section (sales growth, profit growth, etc.).
    This section has multiple small tables that we combine into one.
    """
    tables = soup.select("table.ranges-table")
    if not tables:
        return None
    
    # Collect all data points
    data = {}
    periods = []
    
    for table in tables:
        header = clean_text(table.select_one("th"))
        rows = table.select("tr")[1:]  # Skip the header row
        
        for row in rows:
            cells = row.find_all("td")
            if len(cells) == 2:
                period = clean_text(cells[0]).rstrip(':')  # Remove trailing colon
                value = clean_text(cells[1])
                
                if period not in data:
                    data[period] = {}
                    periods.append(period)
                
                data[period][header] = value
    
    # Build a combined table
    headers = [clean_text(t.select_one("th")) for t in tables]
    combined_rows = [
        [period] + [data[period].get(h, "—") for h in headers] 
        for period in periods
    ]
    
    return {"columns": ["Period"] + headers, "rows": combined_rows}


def scrape(symbol: str):
    """
    Main scraping function - fetches ALL financial data for a stock.
    
    Args:
        symbol: NSE stock symbol (e.g., 'TITAN', 'RELIANCE', 'ASIANPAINT')
    
    Returns:
        dict: Complete financial data including:
            - Quarterly results
            - Profit & Loss statement
            - Growth ranges (1yr, 3yr, 5yr, 10yr)
            - Balance sheet
            - Cash flows
            - Financial ratios
            - Shareholding pattern (quarterly & yearly)
    
    Raises:
        ValueError: If stock not found or if there's a network error
    """
    try:
        soup = fetch_stock_page(symbol)
        
        return {
            "symbol": symbol,
            "quarterly": [extract_section_by_id(soup, "quarters")] 
                if extract_section_by_id(soup, "quarters") else [],
            "profit_loss": [extract_section_by_id(soup, "profit-loss")] 
                if extract_section_by_id(soup, "profit-loss") else [],
            "growth_ranges": [parse_growth_ranges(soup)] 
                if parse_growth_ranges(soup) else [],
            "balance_sheet": [extract_section_by_id(soup, "balance-sheet")] 
                if extract_section_by_id(soup, "balance-sheet") else [],
            "cash_flows": [extract_section_by_id(soup, "cash-flow")] 
                if extract_section_by_id(soup, "cash-flow") else [],
            "ratios": [extract_section_by_id(soup, "ratios")] 
                if extract_section_by_id(soup, "ratios") else [],
            "shareholding": extract_shareholding_data(soup)
        }
    
    except requests.exceptions.HTTPError as e:
        if e.response.status_code == 404:
            raise ValueError(
                f"Stock symbol '{symbol}' not found on Screener.in. "
                f"Please provide a valid NSE/BSE stock symbol."
            )
        raise ValueError(f"HTTP error occurred: {e}")
    
    except requests.exceptions.RequestException as e:
        raise ValueError(f"Network error - couldn't reach Screener.in: {e}")
    
    except Exception as e:
        raise ValueError(f"Error scraping stock data: {e}")