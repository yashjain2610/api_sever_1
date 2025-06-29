from playwright.async_api import async_playwright
import time
import random
import asyncio
import sys
import re
import pandas as pd
import urllib.parse
import json

if sys.platform.startswith('win'):
    asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())


def generate_excel_from_products_flipkart(product_list, filename="flipkart_products.xlsx"):
    """
    Takes a list of product dictionaries and writes them to an Excel file.
    
    Args:
        product_list (list): List of product dictionaries.
        filename (str): Output Excel filename.
    """
    # Normalize bullet_points to string (joining list)
    for product in product_list:
        if isinstance(product.get("bullet_points"), list):
            product["bullet_points"] = "\n• " + "\n• ".join(product["bullet_points"])
    
    # Define the preferred column order (optional)
    column_order = [
        'itm_id','brand','title','price','description','rating','num_of_reviews','url'
    ]
    
    # Create a DataFrame
    df = pd.DataFrame(product_list)

    # Reorder columns if they exist in data
    df = df[[col for col in column_order if col in df.columns]]

    # Write to Excel
    df.to_excel(filename, index=False, engine="openpyxl")

    print(f"Excel file saved as {filename}")

async def scrape_flipkart_items(itm_ids: list[str]) -> list[dict]:
    """
    Given a list of Flipkart itm IDs (e.g. "itm63daa46b80953"),
    visits each https://www.flipkart.com/l/p/{itm_id} page and
    scrapes title, brand, and price according to the markup you provided.
    Returns a list of dicts: [{ "itm_id": ..., "brand": ..., "title": ..., "price": ... }, ...]
    """
    results = []

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        context = await browser.new_context()

        for itm in itm_ids:
            page = await context.new_page()
            url = f"https://www.flipkart.com/l/p/{itm}"
            try:
                await page.goto(url, timeout=30000)
                # wait for the main h1 to appear
                await page.wait_for_selector("h1._6EBuvT", timeout=15000)

                # brand is in the first span inside the H1
                brand = (await page.locator("h1._6EBuvT span.mEh187").inner_text()).strip()

                # the actual product title is in the second span
                title = (await page.locator("h1._6EBuvT span.VU-ZEz").inner_text()).strip()

                # price is inside the div with those two classes
                #price = (await page.locator("div.Nx9bqj.CxhGGd").inner_text()).strip()
                price = (
                    await page
                    .locator("div.Nx9bqj.CxhGGd")
                    .inner_text()
                ).strip()

                description = (
                    await page.locator("div._4aGEkW").inner_text()
                ).strip()
                try:
                    rating = (
                        await page
                        .locator("div.XQDdHH._6er70b")
                        .inner_text()
                    ).strip()
                except Exception:
                    rating = "no rating"

                # number of reviews (fallback to "no reviews")
                try:
                    num_of_reviews = (
                        await page
                        .locator("span.Wphh3N.d4OmzS")
                        .inner_text()
                    ).strip()
                except Exception:
                    num_of_reviews = "no reviews"

                results.append({
                    "itm_id": itm,
                    "brand": brand,
                    "title": title,
                    "price": price,
                    "description": description,
                    "rating": rating,       
                    "num_of_reviews": num_of_reviews,
                    "url": url
                })

            except Exception as e:
                results.append({
                    "itm_id": itm,
                    "error": str(e)
                })
            finally:
                await page.close()

        await browser.close()

    return results


async def get_flipkart_rank(target_itm: str,
                           search_query: str,
                           max_pages: int = 4) -> dict:
    """
    Flipkart search rank via direct anchor scan:
      - target_itm: e.g. "itm63daa46b80953"
      - search_query: your query
      - max_pages: how many pages to scan

    Returns {"page": n, "position_on_page": k} or not found.
    """
    base_url = "https://www.flipkart.com/search?q="
    query = urllib.parse.quote_plus(search_query)
    start_url = f"{base_url}{query}"

    async with async_playwright() as p:
        ua_list = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...",
        ]
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-dev-shm-usage"]
        )
        context = await browser.new_context(user_agent=random.choice(ua_list))
        page = await context.new_page()

        for page_num in range(1, max_pages + 1):
            url = f"{start_url}&page={page_num}"
            print(f"[Flipkart] Checking page {page_num}: {url}")
            await page.goto(url, timeout=60_000)

            # wait for at least one product link
            try:
                await page.wait_for_selector("a.rPDeLR", timeout=20_000)
            except:
                print(f"[Flipkart] No product links on page {page_num}")
                continue

            # grab and enumerate all product anchors
            anchors = await page.query_selector_all("a.rPDeLR")
            for idx, a in enumerate(anchors, start=1):
                href = await a.get_attribute("href")  # e.g. "/.../p/itmXYZ?pid=..."
                if href and f"/p/{target_itm}" in href:
                    await browser.close()
                    return {"page": page_num, "position_on_page": idx}

        await browser.close()
        return {"page": "not found", "position_on_page": "not found"}


if __name__ == "__main__":
    pids = ["itm63daa46b80953","itm917c4db3f463e"]
    results = asyncio.run(scrape_flipkart_items(pids))
    generate_excel_from_products_flipkart(results)
    print(results)
    #print(asyncio.run(get_flipkart_rank("itmafa4c4a469cef", "earrings for women")))