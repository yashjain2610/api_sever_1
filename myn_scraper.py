import re
import asyncio
import random
import pandas as pd
from playwright.async_api import async_playwright

def generate_excel_from_products_myntra(product_list, filename="myntra_products.xlsx"):
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
        'product_id','brand','title','price','description','num_of_imgs','num_of_videos','rating','ratings_count','img_url','url'
    ]
    
    # Create a DataFrame
    df = pd.DataFrame(product_list)

    # Reorder columns if they exist in data
    df = df[[col for col in column_order if col in df.columns]]

    # Write to Excel
    df.to_excel(filename, index=False, engine="openpyxl")

    print(f"Excel file saved as {filename}")



async def scrape_myntra_items(product_ids: list[str]) -> list[dict]:
    """
    Given a list of Myntra numeric product IDs (e.g. "29287056"),
    visits each https://www.myntra.com/a/a/a/{id}/buy page and
    scrapes brand, title, rating and ratings count.
    """
    results = []
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=False,
            args=["--disable-http2", "--no-sandbox", "--disable-dev-shm-usage"]
        )

        user_agents = [
            # Windows Chrome (latest-ish)
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36"
        ]

        context = await browser.new_context(user_agent=random.choice(user_agents))
        
        for pid in product_ids:
            page = await context.new_page()
            url = f"https://www.myntra.com/{pid}?rawQuery={pid}"
            try:
                try:
                    await page.goto(url, wait_until="domcontentloaded")
                except Exception as e:
                    print(f"Warning: navigate timeout: {e}")

                # html = await page.content()  # ← here's your HTML snapshot

                # with open("ec2_mynid.html", "w", encoding="utf-8") as f:
                #     f.write(html)
                # wait for the brand header
                await page.wait_for_selector("h1.pdp-title", timeout=15_000)

                # scrape brand & title
                brand = (await page.locator("h1.pdp-title").inner_text()).strip()
                title = (await page.locator("h1.pdp-name").inner_text()).strip()

                price = "no price"
                price_locator = page.locator("p.pdp-discount-container span.pdp-price strong")
                if await price_locator.count():
                    price = (await price_locator.first.inner_text()).strip()

                
                description = "N/A"
                desc_el = page.locator("p.pdp-product-description-content")
                if await desc_el.count():
                    description = (await desc_el.first.inner_text()).strip()

                # Count image thumbnails
                num_images = await page.locator("div.image-grid-col50").count()

                # Count video thumbnails
                num_videos = await page.locator("div.brightcove-video-col50").count()

                # Wait for at least one image container to appear
                await page.wait_for_selector("div.image-grid-col50 div.image-grid-image", timeout=10000)

                # Grab the `style` attribute from the first image div
                style_attr = await page.locator("div.image-grid-col50 div.image-grid-image").first.get_attribute("style")

                # Extract the URL from the `background-image: url("…")` string
            
                match = re.search(r'url\("([^"]+)"\)', style_attr or "")
                img_url = match.group(1) if match else None

                # rating (e.g. "4.6")
                rating = "no rating"
                rl = page.locator("div.index-overallRating > div")
                if await rl.count():
                    rating = (await rl.first.inner_text()).strip()

                # ratings count (e.g. "21 Ratings")
                ratings_count = "0"
                rc = page.locator("div.index-ratingsCount")
                if await rc.count():
                    txt = (await rc.first.inner_text()).strip()
                    m = re.search(r"(\d+)", txt)
                    if m:
                        ratings_count = m.group(1)

                results.append({
                    "product_id": pid,
                    "brand": brand,
                    "title": title,
                    "price": price,
                    "description": description,
                    "num_of_imgs": num_images,
                    "num_of_videos": num_videos,
                    "rating": rating,
                    "ratings_count": ratings_count,
                    "img_url": img_url,
                    "url": url
                })

            except Exception as e:
                results.append({
                    "product_id": pid,
                    "error": str(e)
                })
            finally:
                await page.close()

        await browser.close()
    return results


if __name__ == "__main__":
    ids = ["29287056"]
    print(asyncio.run(scrape_myntra_items(ids)))