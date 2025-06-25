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

def generate_excel_from_products(product_list, filename="amazon_products.xlsx"):
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
        "asin","brand", "title", "link", "price", "rating", "review_count", "original_price", "image_url", "num_images", "num_videos",
        "date_first_available", "best_seller_rank", "A_plus_content","brand_store","badge","limited_time_deal",
        "bullet_points", "product_description" 
    ]
    
    # Create a DataFrame
    df = pd.DataFrame(product_list)

    # Reorder columns if they exist in data
    df = df[[col for col in column_order if col in df.columns]]

    # Write to Excel
    df.to_excel(filename, index=False, engine="openpyxl")

    print(f"Excel file saved as {filename}")


def read_asins_from_excel(file_path = "asins.xlsx"):
    """
    Reads an Excel file and returns a list of ASINs from the 'ASIN' column.
    """
    df = pd.read_excel(file_path, engine='openpyxl')
    
    # Drop any rows with NaN ASINs and strip spaces
    asin_list = df['ASINS'].dropna().astype(str).str.strip().tolist()
    
    return asin_list

def get_amazon_search_url(query: str) -> str:
    base_url = "https://www.amazon.in/s"
    return f"{base_url}?k={query.replace(' ', '+')}"

def scrape_amazon_products(query: str, url = None,max_results: int = 100, max_pages: int = 3) -> list:
    if url:
        search_url = url
    else:
        search_url = get_amazon_search_url(query)
        print(f"Searching Amazon for: {query}")
        print(f"URL: {search_url}")

    products = []
    page_count = 0

    with sync_playwright() as p:
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...",
            # Add more user agents
        ]

        viewport_sizes = [
            {"width": 1920, "height": 1080},
            {"width": 1366, "height": 768},
            # Add more viewport sizes
        ]

        selected_user_agent = random.choice(user_agents)
        selected_viewport = random.choice(viewport_sizes)

        browser = p.chromium.launch(headless=False)
        context = browser.new_context(
            user_agent=selected_user_agent,
            viewport=selected_viewport
        )
        page = context.new_page()

        page.goto(search_url, timeout=60000)

        while len(products) < max_results and page_count < max_pages:

            page_count += 1
            
            try:  
                page.wait_for_selector("div[data-component-type='s-search-result']", timeout=10000)
                page.wait_for_timeout(2000)
                time.sleep(random.uniform(2, 4))

                product_elements = page.query_selector_all("div[data-component-type='s-search-result']")

                print(f"Found {len(product_elements)} products")

                for product in product_elements[:max_results]:
                    #print(product.inner_html())
                        try:
                            # brand
                            brand_el = (product.query_selector("span.a-size-medium") or
                                        product.query_selector("span.a-size-base-plus") or
                                        product.query_selector("span.a-text-normal"))
                            brand = brand_el.inner_text().strip() if brand_el else "N/A"

                            # Title
                            title_el = product.query_selector("a.a-link-normal.s-line-clamp-2.s-link-style span")
                            title = title_el.inner_text().strip() if title_el else "N/A"

                            # Link
                            link_el = (product.query_selector("a.a-link-normal.s-no-outline") or
                                    product.query_selector("a.a-link-normal"))
                            link = link_el.get_attribute("href") if link_el else None
                            full_link = f"https://www.amazon.in{link}" if link else "N/A"

                            # Price
                            actual_price_el = product.query_selector('span.a-price[data-a-color="base"] span.a-offscreen')
                            price = actual_price_el.inner_text().strip() if actual_price_el else "N/A"

                            # Rating
                            rating_el = product.query_selector("span.a-icon-alt")
                            rating = rating_el.inner_text().split()[0] if rating_el else "N/A"

                            # Review Count
                            review_count_el = (product.query_selector("span.a-size-base.s-underline-text") or
                                            product.query_selector("span.a-size-base"))
                            review_count = review_count_el.inner_text().strip() if review_count_el else "N/A"

                            # Image URL
                            image_el = product.query_selector("img.s-image")
                            image_url = image_el.get_attribute("src") if image_el else "N/A"

                            # Delivery Info
                            # delivery_el = (product.query_selector("span.a-color-base.a-text-bold") or
                            #             product.query_selector("span.a-color-secondary"))
                            # delivery = delivery_el.inner_text().strip() if delivery_el else "N/A"

                            # # Original Price and Discount
                            discount_el = product.query_selector("span.a-price.a-text-price span.a-offscreen")
                            original_price = discount_el.inner_text().strip() if discount_el else "N/A"
                            # has_discount = "Yes" if discount_el else "No"

                            products.append({
                                "brand": brand,
                                "title": title,
                                "link": full_link,
                                "price": price,
                                "rating": rating,
                                "review_count": review_count,
                                "image_url": image_url,
                                "original_price": original_price
                            })

                            time.sleep(random.uniform(1.5, 3.0))  # Mimic human delay


                        except Exception as e:
                            print(f"Error parsing product: {e}")
                
                # Try to click "Next"
                next_button = page.query_selector("a.s-pagination-next:not(.s-pagination-disabled)")
                if next_button:
                    next_button.click()
                    page.wait_for_timeout(3000)
                else:
                    print("No more pages available.")
                    break
            except Exception as e:
                print(f"Failed to scrape page {page_count}: {e}")
                break

        browser.close()

    return products


async def scrape_amazon_product_detail(page,product_partial: dict,detail_url = None) -> dict:
    """
    Phase 2: Given a Playwright page and a partial product dict,
    navigate to the detail URL and scrape deep fields.
    Returns the product_partial dict updated with new detail keys.
    """
    # detail_url = product_partial.get("link")
    # if not detail_url:
    #     return product_partial

    try:
        await page.goto(detail_url, timeout=45000)
        await page.wait_for_timeout(20000)

        html = await page.content()  # ← here's your HTML snapshot

        with open("ec2_asin.html", "w", encoding="utf-8") as f:
            f.write(html)

        title = "N/A"
        title_el = await page.query_selector("#productTitle")
        if title_el:
            title = (await title_el.inner_text()).strip()

        price = "N/A"
        price_el = await page.query_selector("span.a-price-whole")
        if price_el:
            price = (await price_el.inner_text()).strip()

        original_price = "N/A"
        original_price_el = await page.query_selector("span.a-price.a-text-price span.a-offscreen")
        if original_price_el:
            original_price = (await original_price_el.inner_text()).strip()

        rating = "N/A"
        rating_el = await page.query_selector("span.a-icon-alt")
        if rating_el:
            rating = (await rating_el.inner_text()).strip().split(" out")[0]

        review_count = "N/A"
        review_count_el = await page.query_selector("#acrCustomerReviewText")
        if review_count_el:
            review_count = (await review_count_el.inner_text()).strip().split(" ")[0].replace(",", "")

        image_url = "N/A"
        img_el = await page.query_selector("#imgTagWrapperId img")
        if img_el:
            image_url = await img_el.get_attribute("src")

        brand = "N/A"
        brand_el = await page.query_selector("#bylineInfo")
        if brand_el:
                text = (await brand_el.inner_text()).strip()
                # Extract brand from "Visit the <Brand> Store"
                match = re.search(r"Visit the (.+?) Store", text)
                if match:
                    brand = match.group(1).strip()
        if brand == "N/A":
            rows = await page.query_selector_all("#technicalSpecifications_section_1 tr")
            for row in rows:
                th = await row.query_selector("th")
                td = await row.query_selector("td")
                if th and "Brand" in (await th.inner_text()):
                    brand = (await td.inner_text()).strip() if td else "N/A"
                    break

        limited_time_deal = "no"
        limited_time_deal_el = await page.query_selector("div#dealBadge_feature_div")
        if limited_time_deal_el:
            inner_text = (await limited_time_deal_el.inner_text()).strip()
            if inner_text:  # Check if there is visible text inside the div
                limited_time_deal = "yes"



        # Number of images
        image_thumbs = await page.query_selector_all("li.imageThumbnail img")
        num_images = len(image_thumbs)

        # Number of videos
        video_thumbs = await page.query_selector_all("li.videoThumbnail img")
        num_videos = len(video_thumbs)

        brand_store = "brand store not available"
        if await page.query_selector("div#titleBlockLeftSection"):
            brand_store = "brand store is available"
        

        

        # "From the Manufacturer" content
        a_plus_content = "A plus content not available"
        is_a_plus = False
        a_plus_headings = await page.query_selector_all("div#aplus h2")

        for heading in a_plus_headings:
            text = (await heading.inner_text()).strip().lower()
            if "from the brand" in text or "from the manufacturer" in text:
                is_a_plus = True
                break
        
        if is_a_plus:
            a_plus_content = "A plus content is available"

        # Bullet points
        bullet_points = []
        bullets_list = await page.query_selector_all("ul.a-unordered-list.a-vertical.a-spacing-small li span.a-list-item")
        for li in bullets_list:
            txt = (await li.inner_text()).strip()
            if txt:
                bullet_points.append(txt)

        # Product description
        desc_el = await page.query_selector("div#productDescription p")
        product_description = (await desc_el.inner_text()).strip() if desc_el else "N/A"

        if product_description == "":
            desc_el = await page.query_selector("div#productDescription")
            if desc_el:
                product_description = (await desc_el.inner_text()).strip()

        badge = "none"
        # Check for Amazon's Choice
        amazons_choice_el = await page.query_selector("div#acBadge_feature_div")
        if amazons_choice_el:
            choice_text = (await amazons_choice_el.inner_text()).strip()
            if "Amazon's" in choice_text and "Choice" in choice_text:
                badge = "Amazon's Choice"

        # Check for Best Seller
        best_seller_el = await page.query_selector("div#zeitgeistBadge_feature_div")
        if best_seller_el and badge == "none":
            best_seller_text = (await best_seller_el.inner_text()).strip()
            if "#1 Best Seller" in best_seller_text:
                badge = "#1 Best Seller"


        # Date First Available & Best Sellers Rank
        date_first_available = "N/A"
        asin = "N/A"
        best_seller_rank = "N/A"
        # Try techSpec table
        rows = await page.query_selector_all("#productDetails_techSpec_section_1 tr")
        if not rows:
            rows = await page.query_selector_all("#productDetails_detailBullets_sections1 tr")

        for row in rows:
            header = (await row.query_selector("th").inner_text()).strip()
            value = (await row.query_selector("td").inner_text()).strip()
            if "Date First Available" in header:
                date_first_available = value
            elif "Best Sellers Rank" in header:
                best_seller_rank = value

        detail_bullets = await page.query_selector_all("#detailBulletsWrapper_feature_div li")
        for li in detail_bullets:
            try:
                label_el = await li.query_selector("span.a-text-bold")
                value_el = await li.query_selector("span:not(.a-text-bold)")
                if label_el and value_el:
                    label = (await label_el.inner_text()).strip().replace(":", "")
                    value = (await value_el.inner_text()).strip()
                    value = value.split(":", 1)[-1].strip() 

                    # print(label)
                    # print()
                    # print()
                    # print(value)
                    if "Date First Available" in label:
                        date_first_available = value
                    elif "ASIN" in label:
                        asin = value
                    elif best_seller_rank in label:
                        best_seller_rank = value
                    # add more fields as needed
            except Exception as e:
                print(f"Error parsing li in detail bullets: {e}")
        
        try:
            detail_bullets = await page.query_selector_all("ul.detail-bullet-list li span.a-list-item")
            for item in detail_bullets:
                #print(item.inner_text())
                label_el = await item.query_selector("span.a-text-bold")
                if label_el:
                    label = (await label_el.inner_text()).strip().replace(":", "")
                    if "Best Sellers Rank" in label:
                # Remove the label text from the full inner text to get just the value
                        full_text = (await item.inner_text()).strip()
                        best_seller_ranks = full_text.split("\n")
                        best_seller_rank = " | ".join(rank.strip() for rank in best_seller_ranks if rank.strip())
                        break  # We got what we wanted, exit loop
        except Exception as e:
            print(f"Error extracting Best Sellers Rank: {e}")

        # Update and return
        product_partial.update({
            "title": title,
            "price": price,
            "original_price": original_price,
            "rating": rating,
            "review_count": review_count,
            "image_url": image_url,
            "brand": brand,
            "num_images": num_images,
            "num_videos": num_videos,
            "A_plus_content": a_plus_content,
            "bullet_points": bullet_points,
            "product_description": product_description,
            "date_first_available": date_first_available,
            "limited_time_deal": limited_time_deal,
            "asin": asin,
            "best_seller_rank": best_seller_rank,
            "brand_store": brand_store,
            "badge": badge,
        })

    except Exception as e:
        print(f"Error scraping detail for {detail_url}: {e}")

    return product_partial

async def scrape_all_product_details(asins: list) -> list:
    """
    Given a list of partial product dicts (from Phase 1), iterate through each one,
    scrape detail data, and return a new list with updated dicts.
    """
    updated_products = []
    async with async_playwright() as p:
        user_agents = [
            # Windows Chrome (latest-ish)
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
            "Mozilla/5.0 (Windows NT 10.0; WOW64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
            # Windows Firefox
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:137.0) Gecko/20100101 Firefox/137.0",
            # macOS Safari & Chrome
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_5) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/18.3 Safari/605.1.15",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 14_7_5) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
            # Linux Chrome
            "Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Safari/537.36",
            # Mobile Chrome (Android) & Safari (iPhone)
            "Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/135.0.0.0 Mobile Safari/537.36",
            "Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/605.1.15",
        ]
        viewport_sizes = [
            {"width": 1920, "height": 1080},  # Full HD desktop
            {"width": 2560, "height": 1440},  # 2K desktop
            {"width": 3840, "height": 2160},  # 4K desktop
            {"width": 1366, "height": 768},   # Entry-level laptop
            {"width": 1536, "height": 864},   # Mid-range laptop
            {"width": 1440, "height": 900},   # MacBook handles
            {"width": 1280, "height": 720},   # Lower-end widescreens
            {"width": 360, "height": 800},    # Android phone
            {"width": 390, "height": 844},    # Larger iPhones
            {"width": 768, "height": 1024},   # iPads/tablets portrait
            {"width": 800, "height": 1280},   # Android tablets portrait
        ]

        # selected_user_agent = random.choice(user_agents)
        # selected_viewport = random.choice(viewport_sizes)

        browser = await p.chromium.launch(headless=True)
        # context = await browser.new_context(
        #     user_agent=selected_user_agent,
        #     viewport=selected_viewport
        # )
        # page = await context.new_page()

        for idx, asin in enumerate(asins, start=1):
            print(f"Scraping details [{idx}/{len(asins)}] for ASIN: {asin}")

            selected_user_agent = random.choice(user_agents)
            selected_viewport = random.choice(viewport_sizes)

            context = await browser.new_context(
                user_agent=selected_user_agent,
                viewport=selected_viewport
            )
            page = await context.new_page()


            url = f"https://www.amazon.in/dp/{asin}"
            product_dict = {"asin": asin, "link": url}
            try:
                full_product = await scrape_amazon_product_detail(page, product_dict, url)
                full_product["asin"] = asin
                updated_products.append(full_product)
            except Exception as e:
                print(f"Failed to scrape {asin}: {e}")
            # finally:
            #     await con.close()
            # full_product = await scrape_amazon_product_detail(page, product_dict, url)
            # full_product["asin"] = asin
            # updated_products.append(full_product)
            await asyncio.sleep(random.uniform(40,60))  #reduce requests

        await browser.close()

    return updated_products

async def extract_asins_from_cardsclient(page):
    html = await page.content()

    m = re.search(r"<!--CardsClient-->(.*?)</div>", html, re.S)
    if not m:
        return []

    block = m.group(1)
    # find JSON in data-a-carousel-options
    j = re.search(r'data-a-carousel-options="([^"]+)"', block)
    if not j:
        return []

    raw = j.group(1).encode('utf-8').decode('unicode_escape')
    cfg = json.loads(raw)

    # Amazon often lists ASINs under cfg["ajax"]["id_list"]
    asins = cfg.get("ajax", {}).get("id_list", [])
    return list(asins)



async def get_new_asin_list(asin):
    new_asin_list = []
    """
    Given a single ASIN, scrape the "Related products" carousel on its Amazon.in product page,
    extract the ASINs of all the related /dp/ links, and return them as a list.
    """
    # defaults if none provided
    user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...",
        ]
    viewport_sizes = [
            {"width": 1920, "height": 1080},
            {"width": 1366, "height": 768},
        ]

    related_asins = set()
    url = f"https://www.amazon.in/dp/{asin}"

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)

        # choose UA and viewport at random
        ua = random.choice(user_agents)
        vp = random.choice(viewport_sizes)
        context = await browser.new_context(user_agent=ua, viewport=vp)
        page = await context.new_page()

        await page.goto(url,wait_until="networkidle",timeout=45000)
        await page.wait_for_timeout(10000)

        # ans = await extract_asins_from_cardsclient(page)
        # await context.close()
        # return ans

        html = await page.content()  # ← here's your HTML snapshot

        with open("ec2_page.html", "w", encoding="utf-8") as f:
            f.write(html)

        # wait for the related-products carousel container to appear
        # adjust selector if Amazon changes its DOM
        # carousel_sel = "#sp_detail_thematic-prime_theme_for_non_prime_members"
        carousel_selectors = [
            "#sp_detail",
            "#sp_detail_thematic-prime_theme_for_non_prime_members",
            "#sp_detail2",
            "sp_detail_thematic-prime_theme_for_prime_members"
        ]

        el = None
        for sel in carousel_selectors:
            try:
                el = await page.wait_for_selector(sel, timeout=10000)
                print(f"Found carousel via selector: {sel}")
                break
            except Exception:
                continue

        if not el:
            print("not loaded")
            await context.close()
            await browser.close()
            return []

        # try:
        #     # wait for the container to appear
        #     el = await page.wait_for_selector(carousel_sel, timeout=60000)
        # except Exception:
        #     print("not loaded")
        #     await context.close()
        #     await browser.close()
        #     return []

        # pull out the JSON in its data attribute
        raw = await el.get_attribute("data-a-carousel-options")
        if not raw:
            # if Amazon changed the attr-name, you could try other attrs here
            print("here")
            await context.close()
            await browser.close()
            return []

        try:
            cfg = json.loads(raw)
        except json.JSONDecodeError:
            print("here2")
            cfg = {}

        # collect any ASIN arrays we find
        for key in ("initialSeenAsins", "filteredItems"):
            vals = cfg.get(key)
            if isinstance(vals, list):
                related_asins.update(vals)

        await context.close()
        await browser.close()

    return list(related_asins)

async def get_offer_counts(asins):
    """
    Given a list of ASINs, scrape the number of offers for each one on Amazon.in
    and return a list of dicts with the ASIN and number of offers.

    Scrape the "New (X) from" span on each product page to get the number of offers.
    If this span is not found, default to 1 offer.

    Throttle requests to avoid getting rate-limited. Sleep for a random time between
    2 and 4 seconds between requests.

    :param asins: list of ASINs to scrape
    :return: list of dicts with ASIN and number of offers
    """
    results = []

    count = 1

    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        for asin in asins:
            print(f"Scraping offers [{count}/{len(asins)}]")
            url = f"https://www.amazon.in/dp/{asin}"
            try:
                await page.goto(url, timeout=45000)

                # Look for the "New (X) from" span
                offer_elements = await page.locator("span.a-color-base").all_text_contents()
                offers_found = False
                for text in offer_elements:
                    match = re.search(r'New \((\d+)\) from', text)
                    if match:
                        num_offers = int(match.group(1))
                        offers_found = True
                        break

                if not offers_found:
                    num_offers = 1  # default if offer count not found

                results.append({'asin': asin, 'offers': num_offers})

            except Exception as e:
                print(f"Error fetching ASIN {asin}: {e}")
                results.append({'asin': asin, 'offers': 1})  # default in error

            count += 1
            await asyncio.sleep(random.uniform(2, 4))  # Throttle requests

        await browser.close()

    return results


async def get_product_rank(target_asin, search_query,max_pages = 4):
    """
    Given an ASIN and a search query, scrape the product rank on Amazon.in
    and return the rank as an integer.
    """
    amazon_base = "https://www.amazon.in/s?k="
    search_query_encoded = urllib.parse.quote_plus(search_query)
    start_url = amazon_base + search_query_encoded

    async with async_playwright() as p:
        user_agents = [
            "Mozilla/5.0 (Windows NT 10.0; Win64; x64)...",
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)...",
        ]
        viewport_sizes = [
            {"width": 1920, "height": 1080},
            {"width": 1366, "height": 768},
        ]

        selected_user_agent = random.choice(user_agents)
        selected_viewport = random.choice(viewport_sizes)

        browser = await p.chromium.launch(headless=True , args=["--disable-gpu", "--no-sandbox", "--disable-dev-shm-usage"])
        context = await browser.new_context(
            user_agent=selected_user_agent,
            viewport=selected_viewport
        )
        browser = await p.chromium.launch(headless=True)
        page = await context.new_page()

        for page_num in range(1, max_pages + 1):
            url = start_url + f"&page={page_num}"
            print(f"Checking page {page_num}: {url}")
            await page.goto(url, timeout=60000)

            try:
                await page.wait_for_selector("div[data-asin]", timeout=20000)
            except Exception as e:
                print(f"Timeout waiting for product list: {e}")
                continue 
            asins = await page.query_selector_all("div[data-asin]")
            for index, product_el in enumerate(asins, start=1):
                asin = await product_el.get_attribute("data-asin")
                if asin and asin.strip() == target_asin:
                    await browser.close()
                    return {
                        "page": page_num,
                        "position_on_page": index
                    }

        await browser.close()
        return {
            "page": "not found",
            "position_on_page": "not found"
        }


def write_offers_to_excel(data, output_file_path):
    """
    Writes a list of dicts with 'asin' and 'offers' keys to an Excel file.
    
    Parameters:
        data (list of dict): Example - [{'asin': 'B08KNZM425', 'offers': 4}, ...]
        output_file_path (str): Path to save the Excel file.
    """
    df = pd.DataFrame(data)
    df.columns = ['ASIN', 'No. of Offers']  # Rename for clarity
    df.to_excel(output_file_path, index=False, engine='openpyxl')

# if __name__ == "__main__":
#     # asins = read_asins_from_excel()
#     # print(asins)
#     # print()
#     # # offers = get_offer_counts(asins)
#     # # write_offers_to_excel(offers, "result.xlsx")
#     asins = ["B0F8R4MSV5","B0F3XRY495"]
#     products = await scrape_all_product_details(asins)
#     print(products)
#     # # generate_excel_from_products(products, filename="amazon_products.xlsx")
#     # rank = get_product_rank("B0F8R4MSV5", "gold pearl flower earrings green enamel")
#     # print(rank)

if __name__ == "__main__":
    asins = "B0F8R4MSV5"
    products = asyncio.run(get_new_asin_list(asins))
    print(products)
    