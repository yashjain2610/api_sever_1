# img_to_csv.py - Complete Step-by-Step Explanation for Beginners

## What Does This Code Do?
This is a **server application** that takes jewelry product images, analyzes them using AI, and generates detailed product listings for online marketplaces like Amazon, Flipkart, and Meesho.

Think of it like: **Image Upload → AI Analysis → Generate Product Info → Save to Excel File**

---

## Section 1: IMPORTS & SETUP (Lines 1-100)
### What is happening?

```python
from fastapi import FastAPI
from google import genai
import json
```

**Importing Libraries** = Getting tools needed to build the application:
- **FastAPI**: Framework to create a web server that listens to requests
- **Google Genai**: AI model that analyzes images and generates text
- **AWS S3**: Cloud storage to save images
- **Milvus**: Database to store image fingerprints (for finding duplicates)
- **CLIP Model**: AI model to compare if two images are similar

### Key Setup:
```python
app = FastAPI()
```
Creates the web server that will listen for requests from users.

```python
app.add_middleware(CORSMiddleware, allow_origins=["*"])
```
Allows requests from any website (like your React frontend).

```python
API_KEY = os.getenv("GOOGLE_API_KEY")
client = genai.Client(api_key=API_KEY)
```
Gets the Google API key from environment variables and creates connection to Google Gemini AI.

### The `mapp` Dictionary (Lines 90-160):
```python
mapp={
    "Elegence": "Jewelry that remains timeless...",
    "Simplicity": "Pieces designed with less complexity...",
    ...
}
```
This is a **lookup table** that contains descriptions for 50+ jewelry styles. When user picks "Elegant", we look up the full description from this dictionary.

---

## Section 2: DATA MODELS (Lines 165-195)
### What are these?

Data models define the **shape of information** that users send to the server:

```python
class UserNeeds(BaseModel):
    collection_size: int        # How many products user wants
    color: str                  # Preferred color
    category: str               # Type (earrings, necklace, bracelet)
    category_description: List[str]  # Style (elegance, minimalist, etc)
    target_audience: str        # Who will buy it
    manual_prompt: str          # Extra notes from user
    start_price: int            # Budget minimum
    end_price: int              # Budget maximum
    start_date: date            # Date range
    end_date: date              # Date range
    extra_jwellery: List[str]   # Additional products to include
```

Think of it as a **form template**. When user fills the form, the data comes in this exact shape.

---

## Section 3: ENDPOINT 1 - /semantic_filter_jewelry/ (Lines 210-290)

### What Does It Do?
Finds jewelry products that match user preferences using AI similarity search.

### Step-by-Step Process:

**STEP 1: Get user preferences**
```python
payload={
    "start_date": user_date,
    "end_date": user_date,
    "start_price": user_price,
    "end_price": user_price,
    "category": user_category
}
response = requests.request("POST", "https://staging.blyskjewels.com/api/get-all-products", data=payload)
```
- Sends request to external API to fetch ALL jewelry products that match filters
- Gets list of 100+ jewelry items from database

**STEP 2: Convert each product to AI-readable format**
```python
for item in json.loads(response)["data"]:
    combined_text = f"Description: {item['description']}. Attributes: {item['attributes']}. Color: {item['color']}."
```
- Takes each product (name, description, color)
- Combines them into one text string
- This text will be converted to "embeddings" (AI fingerprint)

**STEP 3: Create AI fingerprints (embeddings)**
```python
result = client.models.embed_content(
    model="text-embedding-004",
    contents=combined_text
)
embedding = result.embedding_values[0]
```
- Sends text to Google Gemini
- Gemini returns a list of 768 numbers (embedding)
- These numbers represent the "meaning" of the product
- Similar products have similar numbers

**STEP 4: Store in ChromaDB**
```python
collection.add(
    documents=[combined_text],
    embeddings=[embedding],
    metadatas=[{
        "jwellery_id": item['id'],
        "jwellery_name_generated": item['product_name'],
    }],
    ids=[str(item['id'])]
)
```
- Saves the embedding to database with product info attached

**STEP 5: Convert user preferences to embedding**
```python
user_query = f"Category: {mapp[category]}. Target Audience: {audience}. Color: {color}."
query_embedding = client.models.embed_content(model="text-embedding-004", contents=user_query)
```
- Same process: converts user preferences to AI fingerprint

**STEP 6: Find matching products**
```python
results = collection.query(
    query_embeddings=[query_embedding],
    n_results=collection_size  # e.g., 5 products
)
```
- Database searches for products with similar embeddings
- Returns TOP 5 matching products

**STEP 7: Generate AI description of collection**
```python
response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=["Create eye catching description for collection..."]
)
```
- Uses Gemini to write a marketing description for the collection

**STEP 8: Generate collection title**
```python
coll_title = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=["Create an eye-catching collection title..."]
)
```
- Uses Gemini to create a catchy title

**STEP 9: Return results**
```python
return {
    "top_jewelry_ids": [id1, id2, id3, ...],
    "collection_description": "Beautiful elegant jewelry collection...",
    "collection_title": "Timeless Elegance Collection"
}
```
- Sends back the product IDs, description, and title to user

---

## Section 4: HELPER FUNCTION - manage_undo_file (Lines 293-310)

### What Does It Do?
Creates an "undo history" so users can go back to previous states.

```python
def manage_undo_file(new_data, file_path='undo.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
    else:
        data = []
    
    data.append(new_data)           # Add new action to history
    
    if len(data) > 10:              # Only keep last 10 actions
        data.pop(0)                 # Remove oldest action
    
    with open(file_path, 'w') as file:
        json.dump(data, file)       # Save updated history
```

**How it works:**
1. Checks if `undo.json` file exists
2. If yes, loads existing history; if no, creates empty list
3. Adds new action to list
4. Keeps only last 10 actions (removes oldest if more than 10)
5. Saves updated history back to file

**Example:**
```
undo.json = [action1, action2, action3, action4, action5]
User does new action
undo.json = [action1, action2, action3, action4, action5, action6]
User does another action
undo.json = [action1, action2, action3, action4, action5, action6, action7]
```

---

## Section 5: ENDPOINT 2 - /process-image-and-prompt/ (Lines 385-520)

### What Does It Do?
Converts a photo of a bill/invoice into structured data (table format).

### Use Case:
You take a photo of a receipt → Server extracts product list → Returns JSON

### Step-by-Step:

**STEP 1: Receive image**
```python
async def process_image_and_prompt(image: UploadFile = File(...)):
```
- User sends image file

**STEP 2: Save image temporarily**
```python
image_content = await image.read()
with tempfile.NamedTemporaryFile(delete=False, suffix=image_suffix) as tmp_file:
    tmp_file.write(image_content)
    save_path = tmp_file.name
```
- Reads image data
- Saves to temporary file
- Gets file path for later use

**STEP 3: Create prompt for AI**
```python
prompt = """
You are given an image of a bill or invoice. Extract all product entries...
Return a single valid JSON array...
Example: [
  {"QTY": "2", "DESCRIPTION": "Chair", "UNIT_PRICE": "$250", "TOTAL": "$500"},
  {"Invoice_Number": "INV-001", ...}
]
"""
```
- Creates detailed instructions for Gemini on what to extract
- Tells it to return JSON format

**STEP 4: Send image + prompt to Gemini**
```python
image_data = input_image_setup(io.BytesIO(image_bytes), "image/jpeg")
response_list = get_gemini_responses("Analyze this", image_data, [prompt])
gemini_response = response_list[0]
```
- Converts image to format Gemini accepts
- Sends image + instructions to Gemini AI
- Gets response back

**STEP 5: Clean response**
```python
if gemini_response[0]=="`":
    gemini_response=gemini_response[7:-3]  # Remove markdown code blocks
```
- Gemini sometimes wraps response in ```json ... ```
- This removes those extra characters

**STEP 6: Convert to Python dictionary**
```python
try:
    my_dict = json.loads(gemini_response)
except json.JSONDecodeError:
    raise HTTPException(500, "Failed to parse Gemini response")
```
- Converts JSON string to Python object
- If fails, returns error

**STEP 7: Extract product table and extras**
```python
extras = my_dict[-1]  # Last item is invoice details
table = my_dict[:-1]  # All other items are products
```
- Last item in array = invoice details (invoice number, total, etc)
- Everything else = product rows

**STEP 8: Save undo history**
```python
manage_undo_file({"table": my_dict[:-1], "extras": extras})
```
- Saves this action to undo.json

**STEP 9: Clean up and return**
```python
os.remove(save_path)  # Delete temporary file
return {
    "table": my_dict[:-1],
    "extras": extras
}
```
- Deletes temporary image file
- Returns extracted data to user

**Example Return:**
```json
{
  "table": [
    {"QTY": "2", "DESCRIPTION": "Earrings", "PRICE": "$50"},
    {"QTY": "1", "DESCRIPTION": "Necklace", "PRICE": "$100"}
  ],
  "extras": {
    "Invoice_Number": "INV-2025-001",
    "Total": "$200"
  }
}
```

---

## Section 6: ENDPOINT 3 - /generate_caption (Lines 525-750)

### What Does It Do?
Uploads a jewelry image and generates AI-powered product description.

### Step-by-Step:

**STEP 1: Check for duplicates using CLIP model**
```python
image_bytes = await file.read()
image_s = Image.open(BytesIO(image_bytes)).convert("RGB")
query_emb = embed_image(image_s, clipmodel, processor, device)
results = search_similar(collection_db, query_emb, 1)
```
- Reads uploaded image
- Converts to RGB format
- CLIP model creates an embedding (fingerprint)
- Searches Milvus database to find similar images
- If similarity > 95%, it's a duplicate

**STEP 2: If duplicate found, return early**
```python
if 100 - dist >= 95:
    duplicate = True
    
if duplicate:
    return {
        "display_name": image_name,
        "s3_url": matches[0]["path"],
        "duplicate": "duplicate found"
    }
```
- If same image already exists, don't process again
- Return link to existing image

**STEP 3: If not duplicate, upload to S3**
```python
if not duplicate:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
    unique_id = uuid.uuid4().hex[:8]
    unique_name = f"{timestamp}_{unique_id}_{image_name}"
    
    s3.upload_file(
        Filename=save_path,
        Bucket=S3_BUCKET,
        Key=unique_name
    )
    s3_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{unique_name}"
```
- Creates unique filename (timestamp + random ID + original name)
- Uploads to Amazon S3 cloud storage
- Creates public URL link

**STEP 4: Index image for future searches**
```python
index_single_image_from_s3(collection_db, unique_name, clipmodel, processor, device)
```
- Adds this image's embedding to Milvus database
- So future uploads can find it if similar

**STEP 5: Create detailed prompt for Gemini**
```python
pp = f"""
please generate a short description for the given jewelry: {type}.
STRICTLY MAKE SURE THE JEWELRY IS {type}.
Focus only on the jewelry features and not on the background.

The description should be one or more of these types:-
"Elegance": "Jewelry that embodies sophistication...",
"Charm": "Pieces that convey a sense of warmth...",
... (10 more style options)

Also generate jewelry name.
Also detect the letter A/B/C written on the image.
Also detect the color of the jewelry.

Final Output:-
JSON with keys -> quality: (A/B/C), name: unique jewelry name, 
description: generated description, attributes: jewelry physical features, 
color: color of the jewelry, final_caption: one-line caption
"""
```
- Detailed instructions for Gemini
- Asks for: name, description, attributes, color, quality grade, caption

**STEP 6: Send image to Gemini**
```python
image_data = input_image_setup(io.BytesIO(image_bytes), "image/jpeg")
response_list = get_gemini_responses("Analyze this image carefully.", image_data, [pp])
response0 = response_list[0]
```
- Converts image to Gemini format
- Sends image + prompt
- Gets response

**STEP 7: Parse Gemini response**
```python
if response0[0] == "`":
    response0 = response0[7:-4]  # Remove markdown
js = json.loads(response0)      # Convert to dictionary
```
- Removes markdown code blocks
- Converts JSON string to Python object

**STEP 8: Final response**
```python
return {
    "display_name": image_name,
    "generated_name": js["name"],
    "quality": js["quality"],
    "description": js["final_caption"],
    "attributes": js["attributes"],
    "prompt": js["prompt"],
    "color": js["color"],
    "s3_url": s3_url,
}
```
- Returns all extracted information to user

**Example Response:**
```json
{
  "display_name": "earring1.jpg",
  "generated_name": "Elegant Diamond Drop Earrings",
  "quality": "A",
  "description": "Premium diamond drop earrings with elegant craftsmanship",
  "attributes": "diamonds, gold plating, drop style",
  "color": "gold",
  "s3_url": "https://alyaimg.s3.amazonaws.com/20250123120000_abc123_earring1.jpg"
}
```

---

## Section 7: ENDPOINT 4 - /delete_image (Lines 775-825)

### What Does It Do?
Removes an image from both S3 cloud and Milvus database.

### Step-by-Step:

**STEP 1: Extract S3 key from URL**
```python
def extract_s3_key(s3_url: str) -> str:
    return s3_url.split(".amazonaws.com/")[-1]

# Example:
# Input:  "https://bucket.s3.amazonaws.com/folder/image.jpg"
# Output: "folder/image.jpg"
```

**STEP 2: Handle multiple URLs**
```python
s3_urls = [(url.strip()) for url in request.s3_url.split(",")]

for s3_url in s3_urls:
    s3_key = extract_s3_key(s3_url)
```
- User can delete multiple images at once (comma-separated)
- Process each one

**STEP 3: Find Milvus ID**
```python
with open(INT_HASH_MAP_FILE, "r") as f:
    id_to_path = json.load(f)

milvus_id = None
for _id, path in id_to_path.items():
    if path == s3_url or path.endswith(s3_key):
        milvus_id = int(_id)
        break
```
- Loads mapping file (ID ↔ URL)
- Finds the Milvus ID for this image

**STEP 4: Delete from S3**
```python
s3.delete_object(Bucket=S3_BUCKET, Key=s3_key)
```
- Removes image from cloud storage

**STEP 5: Delete from Milvus**
```python
collection_db.delete(expr=f"id in [{milvus_id}]")
collection_db.flush()
```
- Removes embedding from vector database
- Flushes (saves) changes

**STEP 6: Update mapping file**
```python
del id_to_path[str(milvus_id)]
with open(INT_HASH_MAP_FILE, "w") as f:
    json.dump(id_to_path, f)
```
- Removes the ID from mapping file
- Saves updated mapping

---

## Section 8: ENDPOINT 5 - /catalog-ai (Lines 980-1280)

### What Does It Do?
**Generates complete product listings for multiple marketplaces (Amazon, Flipkart, Meesho).**

This is the **MAIN FEATURE** of the entire server.

### Use Case:
```
Input:
- 5 image URLs
- 5 SKUs
- Marketplace: "Amazon"
- Type: "Earrings"

Output:
- JSON with product details for each image
- Excel file with all products formatted for Amazon
```

### Step-by-Step:

**STEP 1: Setup format**
```python
format = req.marketplace.lower()[:3] + "_" + req.type.lower()[:3]
# Examples: "ama_ear" (Amazon Earrings), "fli_nec" (Flipkart Necklaces)
```

**STEP 2: Get prompts and templates for this format**
```python
input_prompt_map = {
    "fli_ear": ([prompt_questions_earrings_flipkart], fixed_values_earrings, gpt_flipkart_earrings_prompt),
    "ama_ear": ([prompt_questions_earrings_amz], fixed_values_earrings_amz, gpt_amz_earrings_prompt),
    "mee_ear": ([prompt_questions_earrings_meesho], fixed_values_earrings_meesho, gpt_meesho_earrings_prompt),
    ...
}

input_prompts, fixed_values, gpt_prompt = input_prompt_map.get(format)
```
- Each marketplace needs different format (Amazon requires different fields than Flipkart)
- Gets the right prompts and templates for this combination

**STEP 3: Parse input URLs and SKUs**
```python
url_list = [(url.strip()) for url in req.image_urls.split(",")]
skuid_list = [(sku.strip()) for sku in req.skuids.split(",")]

# Example:
# image_urls: "https://...img1.jpg, https://...img2.jpg, https://...img3.jpg"
# skuids: "SKU-001, SKU-002, SKU-003"
```

**STEP 4: Download Excel template from S3**
```python
s3_url = f"https://alyaimg.s3.amazonaws.com/excel_files/{static_file_name}"
response = httpx.get(s3_url)

with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
    tmp_file.write(response.content)
    tmp_path = tmp_file.name
```
- Downloads existing Excel template from cloud
- Saves to temporary file locally

**STEP 5: Process each image**
```python
for url, skuid in zip(url_list, skuid_list):
    r = await client.get(url)
    image_bytes = r.content
```
- Fetches each image from URL
- Gets image data as bytes

**STEP 6: Analyze image with Gemini**
```python
image_data = input_image_setup(io.BytesIO(image_bytes), "image/jpeg")
response_list = get_gemini_responses("Analyze this image carefully.", image_data, input_prompts)
```
- Sends image to Gemini
- Asks questions about jewelry (material, design, size, etc.)
- Gets structured response

**STEP 7: Parse Gemini response**
```python
cleaned = response_list[0].strip().replace("```json", "").replace("```", "").strip()
response_json = json.loads(cleaned)
```
- Removes markdown
- Converts to Python object

**STEP 8: Check if product already exists in database**
```python
available_dict = fetch_data_for_sku(skuid)

if available_dict.get("sku_exists") == False:
    # Product is NEW, use Gemini to generate description
    gpt_dict = ask_gpt_with_image(prompt=gpt_prompt, image_bytes=image_bytes)
else:
    # Product exists, fetch existing description
    gpt_dict["description"] = available_dict.get("description")
```
- Checks if this SKU already has product data
- If NEW: uses GPT to generate everything
- If EXISTS: uses existing data from database

**STEP 9: Merge all data**
```python
final_response = {
    **dict,              # filename, skuid
    **response_json,     # Gemini analysis
    **gpt_dict,          # GPT descriptions
    **fixed_values       # Required Amazon/Flipkart fields
}
```
- Combines Gemini analysis + GPT descriptions + marketplace requirements

**STEP 10: Expand bullet points**
```python
gpt_dict = expand_bullet_points(gpt_dict)
```
- Makes descriptions more detailed and formatted

**STEP 11: Save to local database**
```python
store_data_for_sku(skuid, gpt_dict)
```
- Saves this product data locally for future use

**STEP 12: Write to Excel**
```python
if format == "fli_ear":
    write_to_excel_flipkart(excel_results, filename=tmp_path, target_fields=..., fixed_values=...)
elif format == "ama_ear":
    write_to_excel_amz_xl(excel_results, filename=tmp_path, target_fields=..., fixed_values=...)
```
- Writes product to Excel file in marketplace-specific format

**STEP 13: Upload Excel back to S3**
```python
s3.upload_file(
    Filename=tmp_path,
    Bucket=S3_BUCKET,
    Key=s3_key,
    ExtraArgs={'ContentType': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'}
)
```
- Updates Excel file in cloud

**STEP 14: Return results**
```python
return {
    "results": [
        {
            "attributes": {
                "filename": url,
                "skuid": "SKU-001",
                "product_name": "Diamond Earrings",
                "description": "...",
                "price": "$49.99",
                ... (all fields for marketplace)
            }
        },
        ... (more products)
    ],
    "excel_file": "https://alyaimg.s3.amazonaws.com/excel_files/earrings_amazon.xlsx"
}
```

---

## Section 9: ENDPOINT 6 - /catalog_ai_variations (Lines 1330-1450)

### What Does It Do?
**Same as catalog-ai BUT handles product variations (different sizes, colors).**

### Difference:
```
Regular catalog: 5 images = 5 products
Variations: 5 images = 1 parent product + 5 variations (size/color)
```

### Key Process:
1. **Groups images** based on variation type (size, color, size+color)
2. **Creates parent product** with variation details
3. **Creates child products** for each variation
4. **Writes to Excel** with parent-child relationship

**Example:**
```
User wants: 1 Earring in 3 sizes and 2 colors = 6 total variations

Parent Product: "Diamond Earrings"
├─ Variation: Size=Small, Color=Gold
├─ Variation: Size=Small, Color=Silver
├─ Variation: Size=Medium, Color=Gold
├─ Variation: Size=Medium, Color=Silver
├─ Variation: Size=Large, Color=Gold
└─ Variation: Size=Large, Color=Silver

In Excel: One row with parent SKU, then 6 rows with child SKUs linked to parent
```

---

## Section 10: KEY HELPER FUNCTIONS

### A. `input_image_setup()`
```python
image_data = input_image_setup(io.BytesIO(image_bytes), "image/jpeg")
```
Converts image bytes to format Gemini accepts.

### B. `get_gemini_responses()`
```python
response = get_gemini_responses("Analyze this", image_data, [prompt])
```
Sends image + prompt to Gemini, gets response.

### C. `embed_image()`
```python
embedding = embed_image(image, clipmodel, processor, device)
```
Converts image to CLIP embedding (fingerprint for duplicate detection).

### D. `search_similar()`
```python
results = search_similar(collection_db, query_emb, top_k=1)
```
Finds similar images in Milvus database.

### E. `ask_gpt_with_image()`
```python
gpt_dict = ask_gpt_with_image(prompt=prompt, image_bytes=image_bytes)
```
Uses GPT-4 to analyze image and generate descriptions.

### F. `write_to_excel_*()`
```python
write_to_excel_amz_xl(excel_results, filename=path, target_fields=..., fixed_values=...)
```
Writes product data to Excel in marketplace-specific format.

---

## Common Issues & How to Fix Them

### Issue 1: "API Key not found"
```
Error: "Warning: GOOGLE_API_KEY not found"
```
**Fix:** Add to `.env` file:
```
GOOGLE_API_KEY=your_actual_api_key_here
S3_BUCKET=your_bucket_name
MILVUS_HOST=localhost
```

### Issue 2: "Failed to upload to S3"
```
Error: "Failed to upload to S3"
```
**Fix:** Check AWS credentials in environment:
```
AWS_ACCESS_KEY_ID=...
AWS_SECRET_ACCESS_KEY=...
```

### Issue 3: "Milvus connection failed"
```
Error: "Failed to initialize Milvus connection"
```
**Fix:** Ensure Milvus server is running:
```powershell
# Windows/PowerShell
docker run -d -p 19530:19530 milvusdb/milvus:latest
```

### Issue 4: "No matching jewelry found"
```
Error: HTTPException(status_code=404, detail="No matching jewelry found")
```
**Fix:** External API might be down. Check:
- Is https://staging.blyskjewels.com/api/get-all-products accessible?
- Are there products in the date/price range?

### Issue 5: Excel file format errors
```
Error: "Failed to parse dimensions" or missing fields
```
**Fix:** Check if prompted marketplace format matches in `prompts.py`:
- `target_fields_earrings_amz`
- `target_fields_earrings_flipkart`
- etc.

---

## DATA FLOW DIAGRAM

```
┌─────────────────┐
│  User Sends     │
│  - Images       │
│  - Preferences  │
└────────┬────────┘
         │
         ▼
┌─────────────────┐
│  Server         │
│  (FastAPI)      │
└────────┬────────┘
         │
    ┌────┴────┐
    │          │
    ▼          ▼
┌────────┐  ┌──────────┐
│Gemini  │  │AWS S3    │
│AI      │  │(Upload)  │
└────┬───┘  └──────┬───┘
     │             │
     └──────┬──────┘
            │
            ▼
      ┌──────────┐
      │Milvus DB │
      │(Index)   │
      └────┬─────┘
           │
           ▼
    ┌────────────┐
    │Write Excel │
    │+ Upload    │
    └────┬───────┘
         │
         ▼
    ┌──────────┐
    │Return    │
    │Response  │
    │to User   │
    └──────────┘
```

---

## Next Steps for Improvement

1. **Add error logging**: Log all API calls and errors to file
2. **Add rate limiting**: Prevent too many requests to Gemini
3. **Add authentication**: Secure endpoints with API keys
4. **Improve error messages**: Make them more user-friendly
5. **Add input validation**: Check image file size, format before processing
6. **Cache results**: Store Gemini responses to avoid duplicate API calls
7. **Add progress tracking**: Let users know when processing 5 images (which one is done)
8. **Database cleanup**: Remove old undo.json entries automatically

---

## Summary Table

| Endpoint | Input | Output | Purpose |
|----------|-------|--------|---------|
| `/semantic_filter_jewelry/` | User preferences | Matching product IDs | Find similar products |
| `/process-image-and-prompt/` | Bill image | Product table | Extract from receipts |
| `/generate_caption` | Jewelry image | Product metadata | Generate descriptions |
| `/delete_image` | Image URL | Success/Error | Remove duplicates |
| `/catalog-ai` | Images + SKUs | Excel file | Create marketplace catalog |
| `/catalog_ai_variations` | Images + size/color | Excel with variations | Handle multiple variants |
| `/regenerate` | Previous prompt + style | New description | Change description style |
| `/regenerate_title` | Attributes | New title | Generate new name |

---

**You're now ready to debug and improve this codebase!** 🎉
