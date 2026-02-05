# Complete Project Guide - AI Jewelry E-Commerce Assistant

## 🎯 Project Overview

This is a **full-stack AI application** that automates jewelry product catalog creation for e-commerce platforms. It combines **image analysis, AI text generation, and web scraping** to create marketplace-ready product listings.

### What Does It Do?
1. **Takes jewelry images** → Analyzes them with AI
2. **Extracts product details** → Material, color, design, quality
3. **Generates descriptions** → Creates marketing copy
4. **Creates Excel catalogs** → Formats for Amazon/Flipkart/Meesho
5. **Detects duplicates** → Finds similar images to prevent uploads
6. **Generates variants** → Creates size/color options automatically

---

## 📁 Project Structure

```
api_sever_1-main/
├── 🔵 CORE APPLICATION (Main Server)
│   ├── img_to_csv.py              ⭐ Main FastAPI server with all endpoints
│   ├── requirements.txt            📦 All dependencies needed
│   └── .env                        🔑 Configuration/API keys
│
├── 🟢 FRONTEND UI (User Interface)
│   ├── jwellery_front.py          🎨 Main Streamlit app for jewelry
│   ├── frontend.py                🎨 Simple frontend for bill processing
│   └── jwellery_front.py          🎨 Alternative UI
│
├── 🟡 CONFIGURATION & PROMPTS (AI Instructions)
│   ├── prompts.py                 📝 1737 lines - Instructions for Gemini (Flipkart, Amazon, Meesho)
│   ├── prompts2.py                📝 Image generation prompts (DALL-E)
│   ├── excel_fields.py            📋 Excel field mappings for each marketplace
│   └── project_overview.md        📖 Project documentation
│
├── 🔴 AI & ML MODULES (Smart Features)
│   ├── image_search_engine.py     🔍 CLIP model + Milvus database (find duplicates)
│   ├── openai_utils.py            🤖 GPT-4 Vision API calls
│   ├── utils.py                   ⚙️ Gemini API utilities
│   └── utils2.py                  ⚙️ Image generation, error handling
│
├── 🟠 WEB SCRAPERS (Data Collection)
│   ├── scraper.py                 🕷️ Amazon product scraper
│   ├── flp_scraper.py             🕷️ Flipkart product scraper
│   └── myn_scraper.py             🕷️ Meesho product scraper
│
├── 📦 DATABASE & STORAGE
│   ├── json_storage.py            💾 Local JSON database operations
│   ├── data_store.json            📊 Stored product data
│   ├── indexed_hashes.json        🔑 Image fingerprints
│   └── int_hash_to_path.json      🗺️ ID to image URL mapping
│
├── 📄 DATA FILES
│   ├── bill_extra.csv             📋 Extracted invoice details
│   ├── gemini_output_bill.csv     📋 AI-processed bill data
│   ├── jwelellry.csv              📊 Jewelry products
│   ├── *.jpg                      📸 Sample images
│   └── undo.json                  ⏪ Undo history
│
├── 📁 DIRECTORIES
│   ├── static/                    🖼️ Static files (CSS, JS, images)
│   ├── excel_files/               📊 Generated Excel templates
│   ├── gen_images/                🎨 Generated product images
│   ├── generated_variants_earrings/  🎨 Generated earring variants
│   └── __pycache__/              🔧 Python cache files
│
└── 🛠️ UTILITIES & GENERATION
    ├── generate_variants.py       🎨 Create image variations
    └── project_flow_diagrams.md  📊 Flow diagrams
```

---

## 🏗️ Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    USER (Browser)                        │
│          Streamlit Frontend (jwellery_front.py)         │
└────────────────┬────────────────────────────────────────┘
                 │ HTTP Requests (Images, SKUs)
                 ▼
┌─────────────────────────────────────────────────────────┐
│                  FastAPI Server                          │
│              (img_to_csv.py - Line 1-2000+)            │
│                                                         │
│  • /generate_caption - Image analysis                  │
│  • /catalog-ai - Create full catalog                   │
│  • /catalog_ai_variations - Size/color variants        │
│  • /semantic_filter_jewelry - Find similar products    │
│  • /image_similarity_search - Duplicate detection      │
└────────────────┬────────────────────────────────────────┘
                 │
        ┌────────┼────────┬──────────┐
        │        │        │          │
        ▼        ▼        ▼          ▼
    ┌────┐  ┌────┐  ┌──────┐  ┌──────────┐
    │Gem │  │AWS │  │Milvus│  │ChromaDB  │
    │ini │  │S3  │  │Vector│  │Semantic  │
    │AI  │  │Cloud│ │DB   │  │Search    │
    └────┘  └────┘  └──────┘  └──────────┘
```

---

## 📄 File-by-File Explanation

### 1. **img_to_csv.py** (Main Server - 2000 Lines)

**What is it?**
The **brain of the entire application**. FastAPI server that handles all API requests.

**Key Endpoints:**

| Endpoint | Input | Output | Purpose |
|----------|-------|--------|---------|
| `/generate_caption` | Jewelry image + style | Product metadata (name, description, attributes) | Analyzes image, generates AI descriptions |
| `/catalog-ai` | Multiple images + SKUs + marketplace | JSON + Excel file | Creates complete product catalog |
| `/catalog_ai_variations` | Images + variation type (size/color) | Excel with parent-child products | Handles product size/color options |
| `/semantic_filter_jewelry/` | User preferences (color, price, style) | Matching product IDs | Finds similar products in database |
| `/image_similarity_search` | Query image | Similar images with scores | Detects duplicate images |
| `/process-image-and-prompt/` | Bill/invoice image | Extracted product table | Converts receipt to structured data |
| `/delete_image` | Image URLs | Success/error | Removes image from S3 and database |
| `/regenerate` | Previous description + new style | New description | Changes product description style |
| `/regenerate_title` | Previous title + attributes | New title | Generates alternative product name |

**How it works:**
1. Receives request (image upload, parameters)
2. Calls Google Gemini AI to analyze image
3. Calls GPT-4 to generate descriptions
4. Uploads image to AWS S3
5. Stores image fingerprint in Milvus database
6. Writes to Excel file
7. Returns results to user

---

### 2. **prompts.py** (1737 Lines - AI Instructions)

**What is it?**
A **massive file containing exact instructions** for Gemini AI on how to analyze jewelry images.

**Structure:**
```python
prompt_questions_earrings_flipkart = """
you are jewellery expert and an e-commerce catalog manager for flipkart,
you will be given a image of a jewellery item.
answer the given questions...

Questions {
    {field_name: Type, question: What is the type of earring?, options: [Stud, Hoop, Dangle, ...]}
    {field_name: Color, question: What color?, options: [Gold, Silver, ...]}
    {field_name: Base Material, question: What material?, options: [Alloy, Gold, Silver, ...]}
    ... (50+ more fields)
}
"""
```

**Why multiple marketplace versions?**
- **flipkart version**: "Type", "Earring Back Type", "Trend"
- **amazon version**: Different fields like "Material", "Occasion"
- **meesho version**: Lightweight version with fewer fields

**Example Usage:**
```python
# In img_to_csv.py line ~1055
input_prompts, fixed_values, gpt_prompt = input_prompt_map.get("ama_ear")
# Gets Amazon Earrings prompts
response = get_gemini_responses("Analyze this image", image_data, input_prompts)
```

---

### 3. **prompts2.py** (Image Generation Prompts)

**What is it?**
Instructions for **DALL-E 3** to generate product image variations.

**Examples:**
```python
white_bgd_prompt = """Generate a professional product photo...
- Clean white background
- Studio lighting
- Jewelry centered in frame
- High resolution, premium quality"""

multicolor_1_prompt = """Generate same jewelry...
- Colorful background
- Multiple colors
- Vibrant lighting"""
```

**Used by:**
- `utils2.py` → `generate_images_from_gpt()` function
- `generate_variants.py` → Creates multiple versions of same jewelry

---

### 4. **excel_fields.py** (521 Lines - Marketplace Mappings)

**What is it?**
Defines **Excel column names and fixed values** for each marketplace.

**Structure:**
```python
target_fields_earrings = [
    "Seller SKU ID", "Model Number", "Type", "Color", 
    "Base Material", "Gemstone", "Pearl Type", "Collection",
    "Occasion", "Piercing Required", ...
]

fixed_values_earrings = {
    "Seller SKU ID": "ALYA-EAR",
    "Brand": "Alya",
    "Seller Name": "Our Store"
}

# Similar for Amazon, Meesho, etc...
target_fields_earrings_amz = [...]
fixed_values_earrings_amz = {...}
```

**Why needed?**
Each marketplace has different required fields:
- Flipkart needs: Type, Earring Back Type, Trend
- Amazon needs: Material, Occasion, Number of Gemstones
- Meesho needs: Simplified version

---

### 5. **image_search_engine.py** (530 Lines)

**What is it?**
**Duplicate detection system** using AI image fingerprints.

**How it works:**

```python
# STEP 1: Load CLIP model (converts image to vector)
clipmodel, processor, device = init_clip()

# STEP 2: Create Milvus database connection
collection_db = init_milvus(host="localhost", port="19530", 
                             collection_name="image_embeddings")

# STEP 3: Process new image
image = Image.open("earring.jpg")
embedding = embed_image(image, clipmodel, processor, device)  # 768 numbers

# STEP 4: Search for similar
results = search_similar(collection_db, embedding, top_k=1)  # Find top 1 match
# Returns: [(id=123, distance=2.5), ...]  # distance < 5 = duplicate

# STEP 5: If duplicate found, return existing image URL
if 100 - distance >= 95:  # 95% match = duplicate
    return {"duplicate": "found", "s3_url": existing_url}
else:
    # New image, upload to S3 and Milvus
    s3.upload_file(...)
    index_single_image_from_s3(...)
```

**Key Functions:**
- `init_clip()` - Loads CLIP model to RAM
- `init_milvus()` - Connects to vector database
- `embed_image()` - Converts image to 768-dimensional vector
- `search_similar()` - Finds similar images in database
- `index_single_image_from_s3()` - Adds new image to index

**Data Storage:**
- `indexed_hashes.json` - Stores image hashes
- `int_hash_to_path.json` - Maps ID → S3 URL

---

### 6. **utils.py** (299 Lines - Gemini Utilities)

**What is it?**
Helper functions for **Google Gemini API calls**.

**Key Functions:**

```python
def input_image_setup(file_obj, mime_type):
    """Converts image bytes to Gemini-compatible format"""
    bytes_data = file_obj.getvalue()
    return [types.Part.from_bytes(data=bytes_data, mime_type=mime_type)]

def get_gemini_responses(input_text, image_data, prompts):
    """Sends image + prompts to Gemini, gets responses"""
    for prompt in prompts:
        contents = [input_text, image_data[0], prompt]
        response = client.models.generate_content(
            model='gemini-2.0-flash',
            contents=contents,
            config=config
        )
        all_responses.append(response.text)
    return all_responses

def get_gemini_responses_high_temp(input, image, prompts):
    """Same as above but with higher temperature (more creative)"""
    # temperature=1.5 means more random/creative responses
```

**Temperature Explanation:**
- temperature=1.0 (default) = Consistent, reliable
- temperature=1.5 (high_temp) = More creative, varied
- Used for generating different marketing descriptions

---

### 7. **utils2.py** (243 Lines - Image Generation & Error Handling)

**What is it?**
Advanced utilities for **DALL-E image generation** and **multi-API fallback**.

**Key Features:**

```python
# Multiple API keys for load balancing
API_KEYS = [
    os.getenv("GOOGLE_API_KEY"),
    os.getenv("GOOGLE_API_KEY2"),
    os.getenv("GOOGLE_API_KEY3"),
    os.getenv("GOOGLE_API_KEY4"),
]

# Create clients for each key
clients = [genai.Client(api_key=key) for key in API_KEYS if key]

# Cycle through clients to avoid rate limits
client_cycle = itertools.cycle(clients)

def get_gemini_responses():
    """Uses next client in cycle (round-robin)"""
    client = next(client_cycle)
    response = client.models.generate_content(...)
```

**Why multiple APIs?**
- Google Gemini has rate limits (requests per minute)
- Using 4 API keys = 4x more requests possible
- Cycles through them to distribute load

**Functions:**
- `pil_to_base64()` - Convert PIL image to base64
- `generate_images_from_gpt()` - Call DALL-E 3 to generate images
- `resize_img()`, `resize_img2()` - Image resizing
- `_is_quota_exceeded()` - Check if API limit hit

---

### 8. **openai_utils.py** (56 Lines)

**What is it?**
**GPT-4 Vision API** wrapper for jewelry analysis.

**Function:**
```python
def ask_gpt_with_image(prompt: str, image_path=None, image_bytes=None):
    """
    Sends image + prompt to GPT-4 Vision
    Returns JSON response
    """
    # Convert image to base64
    img_b64 = base64.b64encode(image_bytes).decode("utf-8")
    
    # Call GPT-4
    completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[{
            "role": "user",
            "content": [
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{img_b64}"}}
            ]
        }],
        response_format={"type": "json_object"}  # Force JSON output
    )
    
    return json.loads(completion.choices[0].message.content)
```

**Used for:**
- Generating product descriptions
- Creating titles
- Extracting bullet points

---

### 9. **json_storage.py** (108 Lines)

**What is it?**
**Local database** for storing product information in JSON format.

**Functions:**

```python
def store_data_for_sku(sku_id: str, data: dict):
    """Saves product data to data_store.json"""
    # Loads existing data
    if JSON_PATH.exists():
        with open(JSON_PATH, 'r') as f:
            storage = json.load(f)
    else:
        storage = {}
    
    # Add/update product
    storage[sku_id] = data
    
    # Save back
    with open(JSON_PATH, 'w') as f:
        json.dump(storage, f)

def fetch_data_for_sku(sku_id: str) -> dict:
    """Retrieves product data if it exists"""
    if JSON_PATH.exists():
        with open(JSON_PATH, 'r') as f:
            storage = json.load(f)
        return storage.get(sku_id, {"sku_exists": False})
    return {"sku_exists": False}

def expand_bullet_points(data: dict):
    """Converts: {"bullet_points": {"bp_1": "text"}}
    To: {"bullet_point_1": "text", "bullet_point_2": "text"}"""
    # Flattens nested structure for Excel
```

**Data Structure:**
```json
{
  "SKU-001": {
    "product_name": "Diamond Earrings",
    "description": "Premium diamonds...",
    "price": "$49.99",
    "bullet_points": {
      "bullet_point_1": "High quality",
      "bullet_point_2": "Gold plated"
    }
  }
}
```

**Why?**
- Avoid regenerating descriptions for same product
- Cache results to save API costs
- Fast lookups without calling AI again

---

### 10. **Frontend Files (Streamlit UI)**

#### **jwellery_front.py** (Main UI - 85 Lines)

**What is it?**
Interactive web UI for uploading jewelry images and generating catalogs.

**Flow:**
```
User opens browser → Streamlit app loads
    ↓
User uploads multiple images
    ↓
User selects marketplace (Amazon/Flipkart)
    ↓
User selects jewelry type (Earrings/Necklace)
    ↓
User clicks "Generate Catalog"
    ↓
Sends request to FastAPI: /catalog-ai/
    ↓
Shows progress (processing images...)
    ↓
Returns JSON + Excel file link
    ↓
User downloads Excel file
```

**Key Code:**
```python
uploaded_files = st.file_uploader(
    "Upload images", 
    type=['jpg', 'jpeg', 'png'], 
    accept_multiple_files=True
)

if st.button("Generate Catalog"):
    # Send to API
    response = requests.post(
        f"{API_BASE_URL}/catalog-ai",
        data={
            "image_urls": ",".join(urls),
            "skuids": ",".join(skus),
            "type": "Earrings",
            "marketplace": "Amazon"
        }
    )
    excel_url = response.json()["excel_file"]
    st.download_button("Download Excel", excel_url)
```

#### **frontend.py** (Bill Processing - 119 Lines)

**What is it?**
Simple UI for extracting data from bill/invoice images.

**Flow:**
```
Upload bill image → AI extracts products → Shows table → Further filter → Download
```

---

### 11. **Web Scrapers**

#### **scraper.py** (752 Lines - Amazon Scraper)

**What is it?**
Scrapes product data from Amazon using **Playwright** (headless browser).

**Function:**
```python
def scrape_amazon_products(search_query):
    """
    1. Opens Amazon.com in headless browser
    2. Searches for query
    3. Extracts each product:
       - ASIN (ID)
       - Title
       - Price
       - Rating
       - Image URL
       - Bullet points
    4. Returns list of dicts
    """
```

**Output:**
```python
[
    {
        "asin": "B08JMYZ8J8",
        "title": "Diamond Earrings",
        "price": "$49.99",
        "rating": 4.5,
        "review_count": 234,
        "image_url": "https://amazon.com/image.jpg",
        "bullet_points": ["High quality", "Gold plated"]
    }
]
```

#### **flp_scraper.py** (216 Lines - Flipkart Scraper)

Same as above but scrapes **Flipkart.com** instead.

#### **myn_scraper.py** (150 Lines - Meesho Scraper)

Same as above but scrapes **Meesho.com** instead.

**Why 3 scrapers?**
- Each marketplace has different HTML structure
- Need marketplace-specific data (ratings, reviews format)
- Used for market research and competitive analysis

---

### 12. **generate_variants.py** (67 Lines)

**What is it?**
Generates multiple **image variations** from single jewelry image.

**What it does:**
```
Input: earring.jpg
    ↓
Generate 5 variations:
  1. White background version
  2. Colorful background #1
  3. Colorful background #2
  4. On hand/model
  5. With props
    ↓
Output: 5 images in generated_variants_earrings/
```

**Code:**
```python
def main():
    image = Image.open("IMG_6979.JPG")
    
    output_images = {
        "white_background.png": white_bgd_prompt,
        "multicolor_1.png": multicolor_1_prompt,
        "hand_photo.png": hand_prompt
    }
    
    for filename, prompt in output_images.items():
        generated = generate_images_from_gpt(prompt, image)
        generated.save(f"generated_variants_earrings/{filename}")
```

---

## 🗄️ Data Files Explanation

### **data_store.json**
```json
{
  "SKU-001": {
    "product_name": "Diamond Earrings",
    "description": "Premium...",
    "bullet_points": {"bp_1": "text", ...},
    "price": "$49.99"
  },
  "SKU-002": {...}
}
```
**Purpose:** Cache product data to avoid regenerating descriptions

### **indexed_hashes.json**
```json
{
  "hash_abc123def456": "image_fingerprint_data",
  "hash_xyz789": "image_fingerprint_data"
}
```
**Purpose:** Store image hashes for fast duplicate detection

### **int_hash_to_path.json**
```json
{
  "123": "https://s3.amazon.com/bucket/img_20250123_abc.jpg",
  "124": "https://s3.amazon.com/bucket/img_20250123_def.jpg"
}
```
**Purpose:** Map Milvus IDs to S3 URLs for image lookup

### **undo.json** (Auto-generated)
```json
[
  {"table": [...], "extras": {...}},
  {"table": [...], "extras": {...}}
]
```
**Purpose:** Keep undo history (last 10 actions)

### **bill_extra.csv, gemini_output_bill.csv**
Output files from bill processing endpoint

---

## 🔑 Environment Variables (.env file)

```
# Google Gemini API
GOOGLE_API_KEY=your_api_key_here
GOOGLE_API_KEY2=fallback_key_1
GOOGLE_API_KEY3=fallback_key_2
GOOGLE_API_KEY4=fallback_key_3

# OpenAI GPT-4
API_KEY_GPT=sk-...
PROJ_ID=your_project_id

# AWS S3
AWS_ACCESS_KEY_ID=AKIA...
AWS_SECRET_ACCESS_KEY=...
S3_BUCKET=alyaimg
S3_IMAGE_PREFIX=

# Milvus Vector Database
MILVUS_HOST=localhost
MILVUS_PORT=19530
COLLECTION_NAME=image_embeddings

# Zilliz Cloud (Optional - managed Milvus)
ZILLIZ_URL=https://...
ZILLIZ_TOKEN=...

# Application
API_BASE_URL=http://localhost:8000
```

---

## 🚀 How Data Flows Through The Application

### **Scenario 1: User Uploads Jewelry Image for Catalog**

```
Step 1: User opens Streamlit (jwellery_front.py)
        ↓
Step 2: Uploads image + enters SKU + selects "Amazon Earrings"
        ↓
Step 3: Frontend sends to FastAPI /catalog-ai endpoint
        ↓
Step 4: img_to_csv.py processes:
        - Downloads image from URL
        - Sends to Gemini AI with prompt from prompts.py
        - Gemini analyzes: Type, Color, Material, Gemstone
        - Returns JSON: {type: "Stud", color: "Gold", ...}
        ↓
Step 5: Check if SKU exists in data_store.json
        - If YES: Fetch cached description (skip GPT)
        - If NO: Call GPT-4 Vision to generate description
        ↓
Step 6: Merge all data:
        - Gemini analysis + GPT descriptions + fixed values from excel_fields.py
        ↓
Step 7: Write to Excel using openpyxl
        - Use marketplace-specific column mappings
        - Format according to Excel_fields.py
        ↓
Step 8: Upload Excel to S3
        ↓
Step 9: Return to user:
        {
          "results": [
            {"attributes": {all_product_data}},
            ...
          ],
          "excel_file": "https://s3.../catalog.xlsx"
        }
        ↓
Step 10: User downloads Excel file
```

### **Scenario 2: User Uploads Image & We Detect It's Duplicate**

```
Step 1: User uploads jewelry image via /generate_caption
        ↓
Step 2: image_search_engine.py processes:
        - Load CLIP model
        - Convert image to 768-dimensional vector
        ↓
Step 3: Search Milvus database
        - Compare with existing embeddings
        - Similarity = 97% (> 95% threshold)
        ↓
Step 4: Found duplicate!
        - Lookup S3 URL from int_hash_to_path.json
        - Return existing image without reprocessing
        ↓
Step 5: User sees:
        {
          "duplicate": "duplicate found",
          "s3_url": "https://..."
        }
```

### **Scenario 3: User Generates Image Variations**

```
Step 1: User runs generate_variants.py
        ↓
Step 2: Load original image (IMG_6979.JPG)
        ↓
Step 3: For each prompt in prompts2.py:
        - Call DALL-E 3 via utils2.py
        - Example: "Generate same earring on white background..."
        - DALL-E returns generated image
        ↓
Step 4: Save each variation:
        generated_variants_earrings/white_background.png
        generated_variants_earrings/colorful_1.png
        generated_variants_earrings/on_hand.png
        ↓
Step 5: All variants ready for upload
```

---

## 🔧 How To Run The Application

### **1. Install Dependencies**
```bash
pip install -r requirements.txt
```

### **2. Setup Environment Variables**
Create `.env` file with all API keys and credentials

### **3. Start Database** (if using local Milvus)
```bash
docker run -d -p 19530:19530 milvusdb/milvus:latest
```

### **4. Start FastAPI Server**
```bash
uvicorn img_to_csv:app --host 0.0.0.0 --port 8000 --reload
```

### **5. Start Streamlit Frontend** (in another terminal)
```bash
streamlit run jwellery_front.py
```

### **6. Access Application**
```
FastAPI Docs: http://localhost:8000/docs
Streamlit UI: http://localhost:8501
```

---

## 📊 Technology Stack

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **API Server** | FastAPI | REST API endpoints |
| **Frontend** | Streamlit | Web UI |
| **Image Analysis** | Google Gemini 2.0 Flash | AI image understanding |
| **Text Generation** | GPT-4 Vision | Marketing copy, descriptions |
| **Image Generation** | DALL-E 3 | Create product variations |
| **Image Similarity** | CLIP + PyTorch | Find duplicates |
| **Vector Database** | Milvus | Store embeddings |
| **Semantic Search** | ChromaDB | Find related products |
| **Cloud Storage** | AWS S3 | Store images |
| **Excel Writer** | openpyxl | Create catalogs |
| **Web Scraping** | Playwright | Competitive analysis |
| **Data Storage** | JSON | Local caching |

---

## 🐛 Common Issues & Solutions

### **Issue 1: "GOOGLE_API_KEY not found"**
```
Error: Warning: GOOGLE_API_KEY not found. Gemini client not initialized.
```
**Fix:**
1. Create `.env` file in root directory
2. Add: `GOOGLE_API_KEY=your_actual_key`
3. Restart FastAPI server

### **Issue 2: "Failed to upload to S3"**
```
Error: Failed to upload to S3
```
**Fix:**
1. Check AWS credentials in `.env`
2. Verify S3 bucket exists and is accessible
3. Check IAM permissions allow upload

### **Issue 3: "Milvus connection failed"**
```
Error: Failed to initialize Milvus connection
```
**Fix:**
1. Start Milvus: `docker run -d -p 19530:19530 milvusdb/milvus:latest`
2. Wait 30 seconds for startup
3. Restart FastAPI server

### **Issue 4: "Rate limit exceeded"**
```
Error: 429 Too Many Requests (Gemini API)
```
**Fix:**
1. Multiple API keys are configured for load balancing
2. Add more API keys to `.env`: `GOOGLE_API_KEY2`, `GOOGLE_API_KEY3`
3. utils2.py cycles through them automatically

### **Issue 5: "Excel file is empty or corrupted"**
```
Error: File cannot be opened by Excel
```
**Fix:**
1. Check if all required fields from excel_fields.py are present
2. Verify formatting in write_to_excel_* functions
3. Check S3 upload completed successfully

### **Issue 6: "Duplicate detection not working"**
```
Images not being detected as duplicates
```
**Fix:**
1. Verify Milvus is running and has data
2. Check indexed_hashes.json and int_hash_to_path.json exist
3. Run image re-indexing: `index_images_from_s3()`

---

## 🎯 Key Insights for Development

### **Why This Architecture?**

1. **Multiple AI Services:**
   - Gemini → Fast, cheap, good for structured extraction
   - GPT-4 → Better creativity for marketing copy
   - DALL-E → Image generation for variations
   - CLIP → Fast image fingerprinting

2. **Multiple Databases:**
   - Milvus → Vector embeddings (fast similarity search)
   - ChromaDB → Text semantic search
   - JSON → Simple product caching
   - S3 → Distributed storage

3. **Marketplace-Specific Prompts:**
   - Each marketplace needs different fields
   - Prompts.py defines exact requirements
   - Same image → different data per marketplace

4. **Fallback Mechanisms:**
   - Multiple API keys for rate limiting
   - Try-catch blocks for API failures
   - Local caching to avoid re-processing

5. **Async Processing:**
   - FastAPI uses `async` for concurrent requests
   - Can process 5 images in parallel
   - Better resource utilization

---

## 📈 Performance Optimization Tips

1. **Use Database Caching:**
   - Store product data in data_store.json
   - Avoid regenerating descriptions for same SKU

2. **Batch Process Images:**
   - Process 5 images simultaneously (line 950 in img_to_csv.py)
   - Reduce total processing time by 5x

3. **Image Indexing:**
   - Maintain indexed_hashes.json
   - Duplicate detection is O(1) lookup

4. **Multiple API Keys:**
   - Configure 4 Google API keys
   - Distribute load across them

5. **Local Image Caching:**
   - Downloaded images cached temporarily
   - Avoid re-downloading same image

---

## 🎓 Learning Path

If you're new to this codebase, learn in this order:

1. **img_to_csv.py** (Main endpoints)
2. **prompts.py** (AI instructions)
3. **excel_fields.py** (Data mapping)
4. **image_search_engine.py** (Duplicate detection)
5. **utils.py + utils2.py** (AI utilities)
6. **Frontend files** (User interface)
7. **Scrapers** (Data collection)

---

## 🔗 File Dependencies

```
img_to_csv.py (main)
    ├── prompts.py (AI instructions)
    ├── prompts2.py (Image generation)
    ├── excel_fields.py (Field mappings)
    ├── utils.py (Gemini utilities)
    ├── utils2.py (Image gen + GPT)
    ├── openai_utils.py (GPT-4)
    ├── image_search_engine.py (Duplicates)
    ├── json_storage.py (Local DB)
    ├── scraper.py (Web scraping)
    ├── flp_scraper.py (Flipkart scraper)
    └── myn_scraper.py (Meesho scraper)

jwellery_front.py (Streamlit UI)
    ├── requests (calls img_to_csv.py API)
    └── pandas (display results)

generate_variants.py
    ├── utils2.py (generate_images_from_gpt)
    ├── prompts2.py (image prompts)
    └── PIL (image processing)
```

---

## 🎯 What Each Marketplace Requires

### **Amazon Earrings**
- Required: Material, Occasion, Number of Gemstones
- Format: JSON structure with specific keys
- Uses: `gpt_amz_earrings_prompt` + `target_fields_earrings_amz`

### **Flipkart Earrings**
- Required: Type, Color, Collection, Piercing Required
- Format: JSON with "Type", "Earring Back Type"
- Uses: `gpt_flipkart_earrings_prompt` + `target_fields_earrings`

### **Meesho**
- Required: Simple set (Type, Color, Price)
- Format: Lightweight JSON
- Uses: `gpt_meesho_earrings_prompt` + `target_fields_earrings_meesho`

---

**Now you have a complete understanding of your entire codebase!** 🎉

Use this guide as a reference whenever you need to understand how components work together or troubleshoot issues.
