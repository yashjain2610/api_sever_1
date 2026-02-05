# File Modification Guide - What to Change Where

This guide shows you **exactly where** to modify code when you need to make changes.

---

## 🎯 QUICK REFERENCE TABLE

| Need | File | Line Range | What to Change |
|------|------|-----------|-----------------|
| Add new marketplace | `prompts.py` + `excel_fields.py` + `img_to_csv.py` | See below | Add prompts + field mappings + endpoint logic |
| Change Gemini model | `img_to_csv.py`, `utils.py`, `utils2.py` | Multiple | Replace 'gemini-2.0-flash' with new model |
| Fix duplicate detection | `image_search_engine.py` | Line ~200 | Change similarity threshold or model |
| Add new product style | `img_to_csv.py` | Line ~90 | Add to `mapp` dictionary |
| Change S3 bucket | `.env` | Line 1 | `S3_BUCKET=new_name` |
| Increase image limit | `img_to_csv.py` | Line ~950 | Change loop count or batch size |
| Add logging | Any file | Top | Add import logging, setup logger |
| Change Excel format | `excel_fields.py` | Specific section | Modify `target_fields_*` and `fixed_values_*` |

---

## 📝 DETAILED MODIFICATION GUIDES

---

## 1️⃣ ADD A NEW MARKETPLACE (Amazon → Alibaba)

### Step 1: Add Prompts in prompts.py

**Location:** `prompts.py` (Around line 1)

```python
# EXISTING CODE (lines 1-100)
prompt_questions_earrings_flipkart = """..."""
prompt_questions_earrings_amz = """..."""
# ... more prompts

# 👇 ADD NEW MARKETPLACE PROMPT HERE (after existing ones)
prompt_questions_earrings_alibaba = """
you are jewellery expert and an e-commerce catalog manager for Alibaba,
you will be given a image of a jewellery item.
answer the given questions.

Questions {
    {field_name: Product Type, question: What is the type?, options: [Stud, Hoop, Drop, ...]}
    {field_name: Metal, question: What is the base metal?, options: [Gold, Silver, Alloy, ...]}
    {field_name: Stones, question: Does it have gemstones?, options: [Yes, No]}
    ... (add more fields as needed)
}
Return as JSON: {"Product Type": "Stud", "Metal": "Gold", ...}
"""

# For description generation
gpt_alibaba_earrings_prompt = """Generate a professional product description for Alibaba...
Focus on: Quality, Value, Features suitable for Alibaba market."""

# Also add dimension prompts if needed
prompt_dimensions_earrings_alibaba = """Extract dimensions from the image...
Return JSON with width, height, diameter..."""
```

### Step 2: Add Field Mappings in excel_fields.py

**Location:** `excel_fields.py` (Around line 300-350)

```python
# EXISTING CODE
target_fields_earrings = [...]  # Flipkart fields
fixed_values_earrings = {...}   # Flipkart fixed values

target_fields_earrings_amz = [...] # Amazon fields
fixed_values_earrings_amz = {...}  # Amazon fixed values

# 👇 ADD NEW MARKETPLACE MAPPINGS
target_fields_earrings_alibaba = [
    "Product Type", "Metal", "Stones", "SKU", 
    "Title", "Description", "Price",
    # ... add all fields Alibaba requires
]

fixed_values_earrings_alibaba = {
    "Seller": "Our Store",
    "Shipping": "International",
    "Return Policy": "30 days",
    # ... add fixed values for Alibaba
}

# Repeat for necklace and bracelet
target_fields_necklace_alibaba = [...]
fixed_values_necklace_alibaba = {...}

target_fields_bracelet_alibaba = [...]
fixed_values_bracelet_alibaba = {...}
```

### Step 3: Add Prompts in prompts2.py (For descriptions)

**Location:** `prompts2.py` (At the end)

```python
# EXISTING CODE
gpt_flipkart_earrings_prompt = """..."""
gpt_amz_earrings_prompt = """..."""

# 👇 ADD NEW MARKETPLACE PROMPTS
gpt_alibaba_earrings_prompt = """
You are writing product descriptions for Alibaba marketplace.
Create compelling product description highlighting:
- Quality and durability
- Value for money
- Suitable for international customers
Format as detailed paragraph (3-4 sentences).
No preambles, return only the description.
"""

# Also for necklace and bracelet if needed
gpt_alibaba_necklace_prompt = """..."""
gpt_alibaba_bracelet_prompt = """..."""
```

### Step 4: Add Excel Writing Function in excel_fields.py

**Location:** `excel_fields.py` (At the end, after existing write_to_excel_* functions)

```python
# EXISTING CODE
def write_to_excel_flipkart(excel_results, ...):
    """..."""

def write_to_excel_amz_xl(excel_results, ...):
    """..."""

# 👇 ADD NEW MARKETPLACE FUNCTION
def write_to_excel_alibaba(excel_results, filename, target_fields, fixed_values):
    """
    Write product data to Excel in Alibaba format.
    
    Args:
        excel_results: List of (sku, response_json, description) tuples
        filename: Output Excel file path
        target_fields: List of column names for Alibaba
        fixed_values: Dict of fixed values for Alibaba
    """
    from openpyxl import load_workbook
    
    wb = load_workbook(filename)
    ws = wb.active
    
    # Find next empty row
    next_row = ws.max_row + 1
    
    # Write headers (if first time)
    if ws.max_row == 1 and ws[1][0].value is None:
        for col_idx, field in enumerate(target_fields, 1):
            ws.cell(1, col_idx, field)
    
    # Write data rows
    for idx, (sku, response_json, description) in enumerate(excel_results):
        row = next_row + idx
        
        # Write values from response_json and fixed_values
        for col_idx, field in enumerate(target_fields, 1):
            value = response_json.get(field) or fixed_values.get(field, "")
            ws.cell(row, col_idx, value)
    
    wb.save(filename)
```

### Step 5: Update img_to_csv.py Endpoint

**Location:** `img_to_csv.py` (Around line 1000-1100)

```python
# EXISTING CODE (lines 1000-1050)
input_prompt_map = {
    "fli_ear": ([prompt_questions_earrings_flipkart], fixed_values_earrings, gpt_flipkart_earrings_prompt),
    "ama_ear": ([prompt_questions_earrings_amz], fixed_values_earrings_amz, gpt_amz_earrings_prompt),
    "mee_ear": ([prompt_questions_earrings_meesho], fixed_values_earrings_meesho, gpt_meesho_earrings_prompt),
    "fli_nec": ([prompt_questions_necklace_flipkart], fixed_values_necklace_flipkart, gpt_flipkart_necklace_prompt),
    "ama_nec": ([prompt_questions_necklace_amz], fixed_values_necklace_amz, gpt_amz_necklace_prompt),
    "mee_nec": ([prompt_questions_necklace_meesho], fixed_values_necklace_meesho, gpt_meesho_necklace_prompt),
    
    # 👇 ADD NEW MARKETPLACE ENTRIES
    "ali_ear": ([prompt_questions_earrings_alibaba], fixed_values_earrings_alibaba, gpt_alibaba_earrings_prompt),
    "ali_nec": ([prompt_questions_necklace_alibaba], fixed_values_necklace_alibaba, gpt_alibaba_necklace_prompt),
    "ali_bra": ([prompt_questions_bracelet_alibaba], fixed_values_bracelet_alibaba, gpt_alibaba_bracelet_prompt),
}

# EXISTING CODE (lines 1120-1140)
dims_prompt_map = {
    "fli_ear": [prompt_dimensions_earrings_flipkart],
    "ama_ear": [prompt_dimensions_earrings_amz],
    # ... more dimensions
    
    # 👇 ADD DIMENSIONS IF NEEDED
    "ali_ear": [prompt_dimensions_earrings_alibaba],
}

# EXISTING CODE (lines 1200-1250) - Inside catalog-ai function
filename_map = {
    "fli_ear": "earrings_flipkart.xlsx",
    "ama_ear": "earrings_amz.xlsx",
    "mee_ear": "earrings_meesho.xlsx",
    # ... more filenames
    
    # 👇 ADD NEW MARKETPLACE FILENAMES
    "ali_ear": "earrings_alibaba.xlsx",
    "ali_nec": "necklace_alibaba.xlsx",
    "ali_bra": "bracelet_alibaba.xlsx",
}

# EXISTING CODE (lines 1260-1300) - Excel writing section
if format == "fli_ear":
    write_to_excel_flipkart(excel_results, filename=tmp_path, target_fields=..., fixed_values=...)
elif format == "ama_ear":
    write_to_excel_amz_xl(excel_results, filename=tmp_path, target_fields=..., fixed_values=...)
# ... more elif for other formats

# 👇 ADD NEW MARKETPLACE ELIF BLOCK
elif format == "ali_ear":
    write_to_excel_alibaba(excel_results, filename=tmp_path, target_fields=target_fields_earrings_alibaba, fixed_values=fixed_values_earrings_alibaba)
elif format == "ali_nec":
    write_to_excel_alibaba(excel_results, filename=tmp_path, target_fields=target_fields_necklace_alibaba, fixed_values=fixed_values_necklace_alibaba)
elif format == "ali_bra":
    write_to_excel_alibaba(excel_results, filename=tmp_path, target_fields=target_fields_bracelet_alibaba, fixed_values=fixed_values_bracelet_alibaba)
```

### Step 6: Update clear_excel_file Function

**Location:** `img_to_csv.py` (Around line 1410)

```python
# EXISTING CODE
def clear_excel_file(type: str = Form(...), marketplace: str = Form(...)):
    format = marketplace.lower()[:3] + "_" + type.lower()[:3]
    
    filename_map = {
        "fli_ear": "earrings_flipkart.xlsx",
        "ama_ear": "earrings_amz.xlsx",
        # ... existing mappings
        
        # 👇 ADD NEW MARKETPLACE
        "ali_ear": "earrings_alibaba.xlsx",
        "ali_nec": "necklace_alibaba.xlsx",
        "ali_bra": "bracelet_alibaba.xlsx",
    }
```

### Summary of Changes for New Marketplace:

✅ `prompts.py` - Add marketplace-specific questions
✅ `prompts.py` - Add marketplace-specific GPT prompts  
✅ `excel_fields.py` - Add target fields list
✅ `excel_fields.py` - Add fixed values dict
✅ `excel_fields.py` - Add write_to_excel_* function
✅ `img_to_csv.py` - Update input_prompt_map
✅ `img_to_csv.py` - Update dims_prompt_map
✅ `img_to_csv.py` - Update filename_map
✅ `img_to_csv.py` - Add elif blocks in catalog-ai
✅ `img_to_csv.py` - Update clear_excel_file

---

## 2️⃣ CHANGE AI MODEL

### Change Gemini Model (e.g., gemini-2.0-flash → gemini-pro)

**Step 1: In utils.py**

```python
# Location: utils.py (around line 45)
# BEFORE
response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=contents,
    config=config
)

# AFTER
response = client.models.generate_content(
    model='gemini-pro',  # ← CHANGE HERE
    contents=contents,
    config=config
)
```

**Step 2: In utils2.py**

```python
# Location: utils2.py (around line 80)
# BEFORE
response = next(client_cycle).models.generate_content(
    model='gemini-2.0-flash',
    contents=contents,
    ...
)

# AFTER
response = next(client_cycle).models.generate_content(
    model='gemini-pro',  # ← CHANGE HERE
    contents=contents,
    ...
)
```

**Step 3: In img_to_csv.py (Multiple locations)**

```python
# Location: img_to_csv.py (around line 280)
response = client.models.generate_content(
    model='gemini-2.0-flash',  # ← CHANGE ALL
    contents=[...],
    config=types.GenerateContentConfig(...)
)

# Location: img_to_csv.py (around line 300)
coll_title = client.models.generate_content(
    model='gemini-2.0-flash',  # ← CHANGE ALL
    contents=[...],
    config=types.GenerateContentConfig(...)
)

# Search for all occurrences:
# Ctrl+F: 'gemini-2.0-flash'
# Replace all with: 'gemini-pro'
```

---

## 3️⃣ ADD NEW PRODUCT STYLE

### Add "Retro" Style to mapp Dictionary

**Location:** `img_to_csv.py` (Around line 90)

```python
# EXISTING CODE (lines 85-110)
mapp={
    "Elegence": "Jewelry that remains timeless...",
    "Simplicity": "Pieces designed with less complexity...",
    "Heritage": "Designs that evoke a sense of nostalgia...",
    # ... many more styles

    # 👇 ADD NEW STYLE
    "Retro": "Jewelry with vintage 1950s-1980s influences, featuring bold shapes, bright colors, and nostalgic patterns that capture the playful spirit of past decades.",
    
    "Victorian": "Jewelry with intricate designs...",
    # ... more existing styles
}
```

**Important:** The key (e.g., "Retro") must match what users select in the frontend.

---

## 4️⃣ CHANGE DUPLICATE DETECTION THRESHOLD

### Adjust Similarity Score

**Location:** `img_to_csv.py` (Around line 565)

```python
# BEFORE (95% match = duplicate)
if 100 - dist >= 95:
    path = id_to_path.get(str(_id), "unknown")
    matches.append({"id": _id, "distance": dist, "path": path})

# AFTER (90% match = duplicate - more sensitive)
if 100 - dist >= 90:  # ← CHANGE THRESHOLD
    path = id_to_path.get(str(_id), "unknown")
    matches.append({"id": _id, "distance": dist, "path": path})

# AFTER (98% match = duplicate - less sensitive)
if 100 - dist >= 98:  # ← STRICTER
    path = id_to_path.get(str(_id), "unknown")
    matches.append({"id": _id, "distance": dist, "path": path})
```

**What This Means:**
- `>= 95` = More false positives (detects similar images)
- `>= 98` = More false negatives (misses some duplicates)
- Adjust based on your needs

---

## 5️⃣ CHANGE IMAGE PROCESSING BATCH SIZE

### Process More/Fewer Images Simultaneously

**Location:** `img_to_csv.py` (Around line 1230)

```python
# EXISTING CODE (inside catalog-ai function)
count = 1
async with httpx.AsyncClient() as client:
    for url, skuid in zip(url_list, skuid_list):
        # ... process image ...
        
        count += 1
        if count%5 == 0:  # ← CHANGE THIS NUMBER
            time.sleep(30)  # Wait every 5 images

# CHANGE TO:
if count%10 == 0:  # Wait every 10 images (faster)
    time.sleep(30)

# OR:
if count%3 == 0:  # Wait every 3 images (safer, fewer rate limits)
    time.sleep(30)
```

---

## 6️⃣ CHANGE S3 BUCKET

### Update Cloud Storage Location

**Option 1: Via .env File (RECOMMENDED)**

```
# File: .env
# BEFORE
S3_BUCKET=alyaimg

# AFTER
S3_BUCKET=my-new-bucket-name
```

**Option 2: In Code (NOT RECOMMENDED)**

```python
# Location: img_to_csv.py (line 57)
# BEFORE
S3_BUCKET = os.getenv("S3_BUCKET", "alyaimg")

# AFTER
S3_BUCKET = os.getenv("S3_BUCKET", "my-new-bucket")

# Also in image_search_engine.py (line 27)
# BEFORE
S3_BUCKET = os.getenv("S3_BUCKET", "alyaimg")

# AFTER
S3_BUCKET = os.getenv("S3_BUCKET", "my-new-bucket")
```

---

## 7️⃣ ADD LOGGING TO TRACK ISSUES

### Add Detailed Logging Throughout Code

**Step 1: Setup Logging (at top of file)**

```python
# Location: img_to_csv.py (after imports, line 45)
import logging

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("app.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
```

**Step 2: Add Logs to Key Functions**

```python
# Location: catalog-ai endpoint (around line 1045)

async def catalog_ai(req: CatalogRequest):
    logger.info(f"Starting catalog generation for {req.marketplace} {req.type}")
    logger.info(f"Processing {len(url_list)} images")
    
    try:
        # ... existing code ...
        for url, skuid in zip(url_list, skuid_list):
            logger.debug(f"Processing image: {url}")
            
            response_list = get_gemini_responses("Analyze this", image_data, input_prompts)
            logger.debug(f"Gemini response: {response_list[0][:100]}...")
            
            # ... more processing ...
            logger.info(f"✓ Processed {skuid}")
    
    except Exception as e:
        logger.error(f"Error in catalog generation: {str(e)}", exc_info=True)
        raise

    logger.info("Catalog generation completed successfully")
    return results
```

**Step 3: Check Logs**

```bash
# On Linux/Mac
tail -f app.log

# On Windows PowerShell
Get-Content app.log -Wait

# Or open the file:
app.log
```

---

## 8️⃣ CHANGE EXCEL OUTPUT FORMAT

### Modify Column Order or Add New Columns

**Location:** `excel_fields.py` (Specific marketplace section)

```python
# BEFORE
target_fields_earrings_amz = [
    "Seller SKU ID",
    "Model Name",
    "Type",
    "Color",
    "Base Material",
    # ... more fields
]

# AFTER (added new field, changed order)
target_fields_earrings_amz = [
    "Seller SKU ID",
    "Model Name",
    "Type",
    "Color",
    "Base Material",
    "New Field 1",      # ← NEW FIELD ADDED
    "New Field 2",      # ← NEW FIELD ADDED
    # ... more fields
]
```

**IMPORTANT:** Also update corresponding Gemini prompts to extract the new field!

---

## 9️⃣ CHANGE MAXIMUM FILE SIZE LIMIT

### Increase Image Size Limit

**Location:** `img_to_csv.py` (inside `/generate_caption` endpoint, around line 525)

```python
# BEFORE (implicit limit from file upload)
async def generate_caption(file: UploadFile = File(...), type: str = Form(...)):
    image_bytes = await file.read()

# AFTER (add explicit check)
async def generate_caption(file: UploadFile = File(...), type: str = Form(...)):
    # Read file
    image_bytes = await file.read()
    
    # Check size (10MB limit)
    MAX_FILE_SIZE = 10 * 1024 * 1024  # 10MB
    if len(image_bytes) > MAX_FILE_SIZE:
        raise HTTPException(
            status_code=413,
            detail=f"File too large. Max size: 10MB, Got: {len(image_bytes)/(1024*1024):.2f}MB"
        )
    
    # Continue processing...
```

---

## 🔟 CHANGE API RATE LIMITING

### Add Request Rate Limiting

**Location:** `img_to_csv.py` (after imports)

```python
from fastapi_limiter import FastAPILimiter
from fastapi_limiter.util import get_remote_address
import redis.asyncio as redis

# Setup rate limiter
async def startup():
    redis_conn = await redis.from_url("redis://localhost")
    FastAPILimiter.init(redis_conn, key_func=get_remote_address)

app.add_event_handler("startup", startup)

# Apply to specific endpoint
from fastapi_limiter.depends import RateLimiter

@app.post("/catalog-ai")
@limiter.limit("5/minute")  # 5 requests per minute
async def catalog_ai(req: CatalogRequest):
    # ... existing code ...
```

---

## 1️⃣1️⃣ CHANGE JSON STORAGE PATH

### Store Data in Different Location

**Location:** `json_storage.py` (line 3)

```python
# BEFORE
JSON_PATH = Path("data_store.json")

# AFTER
JSON_PATH = Path("./storage/data_store.json")

# Or from environment
JSON_PATH = Path(os.getenv("DATA_STORE_PATH", "data_store.json"))
```

**Also Update:** Add .env variable if using environment
```
DATA_STORE_PATH=./storage/data_store.json
```

---

## 1️⃣2️⃣ CHANGE MILVUS DATABASE SETTINGS

### Use Cloud Milvus (Zilliz) Instead of Local

**Location:** `image_search_engine.py` (Around line 90-120)

```python
# BEFORE (Local Milvus)
def init_milvus(host="localhost", port="19530", collection_name="image_embeddings"):
    connections.connect(
        "default",
        host=host,
        port=port
    )

# AFTER (Cloud Milvus)
def init_milvus(host=None, port=None, collection_name="image_embeddings"):
    url = os.getenv("ZILLIZ_URL")
    token = os.getenv("ZILLIZ_TOKEN")
    
    connections.connect(
        "default",
        uri=url,
        token=token
    )
```

**Also Update .env:**
```
ZILLIZ_URL=https://your-cluster-url:19530
ZILLIZ_TOKEN=your_token_here
```

---

## 1️⃣3️⃣ MODIFY UNDO HISTORY LIMIT

### Keep More or Fewer Undo Actions

**Location:** `img_to_csv.py` (Around line 305)

```python
def manage_undo_file(new_data, file_path='undo.json'):
    # ... load existing data ...
    
    data.append(new_data)
    
    # BEFORE (keep only 10 actions)
    if len(data) > 10:
        data.pop(0)
    
    # AFTER (keep 50 actions)
    if len(data) > 50:  # ← CHANGE THIS NUMBER
        data.pop(0)
    
    # Save to file
    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)
```

---

## 1️⃣4️⃣ ADD CUSTOM MARKETPLACE FIELD VALIDATION

### Ensure Required Fields Have Values

**Location:** `excel_fields.py` (at end)

```python
# Add validation function
def validate_product_data(data: dict, marketplace: str, jewelry_type: str) -> list:
    """
    Validates that all required fields are filled.
    Returns list of missing fields.
    """
    # Get required fields for this marketplace+type
    if marketplace == "amazon" and jewelry_type == "earrings":
        required_fields = target_fields_earrings_amz
    elif marketplace == "flipkart" and jewelry_type == "earrings":
        required_fields = target_fields_earrings
    else:
        return []  # Unknown marketplace
    
    missing = []
    for field in required_fields:
        if field not in data or not data[field]:
            missing.append(field)
    
    return missing
```

**Use in img_to_csv.py:**
```python
# After merging all data
missing = validate_product_data(final_response, req.marketplace, req.type)
if missing:
    logger.warning(f"Missing fields for {skuid}: {missing}")
    # Either warn user or fill with defaults
```

---

## SUMMARY OF COMMON MODIFICATIONS

| Change | File | Line | Old Value | New Value |
|--------|------|------|-----------|-----------|
| Change model | multiple | varies | 'gemini-2.0-flash' | 'gemini-pro' |
| Change bucket | .env | 1 | S3_BUCKET=alyaimg | S3_BUCKET=new |
| Add style | img_to_csv.py | 90 | mapp={...} | mapp={..., "New": "..."} |
| Change threshold | img_to_csv.py | 565 | >= 95 | >= 90 |
| Change batch size | img_to_csv.py | 1230 | count%5 | count%10 |
| Undo history | img_to_csv.py | 310 | > 10 | > 50 |
| Change host | image_search_engine.py | 110 | "localhost" | new_host |
| Max file size | img_to_csv.py | 525 | implicit | 10 * 1024 * 1024 |

---

## ✅ TESTING YOUR CHANGES

After modifying code, always test:

```python
# 1. Syntax check
python -m py_compile modified_file.py

# 2. Import check
python -c "import modified_module"

# 3. Unit test specific function
python -c "from module import function; function(test_input)"

# 4. Full integration test
# Upload test image → Check output

# 5. Edge case test
# Try with empty input, large file, special characters
```

---

## 🔄 VERSION CONTROL BEST PRACTICE

Before making changes:

```bash
# Check what you're about to change
git diff modified_file.py

# Make backup
git stash

# Make changes

# Check if it works
# Test your changes

# If good: commit
git add modified_file.py
git commit -m "Description of change"

# If bad: revert
git checkout modified_file.py
```

---

**Remember:** Always test changes in a development environment before deploying to production! 🚀
