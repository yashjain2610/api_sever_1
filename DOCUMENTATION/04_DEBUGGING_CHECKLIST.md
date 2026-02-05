# Debugging Checklist - Complete Guide

This checklist helps you systematically find and fix issues in the application.

---

## 🔍 BEFORE YOU START DEBUGGING

### Initial Assessment

- [ ] **Check Error Message:**
  - Is it a clear error or vague?
  - Does it show line number?
  - Is it from your code or external API?

- [ ] **Reproduce the Issue:**
  - Can you reproduce it consistently?
  - Does it happen with specific inputs only?
  - Works on different images? Different SKUs?

- [ ] **Check Recent Changes:**
  - Did you modify code recently?
  - Did API keys change?
  - Did deployment environment change?

---

## 🚀 APPLICATION NOT STARTING

### FastAPI Server Won't Start

**Checklist:**
```
[ ] Is Python installed? 
    Run: python --version
    Expected: Python 3.8+

[ ] Are dependencies installed?
    Run: pip install -r requirements.txt
    Look for: "Successfully installed"

[ ] Is .env file present in root?
    Check: ls -la .env
    If missing: Create it with API keys

[ ] Is port 8000 available?
    Run: netstat -ano | findstr :8000
    If in use: Kill process or use different port

[ ] Any syntax errors in img_to_csv.py?
    Run: python -m py_compile img_to_csv.py
    Look for: SyntaxError messages

[ ] Try running server with detailed output:
    Run: python -m uvicorn img_to_csv:app --reload
    Look for: "Uvicorn running on http://127.0.0.1:8000"
```

**Common Fixes:**
```python
# If import error for module X:
pip install X

# If GOOGLE_API_KEY error:
# Edit .env file and add:
GOOGLE_API_KEY=your_key_here

# If port already in use:
uvicorn img_to_csv:app --port 9000
```

---

## 🖼️ IMAGE UPLOAD ISSUES

### Issue: "File not found" or "Cannot read image"

**Checklist:**
```
[ ] Image file format supported?
    Supported: jpg, jpeg, png, webp
    Run: file image.jpg (shows format)

[ ] File is not corrupted?
    Try: from PIL import Image; Image.open("image.jpg")
    If error: File is corrupted

[ ] File size not too large?
    Max recommended: 10MB
    Check: ls -lh image.jpg (shows size)

[ ] Correct path in code?
    Check: file path is absolute or relative correctly

[ ] Temporary file cleanup?
    Check: tempfile.NamedTemporaryFile is closing properly
```

**Debug Code:**
```python
# Add to your code to test image loading
from PIL import Image
import io

image_bytes = await file.read()
try:
    image = Image.open(io.BytesIO(image_bytes))
    print(f"✓ Image loaded: {image.format}, {image.size}")
except Exception as e:
    print(f"✗ Image error: {e}")
    raise HTTPException(400, f"Invalid image: {str(e)}")
```

---

## 🤖 GEMINI API ISSUES

### Issue: "GOOGLE_API_KEY not found"

**Checklist:**
```
[ ] .env file exists?
    Run: cat .env | grep GOOGLE_API_KEY

[ ] API key is correct?
    Check: Key starts with "AIza..." or similar
    Try: Manual API call to verify key works

[ ] env file loaded in code?
    Check utils.py line 15: load_dotenv()
    Should be called BEFORE using os.getenv()

[ ] Running from correct directory?
    .env must be in root directory where you run server
    Run: pwd (shows current directory)
```

**Fix:**
```python
# In utils.py, ensure this is at top:
from dotenv import load_dotenv
import os

load_dotenv()  # MUST be before os.getenv()

API_KEY = os.getenv("GOOGLE_API_KEY")
if not API_KEY:
    print("ERROR: GOOGLE_API_KEY not found in .env")
    raise ValueError("Missing GOOGLE_API_KEY")
```

### Issue: "Failed to initialize Gemini client"

**Checklist:**
```
[ ] API key valid and has credits?
    Go to: console.cloud.google.com
    Check: Billing enabled
    Check: Quota remaining

[ ] Correct model name used?
    Check img_to_csv.py for: 'gemini-2.0-flash'
    This model must be enabled in Google Cloud

[ ] Rate limit exceeded?
    Error: 429 Too Many Requests
    Check: How many requests in last minute?
    Fix: Add delays or use multiple API keys

[ ] Network connectivity?
    Run: ping google.com
    Check: Firewall blocking API calls
```

**Debug Code:**
```python
from google import genai

try:
    client = genai.Client(api_key=API_KEY)
    response = client.models.generate_content(
        model='gemini-2.0-flash',
        contents=["Test message"]
    )
    print("✓ Gemini API works!")
except Exception as e:
    print(f"✗ Gemini error: {type(e).__name__}: {e}")
    # Log full error for debugging
    import traceback
    traceback.print_exc()
```

### Issue: "Gemini returns empty response or error"

**Checklist:**
```
[ ] Prompt is valid?
    Check: Prompt not None, not empty
    Check: No special characters breaking JSON

[ ] Content is safe?
    Gemini has content filters
    Try: Removing explicit content from image

[ ] Image format correct for Gemini?
    Check utils.py: input_image_setup()
    Must be: ["Part.from_bytes(data=bytes, mime_type='image/jpeg')"]

[ ] Response parsing correct?
    Check: if gemini_response[0]=="`: removing markers
    Check: json.loads() is valid JSON
```

**Debug Code:**
```python
# Log full response to see what Gemini returned
response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=[prompt]
)

print(f"Raw response: {response}")
print(f"Response text: {response.text}")
print(f"Response length: {len(response.text)}")

# Check if it looks like JSON
if response.text.startswith("{"):
    print("✓ Looks like JSON")
else:
    print("✗ Not JSON format")
```

---

## 💾 AWS S3 ISSUES

### Issue: "Failed to upload to S3"

**Checklist:**
```
[ ] AWS credentials valid?
    Run: aws configure
    Check: Access key ID and secret key correct

[ ] S3 bucket exists?
    Run: aws s3 ls
    Look for: Your bucket name in list

[ ] Bucket permissions correct?
    Check: IAM policy allows s3:PutObject
    Test: aws s3 cp test.txt s3://your-bucket/

[ ] Credentials in environment?
    Check .env has:
    AWS_ACCESS_KEY_ID=...
    AWS_SECRET_ACCESS_KEY=...
    S3_BUCKET=your-bucket-name

[ ] Region correct?
    If bucket in different region, add to .env:
    AWS_REGION=us-west-2
```

**Debug Code:**
```python
import boto3
import os

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    region_name=os.getenv("AWS_REGION", "us-east-1")
)

try:
    # Test upload
    s3.put_object(
        Bucket=os.getenv("S3_BUCKET"),
        Key="test.txt",
        Body=b"test content"
    )
    print("✓ S3 upload works!")
except Exception as e:
    print(f"✗ S3 error: {e}")
    import traceback
    traceback.print_exc()
```

### Issue: "Image uploaded but URL is invalid"

**Checklist:**
```
[ ] S3 URL format correct?
    Check: URL is https://bucket.s3.amazonaws.com/key
    Not: file:// or plain path

[ ] File is public?
    S3 bucket policy must allow public read
    Check: AWS console → Bucket policy

[ ] URL points to actual file?
    Copy URL to browser, try accessing
    If 403: Permissions issue
    If 404: File not uploaded

[ ] Special characters in filename?
    Check: Filename has spaces or special chars
    Fix: Use URL encoding or safe names
```

---

## 📊 MILVUS DATABASE ISSUES

### Issue: "Failed to initialize Milvus connection"

**Checklist:**
```
[ ] Milvus server running?
    Run: docker ps | grep milvus
    If not running, start it:
    docker run -d -p 19530:19530 milvusdb/milvus:latest

[ ] Port 19530 accessible?
    Run: telnet localhost 19530
    If connection refused: Milvus not running

[ ] Milvus host/port correct in .env?
    Check: MILVUS_HOST=localhost
    Check: MILVUS_PORT=19530

[ ] Wait for startup?
    Milvus takes 30-60 seconds to start
    Run again after waiting

[ ] Memory sufficient?
    Milvus needs 2GB+ RAM
    Check: docker stats milvus
```

**Debug Code:**
```python
from pymilvus import connections

try:
    connections.connect(
        host=os.getenv("MILVUS_HOST", "localhost"),
        port=int(os.getenv("MILVUS_PORT", "19530"))
    )
    print("✓ Milvus connection works!")
except Exception as e:
    print(f"✗ Milvus error: {e}")
```

### Issue: "Image embedding dimension mismatch"

**Checklist:**
```
[ ] CLIP model produces correct dimension?
    Check: embed_image() returns 768-dim vector
    CLIP model should output: 768 dimensions

[ ] Milvus collection schema correct?
    Check image_search_engine.py: CollectionSchema
    Should have: dimension=768 for embeddings

[ ] All embeddings same size?
    Different image encoders = different sizes
    Don't mix CLIP with other encoders
```

---

## 🔍 DUPLICATE DETECTION NOT WORKING

### Issue: "Same images not detected as duplicates"

**Checklist:**
```
[ ] Milvus database has data?
    Check: indexed_hashes.json not empty
    Check: int_hash_to_path.json has entries

[ ] Similarity threshold correct?
    Check img_to_csv.py line ~565:
    if 100 - dist >= 95:  # 95% match threshold
    Try lowering to 90 if too strict

[ ] Images actually similar?
    Just because similar to human eye doesn't mean AI sees them same
    Example: Different angle or lighting = different embedding

[ ] Index not updated?
    After uploading images, did you call:
    index_single_image_from_s3()?
    Or index_images_from_s3()?
```

**Debug Code:**
```python
# Check if image data exists
import json

with open("int_hash_to_path.json", "r") as f:
    id_map = json.load(f)
    print(f"Total images in database: {len(id_map)}")

# Test similarity manually
from image_search_engine import embed_image, search_similar
from PIL import Image

image = Image.open("test.jpg")
embedding = embed_image(image, clipmodel, processor, device)
results = search_similar(collection_db, embedding, top_k=5)

for _id, distance in results:
    print(f"ID: {_id}, Similarity Score: {100-distance:.2f}%")
```

---

## 📝 EXCEL FILE ISSUES

### Issue: "Excel file is empty or corrupted"

**Checklist:**
```
[ ] openpyxl library installed?
    Run: pip show openpyxl
    If missing: pip install openpyxl

[ ] All required fields present?
    Check excel_fields.py for required columns
    Check: target_fields_earrings, fixed_values_earrings

[ ] Data format matches expectations?
    Check: No None values in required fields
    Check: Text not too long for cells

[ ] File write successful?
    Check: return value from write_to_excel_* functions
    Add logging before return

[ ] File still being written?
    Check: flush() and close() called
    Wait 2-3 seconds after write before download
```

**Debug Code:**
```python
from openpyxl import load_workbook

try:
    wb = load_workbook("output.xlsx")
    ws = wb.active
    
    print(f"✓ Excel file valid")
    print(f"Rows: {ws.max_row}")
    print(f"Columns: {ws.max_column}")
    
    # Check first row
    for col in range(1, ws.max_column + 1):
        cell_value = ws.cell(1, col).value
        print(f"Column {col}: {cell_value}")
        
except Exception as e:
    print(f"✗ Excel error: {e}")
```

### Issue: "Data written to wrong columns"

**Checklist:**
```
[ ] Column mapping correct?
    Check excel_fields.py
    Compare with actual column names in template

[ ] Field order matches?
    Check: Order of columns in target_fields
    Should match Excel column order

[ ] Fixed values correct?
    Check: fixed_values dict
    Verify values match requirements
```

---

## 📊 JSON STORAGE ISSUES

### Issue: "data_store.json file not found or not readable"

**Checklist:**
```
[ ] File exists?
    Run: ls -la data_store.json

[ ] File is valid JSON?
    Run: python -m json.tool data_store.json
    If error: JSON is corrupted

[ ] File permissions?
    Run: chmod 644 data_store.json

[ ] Proper JSON structure?
    Should be: { "SKU-001": {...}, "SKU-002": {...} }
```

**Debug Code:**
```python
import json
from pathlib import Path

file_path = Path("data_store.json")

if not file_path.exists():
    print("✗ data_store.json not found")
else:
    try:
        with open(file_path, 'r') as f:
            data = json.load(f)
        print(f"✓ Valid JSON with {len(data)} entries")
    except json.JSONDecodeError as e:
        print(f"✗ Invalid JSON: {e}")
```

### Issue: "Product data not found for SKU"

**Checklist:**
```
[ ] SKU exists in data_store.json?
    Check: json.load()["SKU-001"] key exists
    Or: fetch_data_for_sku("SKU-001")

[ ] SKU format matches?
    Check: Case sensitivity (SKU-001 vs sku-001)
    Check: No extra spaces

[ ] Data saved after processing?
    Check: store_data_for_sku() called
    Check: No exception before storing
```

---

## 🌐 GEMINI RATE LIMIT ISSUES

### Issue: "429 Too Many Requests"

**Checklist:**
```
[ ] How many requests per minute?
    Google Gemini limit: ~60 requests/minute per key
    Your app sends: 2 requests per image (Gemini + GPT)

[ ] Multiple API keys configured?
    Check .env:
    GOOGLE_API_KEY=key1
    GOOGLE_API_KEY2=key2
    GOOGLE_API_KEY3=key3
    GOOGLE_API_KEY4=key4

[ ] Using client cycle?
    Check utils2.py:
    client_cycle = itertools.cycle(clients)
    client = next(client_cycle)

[ ] Add delays between requests?
    Check: time.sleep(2) between requests
    In img_to_csv.py line ~1230: if count%5 == 0: time.sleep(30)
```

**Quick Fix:**
```python
# Add to img_to_csv.py before API call
import time
time.sleep(1)  # Wait 1 second between requests

# Or for batch processing
if count % 5 == 0:
    time.sleep(30)  # Wait 30 sec after every 5 images
```

---

## 🎨 IMAGE ANALYSIS INCORRECT

### Issue: "Gemini returns wrong product details"

**Checklist:**
```
[ ] Prompt is clear enough?
    Check prompts.py - detailed prompts help Gemini
    Example: "What is the TYPE of earring?"
    With options helps Gemini choose correctly

[ ] Image quality good?
    Low quality images = poor analysis
    Check: Image not blurry, small, or dark

[ ] Product visible in image?
    Background objects confusing Gemini
    Check: Product is main focus

[ ] Options in prompt match reality?
    Example: Prompt asks for "Gold or Silver"
    But image shows "Rose Gold"
    Fix: Add "Rose Gold" to options

[ ] Test with manual Gemini?
    Go to https://gemini.google.com
    Upload same image, ask same question
    See what Gemini returns
```

**Debug Code:**
```python
# Test Gemini response
response = client.models.generate_content(
    model='gemini-2.0-flash',
    contents=[
        "Analyze this jewelry image carefully",
        image_data[0],
        "What color is the jewelry?"
    ]
)

print(f"Gemini says: {response.text}")
# Compare with actual color in image
```

---

## ✅ RESPONSE PARSING ERRORS

### Issue: "Failed to parse Gemini response as JSON"

**Checklist:**
```
[ ] Response wrapped in markdown?
    Check: if response.startswith("```")
    Fix: Strip ```json and ``` markers

[ ] Missing comma or quotes in JSON?
    Check: json.loads() error message
    Validate: Use online JSON validator

[ ] Response incomplete?
    Gemini might cut off response
    Check: max_output_tokens setting
    Increase if needed

[ ] Extra text before/after JSON?
    Gemini adds preamble sometimes
    Fix: Extract JSON part only
```

**Debug Code:**
```python
import json
import re

response_text = gemini_response

# Try to extract JSON
json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
if json_match:
    json_str = json_match.group(0)
else:
    json_str = response_text

# Remove markdown
json_str = json_str.strip('`').replace('```json', '').replace('```', '')

try:
    data = json.loads(json_str)
    print("✓ JSON parsed successfully")
except json.JSONDecodeError as e:
    print(f"✗ JSON error: {e}")
    print(f"String: {json_str[:200]}...")
```

---

## 🔗 ENDPOINT NOT RESPONDING

### Issue: "Connection refused" or "Timeout"

**Checklist:**
```
[ ] Server running?
    Check: uvicorn running in terminal
    See: "Uvicorn running on..."

[ ] Correct URL?
    Check: http://localhost:8000/docs
    Not: http://localhost:8501 (that's Streamlit)

[ ] Endpoint exists?
    Check: @app.post("/endpoint") decorator
    Verify: Function name matches

[ ] Port correct?
    Check: Server on port 8000
    Check: Request hitting port 8000

[ ] Firewall blocking?
    Windows Firewall blocks ports sometimes
    Check: Windows Defender Firewall
```

**Debug Code:**
```python
import requests

try:
    response = requests.get("http://localhost:8000/docs")
    print(f"✓ Server responding: {response.status_code}")
except requests.exceptions.ConnectionError:
    print("✗ Server not running on localhost:8000")
except Exception as e:
    print(f"✗ Error: {e}")
```

---

## 🎯 STREAMLIT UI NOT LOADING

### Issue: "Streamlit page blank or stuck"

**Checklist:**
```
[ ] Streamlit running?
    Check: streamlit run jwellery_front.py
    See: "You can now view your Streamlit app in your browser"

[ ] Correct port?
    Streamlit default: http://localhost:8501
    Not: http://localhost:8000

[ ] API endpoint accessible?
    Check: FastAPI running on 8000
    Streamlit tries to call it

[ ] API_BASE_URL correct?
    Check jwellery_front.py:
    API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")
    Verify: Points to running FastAPI

[ ] Browser cache?
    Hard refresh: Ctrl+Shift+R (Windows) or Cmd+Shift+R (Mac)
    Clear cookies and cache

[ ] Check browser console?
    Press F12 → Console tab
    Look for JavaScript errors
```

**Debug Code:**
```python
# In jwellery_front.py, add logging
import streamlit as st

st.write(f"API_BASE_URL: {API_BASE_URL}")

# Test API connection
try:
    response = requests.get(f"{API_BASE_URL}/docs")
    st.success("✓ API server is running")
except:
    st.error("✗ Cannot connect to API server")
```

---

## 🐛 SYSTEMATIC DEBUGGING APPROACH

When stuck, follow this order:

### Level 1: Check Basics
```
1. [ ] Code syntax valid? (python -m py_compile file.py)
2. [ ] All imports work? (python -c "import module")
3. [ ] Configuration correct? (check .env file)
4. [ ] Services running? (check all servers/databases)
5. [ ] Permissions correct? (file read/write access)
```

### Level 2: Test Components
```
1. [ ] Test API endpoints directly (use /docs)
2. [ ] Test database connections separately
3. [ ] Test external API calls (Gemini, GPT, S3)
4. [ ] Test file I/O operations
5. [ ] Test with minimal input (single image, one API call)
```

### Level 3: Add Logging
```python
# Add to your code:
import logging

logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

# Log before/after each operation
logger.debug(f"Starting operation with input: {input_data}")
logger.info(f"API call response: {response}")
logger.error(f"Error occurred: {error}")

# Check logs
# Linux/Mac: tail -f debug.log
# Windows: Get-Content debug.log -Wait
```

### Level 4: Binary Search
```
1. [ ] Comment out half the code
2. [ ] Does it work?
3. [ ] If yes: Bug in disabled half
4. [ ] If no: Bug in enabled half
5. [ ] Repeat with smaller sections
```

### Level 5: Compare with Working Version
```
1. [ ] Do you have git version control?
2. [ ] Check git log: git log --oneline
3. [ ] Find last working commit: git log --all --full-history
4. [ ] Compare files: git diff
5. [ ] Revert if needed: git revert [commit]
```

---

## 📋 CREATING DEBUG REPORT

When asking for help, include:

```markdown
## Debug Report

### Environment
- OS: Windows/Linux/Mac
- Python version: 3.8/3.9/3.10
- FastAPI running? Yes/No
- Milvus running? Yes/No

### Error Message
```
[Paste exact error here]
```

### Steps to Reproduce
1. [Step 1]
2. [Step 2]
3. [Step 3]

### What I've Already Tried
- [ ] Item 1
- [ ] Item 2
- [ ] Item 3

### Relevant Code
```python
[Paste relevant code snippets]
```

### Logs
```
[Paste error logs]
```
```

---

## 🎯 QUICK REFERENCE CHECKLIST

**First 5 things to check when something breaks:**

```
☐ 1. Is the error message clear or generic?
      If generic → Add try/except with detailed error logging

☐ 2. Is the error from my code or external service?
      External → Check credentials and API status
      My code → Add print statements to find exact line

☐ 3. Was this working before I changed something?
      Yes → Use git diff to see exact changes
      No → Environment/config issue

☐ 4. Is the issue consistent or intermittent?
      Consistent → Likely code logic issue
      Intermittent → Network, API rate limit, or timing issue

☐ 5. Does it work in isolation vs full system?
      Isolation works → Integration issue
      Isolation fails → Component issue
```

---

## 🚀 PERFORMANCE DEBUGGING

### Issue: "Application is slow"

**Checklist:**
```
[ ] Which operation is slow?
    Add timing: start = time.time(); ...; print(time.time() - start)

[ ] API calls taking too long?
    Check: Google Gemini, GPT-4, S3 latency
    Add: time.sleep() between requests

[ ] Database queries slow?
    Check: Milvus search_similar() time
    Verify: Index created properly

[ ] Image processing slow?
    Check: CLIP embedding time
    Check: Image resizing/formatting time

[ ] Memory usage high?
    Check: python -m memory_profiler
    Look for: Memory leaks in loops

[ ] Processing many images?
    Optimize: Process in parallel with asyncio
    Batch operations together
```

---

## ✅ VERIFICATION CHECKLIST

After fixing an issue, verify:

```
☐ 1. Can you reproduce the issue? (Should NOT reproduce)
☐ 2. Does the fix apply to all cases? (Test edge cases)
☐ 3. Did you break anything else? (Run other endpoints)
☐ 4. Is the error message now clear? (If it errors again)
☐ 5. Did you add logging for future debugging? (Yes/No)
☐ 6. Can someone else understand the fix? (Is code readable)
☐ 7. Should this be prevented from future occurring? (Add validation)
```

---

**Remember: Most bugs are in these areas (in order of frequency):**
1. Missing/wrong API keys or credentials (40%)
2. External service not running (30%)
3. Data format/parsing issues (15%)
4. Logic errors in code (10%)
5. Environment configuration (5%)

Good luck debugging! 🚀
