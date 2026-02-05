# MASTER SYSTEM DIAGRAM
## Complete Architecture & Data Flow Reference

---

# THE BIG PICTURE (Print This!)

```
+=====================================================================================+
|                         JEWELRY AI CATALOG SYSTEM - COMPLETE FLOW                   |
+=====================================================================================+

                                    +------------------+
                                    |      USER        |
                                    +--------+---------+
                                             |
               +-----------------------------+-----------------------------+
               |                             |                             |
               v                             v                             v
    +--------------------+       +--------------------+       +--------------------+
    |   STREAMLIT UI     |       |   STREAMLIT UI     |       |   DIRECT API       |
    |   frontend.py      |       | jwellery_front.py  |       |   CALLS            |
    | (Bill Extractor)   |       | (Description Gen)  |       |                    |
    +--------+-----------+       +--------+-----------+       +--------+-----------+
             |                            |                            |
             +----------------------------+----------------------------+
                                          |
        +---------------------------------+---------------------------------+
        |                |                |                |                |
        v                v                v                v                v
+-------------+  +-------------+  +---------------+  +-------------+  +-------------+
|/generate_   |  |/catalog-ai  |  |/image_search  |  |/semantic_   |  |/process-    |
|caption      |  |(Bulk+Excel) |  |(Similarity)   |  |filter_      |  |image-and-   |
|(Single Img) |  |             |  |               |  |jewelry      |  |prompt       |
+------+------+  +------+------+  +-------+-------+  +------+------+  |(Bill->CSV)  |
       |                |                 |                 |         +------+------+
       +----------------+-----------------+-----------------+-----------------+
                                          |
                                          v
+=============================================================================================+
|                                    FASTAPI SERVER                                           |
|                                    (img_to_csv.py)                                          |
|                                                                                             |
|   +-----------------------------------------------------------------------------------+     |
|   |                           STEP 1: DUPLICATE DETECTION                             |     |
|   |                                                                                   |     |
|   |   +-------------+     +------------------+     +------------------+               |     |
|   |   |   Incoming  | --> |   CLIP Model     | --> |  512-dim Vector  |               |     |
|   |   |    Image    |     | (clip-vit-base)  |     |   (Embedding)    |               |     |
|   |   +-------------+     +------------------+     +--------+---------+               |     |
|   |                                                         |                         |     |
|   |                                                         v                         |     |
|   |                                                +------------------+               |     |
|   |                                                |   MILVUS/ZILLIZ  |               |     |
|   |                                                |   Vector Search  |               |     |
|   |                                                | (L2 Distance)    |               |     |
|   |                                                +--------+---------+               |     |
|   |                                                         |                         |     |
|   |                              +-------------+------------+------------+            |     |
|   |                              |             |                         |            |     |
|   |                              v             v                         v            |     |
|   |                     +---------------+ +---------------+     +---------------+     |     |
|   |                     | Distance < 5  | | Distance 5-95 |     | Distance > 95 |     |     |
|   |                     | = DUPLICATE!  | | = Similar     |     | = NEW IMAGE   |     |     |
|   |                     | Return exist. | | (show matches)|     | Continue...   |     |     |
|   |                     +---------------+ +---------------+     +-------+-------+     |     |
|   |                                                                     |             |     |
|   +---------------------------------------------------------------------+-------------+     |
|                                                                         |                   |
|   +---------------------------------------------------------------------v-------------+     |
|   |                       STEP 2: S3 UPLOAD & INDEXING (DUAL BUCKET)                 |     |
|   |                                                                                   |     |
|   |   +------------------+     +------------------+     +------------------+          |     |
|   |   |  Generate Name   | --> |   Upload to S3   | --> |   Index Image    |          |     |
|   |   | timestamp_uuid_  |     | (alya OR blysk)  |     |                  |          |     |
|   |   | original.jpg     |     +--------+---------+     +--------+---------+          |     |
|   |   +------------------+              |                        |                    |     |
|   |                                     |                        |                    |     |
|   |                          +----------+----------+             v                    |     |
|   |                          |                     |    +------------------+          |     |
|   |                          v                     v    | indexed_hashes   |          |     |
|   |                 +----------------+   +----------------+  .json         |          |     |
|   |                 |  S3: alyaimg   |   |  S3: blyskimg  | {sha256: s3_key}|          |     |
|   |                 | +-----------+  |   | +-----------+  | +------------------+          |     |
|   |                 | | images/   |  |   | | images/   |  |          +                    |     |
|   |                 | | gen_imgs/ |  |   | | gen_imgs/ |  | +------------------+          |     |
|   |                 | | excel/    |  |   | | excel/    |  | | int_hash_to_path |          |     |
|   |                 | +-----------+  |   | +-----------+  | |    .json         |          |     |
|   |                 +----------------+   +----------------+ | {int_id: s3_url} |          |     |
|   |                                                     +------------------+          |     |
|   +-----------------------------------------------------------------------------------+     |
|                                                                                             |
|   +-----------------------------------------------------------------------------------+     |
|   |                           STEP 3: AI ANALYSIS                                     |     |
|   |                                                                                   |     |
|   |                    +----------------------------------------+                     |     |
|   |                    |            GEMINI 2.0 FLASH            |                     |     |
|   |                    |         (Primary AI Analyzer)          |                     |     |
|   |                    +----------------------------------------+                     |     |
|   |                                       |                                           |     |
|   |            +------------+-------------+-------------+------------+                |     |
|   |            |            |             |             |            |                |     |
|   |            v            v             v             v            v                |     |
|   |       +---------+ +---------+   +---------+   +---------+  +---------+            |     |
|   |       | Quality | |  Name   |   |  Color  |   |Material |  |Gemstones|            |     |
|   |       |  A/B/C  | |"Diamond |   | "Gold"  |   |"Sterling|  |"Emerald,|            |     |
|   |       |         | | Studs"  |   |         |   | Silver" |  | Ruby"   |            |     |
|   |       +---------+ +---------+   +---------+   +---------+  +---------+            |     |
|   |                                                                                   |     |
|   |       Questions Asked (from prompts.py):                                          |     |
|   |       - What type of jewelry?  - What is the primary color?                       |     |
|   |       - What material is it?   - Any gemstones visible?                           |     |
|   |       - What style/design?     - What occasion is it for?                         |     |
|   |       - Estimated quality?     - Target demographic?                              |     |
|   |       - (20+ questions per marketplace)                                           |     |
|   +-----------------------------------------------------------------------------------+     |
|                                                                                             |
|   +-----------------------------------------------------------------------------------+     |
|   |                           STEP 4: DESCRIPTION GENERATION                          |     |
|   |                                                                                   |     |
|   |   +------------------+            +------------------+                            |     |
|   |   | Check JSON Cache | ---------> | data_store.json  |                            |     |
|   |   | (by SKU)         |            | {sku: {title,    |                            |     |
|   |   +--------+---------+            |  description,    |                            |     |
|   |            |                      |  bullet_points}} |                            |     |
|   |            |                      +------------------+                            |     |
|   |            |                                                                      |     |
|   |      +-----+-----+                                                                |     |
|   |      |           |                                                                |     |
|   |  [CACHE HIT]  [CACHE MISS]                                                        |     |
|   |      |           |                                                                |     |
|   |      v           v                                                                |     |
|   |  +----------+ +------------------+     +------------------+                       |     |
|   |  | Use      | |   GPT-4 VISION   | --> | Marketing Copy:  |                       |     |
|   |  | Existing | | (Description Gen)|     | - Title          |                       |     |
|   |  | Data     | +------------------+     | - 5 Bullet Points|                       |     |
|   |  +----------+                          | - Long Desc      |                       |     |
|   |                                        +------------------+                       |     |
|   +-----------------------------------------------------------------------------------+     |
|                                                                                             |
+=============================================================================================+
                                          |
                                          v
+=============================================================================================+
|                                    OUTPUT GENERATION                                        |
+=============================================================================================+
|                                                                                             |
|   For /generate_caption:                    For /catalog-ai:                               |
|   +---------------------------+             +------------------------------------------+    |
|   | JSON Response:            |             |                                          |    |
|   | {                         |             |   MARKETPLACE ROUTING                    |    |
|   |   "s3_url": "https://...",|             |                                          |    |
|   |   "name": "Diamond Studs",|             |   +-----------+  +-----------+  +------+ |    |
|   |   "description": "...",   |             |   |  AMAZON   |  | FLIPKART  |  |MEESHO| |    |
|   |   "attributes": "...",    |             |   +-----------+  +-----------+  +------+ |    |
|   |   "color": "gold",        |             |        |              |            |     |    |
|   |   "quality": "A"          |             |        v              v            v     |    |
|   | }                         |             |   +-----------------------------------------+ |
|   +---------------------------+             |   |         EXCEL GENERATION               | |
|                                             |   |                                         | |
|                                             |   |  1. Load template from S3               | |
|                                             |   |  2. Map AI data to columns              | |
|                                             |   |     (excel_fields.py)                   | |
|                                             |   |  3. Add fixed values                    | |
|                                             |   |     (brand, shipping, etc.)             | |
|                                             |   |  4. Write product rows                  | |
|                                             |   |  5. Upload to S3                        | |
|                                             |   |  6. Return download URL                 | |
|                                             |   +-----------------------------------------+ |
|                                             +------------------------------------------+    |
|                                                                                             |
+=============================================================================================+


+=====================================================================================+
|                              EXTERNAL SERVICES MAP                                  |
+=====================================================================================+

    +------------------+       +------------------+       +------------------+
    |   GOOGLE CLOUD   |       |       AWS        |       |     ZILLIZ       |
    +------------------+       +------------------+       +------------------+
    |                  |       |                  |       |                  |
    | +-------------+  |       | +-------------+  |       | +-------------+  |
    | | Gemini 2.0  |  |       | |   S3        |  |       | |   Milvus    |  |
    | | Flash       |  |       | | (DUAL)      |  |       | |   Cloud     |  |
    | +-------------+  |       | | - alyaimg   |  |       | +-------------+  |
    |                  |       | | - blyskimg  |  |       |                  |
    +------------------+       | +-------------+  |       +------------------+
            |                  +------------------+              |
            |   Image Analysis        |                         |   Vector Search
            |   20+ questions         |   Image Storage         |   512-dim CLIP
            |                         |   Excel Files           |   HNSW Index
            |                         |   Generated Images      |
    +-------+-------------------------+-------------------------+---------+
    |                                                                     |
    |                          YOUR SERVER                                |
    |                        (img_to_csv.py)                              |
    |                                                                     |
    +-------+-------------------------+-------------------------+---------+
            |                         |                         |
            |                         |                         |
    +------------------+       +------------------+       +------------------+
    |     OPENAI       |       |    LOCAL FILES   |       |    CHROMADB      |
    +------------------+       +------------------+       +------------------+
    |                  |       |                  |       |                  |
    | +-------------+  |       | +-------------+  |       | +-------------+  |
    | | GPT-4.1     |  |       | | JSON Files  |  |       | | Semantic    |  |
    | | Vision      |  |       | | - data_store|  |       | | Search      |  |
    | +-------------+  |       | | - indexed_  |  |       | | (Jewelry    |  |
    | | GPT-4       |  |       | |   hashes    |  |       | |  Filtering) |  |
    | | Image Gen   |  |       | | - int_hash  |  |       | +-------------+  |
    | +-------------+  |       | +-------------+  |       |                  |
    +------------------+       +------------------+       +------------------+
       |         |
       |         |
  Descriptions  Image
  & Marketing   Variations
  Copy          Generation


+=====================================================================================+
|                              WEB SCRAPERS (PLAYWRIGHT)                              |
+=====================================================================================+

    +------------------+       +------------------+       +------------------+
    |   scraper.py     |       |  flp_scraper.py  |       |  myn_scraper.py  |
    |    (AMAZON)      |       |   (FLIPKART)     |       |    (MYNTRA)      |
    +------------------+       +------------------+       +------------------+
    |                  |       |                  |       |                  |
    | - ASIN           |       | - Item IDs       |       | - Product ID     |
    | - Brand          |       | - Brand          |       | - Brand          |
    | - Price          |       | - Title          |       | - Title          |
    | - Rating         |       | - Price          |       | - Price          |
    | - Images/Videos  |       | - HTML Snapshot  |       | - Rating         |
    | - Excel Export   |       |                  |       | - Images         |
    +------------------+       +------------------+       +------------------+
           |                          |                          |
           +---------------+----------+----------+---------------+
                           |                     |
                           v                     v
                    +-------------+       +--------------+
                    | Async       |       | pandas       |
                    | Playwright  |       | Excel Export |
                    +-------------+       +--------------+


+=====================================================================================+
|                              FILE RESPONSIBILITIES                                  |
+=====================================================================================+

+------------------------------------------------------------------------------------+
|  FILE                    |  RESPONSIBILITY                    |  TALKS TO          |
+------------------------------------------------------------------------------------+
|  img_to_csv.py           |  Main API server, all endpoints    |  Everything        |
|  (2700+ lines)           |  Route handling, coordination      |                    |
|                          |  100+ jewelry style descriptions   |                    |
|                          |  Dual S3 bucket support            |                    |
+------------------------------------------------------------------------------------+
|  image_search_engine.py  |  Duplicate detection               |  CLIP, Milvus,     |
|  (600+ lines)            |  Image indexing, orphan cleanup    |  S3, JSON files    |
|                          |  Batch embedding generation        |                    |
+------------------------------------------------------------------------------------+
|  prompts.py              |  AI question templates             |  Gemini (via       |
|  (112 KB)                |  Marketplace-specific prompts      |  img_to_csv.py)    |
|                          |  Flipkart extraction with options  |                    |
+------------------------------------------------------------------------------------+
|  prompts2.py             |  Image generation prompts          |  GPT-4 Image Gen   |
|  (18 KB)                 |  Advanced prompt variations        |                    |
+------------------------------------------------------------------------------------+
|  excel_fields.py         |  Column mappings per marketplace   |  Excel generation  |
|  (22 KB)                 |  Flipkart (100+ fields)            |  in utils.py       |
|                          |  Amazon bracelets/necklaces        |                    |
|                          |  Fixed values (brand, shipping)    |                    |
+------------------------------------------------------------------------------------+
|  utils.py                |  Gemini API calls                  |  Google AI,        |
|  (21 KB)                 |  Excel writing functions           |  S3, openpyxl      |
|                          |  Image processing helpers          |                    |
+------------------------------------------------------------------------------------+
|  utils2.py               |  API key rotation & failover       |  Google AI         |
|  (234 lines)             |  Multi-image Gemini input          |  (multiple keys),  |
|                          |  GPT-4 image generation            |  OpenAI            |
|                          |  Image resizing/preprocessing      |                    |
+------------------------------------------------------------------------------------+
|  openai_utils.py         |  GPT-4.1 Vision calls              |  OpenAI API        |
|  (56 lines)              |  Base64 image encoding             |                    |
|                          |  Structured JSON responses         |                    |
+------------------------------------------------------------------------------------+
|  json_storage.py         |  SKU data caching                  |  data_store.json   |
|  (108 lines)             |  Compression/expansion of nested   |                    |
|                          |  structures (bullet points)        |                    |
+------------------------------------------------------------------------------------+
|  frontend.py             |  Streamlit UI                      |  FastAPI backend   |
|  (117 lines)             |  Bill image upload -> CSV          |  (Render hosted)   |
+------------------------------------------------------------------------------------+
|  jwellery_front.py       |  Streamlit UI                      |  FastAPI backend   |
|  (65 lines)              |  Jewelry description generation    |                    |
+------------------------------------------------------------------------------------+
|  scraper.py              |  Amazon product scraper            |  Playwright,       |
|  (200+ lines)            |  ASIN, brand, price, images        |  pandas, Excel     |
+------------------------------------------------------------------------------------+
|  flp_scraper.py          |  Flipkart product scraper          |  Playwright        |
|  (80+ lines)             |  Item IDs, HTML snapshots          |                    |
+------------------------------------------------------------------------------------+
|  myn_scraper.py          |  Myntra product scraper            |  Playwright        |
|  (150+ lines)            |  Product details, async scraping   |                    |
+------------------------------------------------------------------------------------+


+=====================================================================================+
|                           ENDPOINT QUICK REFERENCE                                  |
+=====================================================================================+

  ENDPOINT                    INPUT                         OUTPUT
  ────────────────────────────────────────────────────────────────────────────────
  /generate_caption           Image file                    {s3_url, name, desc,
                                                             attributes, color}
  ────────────────────────────────────────────────────────────────────────────────
  /catalog-ai                 URLs + SKUs + marketplace     {results[], excel_url}
  ────────────────────────────────────────────────────────────────────────────────
  /catalog_ai_variations      Images + variation type       Excel with parent-child
  ────────────────────────────────────────────────────────────────────────────────
  /image-search               Query image                   {similar_images, scores}
  /image_similarity_search    Image URL                     (alias)
  ────────────────────────────────────────────────────────────────────────────────
  /upload-and-index           Image file                    {s3_url, indexed: true}
  ────────────────────────────────────────────────────────────────────────────────
  /delete_image               Comma-separated URLs          Success/error
  ────────────────────────────────────────────────────────────────────────────────
  /semantic_filter_jewelry    color, category, audience,    {jewelry_ids,
                              price_range, date_range       collection_title, desc}
  ────────────────────────────────────────────────────────────────────────────────
  /process-image-and-prompt   Bill/invoice image + prompt   {table, extracted_data}
  ────────────────────────────────────────────────────────────────────────────────
  /further_req                Previous data + refinement    {filtered_csv}
  ────────────────────────────────────────────────────────────────────────────────
  /generate-images            Product desc + type           {image_urls, zip}
  ────────────────────────────────────────────────────────────────────────────────


+=====================================================================================+
|                           DUPLICATE DETECTION THRESHOLDS                            |
+=====================================================================================+

                          CLIP L2 DISTANCE SCALE

  0 ──────────── 5 ──────────── 25 ──────────── 95 ──────────── 150+
  │              │               │               │               │
  │   DUPLICATE  │    SIMILAR    │   DIFFERENT   │   UNRELATED   │
  │   (same img) │   (same type) │   (diff type) │   (car vs     │
  │              │               │               │    jewelry)   │
  │              │               │               │               │
  └──────────────┴───────────────┴───────────────┴───────────────┘

  Threshold = 95
  - Distance < 95  →  Duplicate detected, return existing image
  - Distance >= 95 →  New image, proceed with upload


+=====================================================================================+
|                           STARTUP SEQUENCE                                          |
+=====================================================================================+

  Server Start (uvicorn img_to_csv:app)
         │
         v
  +------------------+
  | Load .env        |
  | - API keys       |
  | - S3 credentials |
  | - Milvus config  |
  +--------+---------+
           │
           v
  +------------------+
  | Check Flags      |
  +--------+---------+
           │
     +-----+-----+
     │           │
     v           v
  CLEAN_ORPHANS  REINDEX_ON_STARTUP
  =true?         =true?
     │              │
     v              v
  Delete stale   Re-index ALL
  Milvus entries S3 images
  (~1 min)       (~3 min for 1255 images)
     │              │
     +------+-------+
            │
            v
  +------------------+
  | Server Ready     |
  | Listening on     |
  | port 8000        |
  +------------------+


+=====================================================================================+
|                           DATA FLOW: COMPLETE JOURNEY                               |
+=====================================================================================+

  USER UPLOADS IMAGE
         │
         │ POST /generate_caption
         v
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 1. RECEIVE IMAGE                                                            │
  │    - FastAPI receives multipart form data                                   │
  │    - Image saved to memory                                                  │
  └─────────────────────────────────────────────────────────────────────────────┘
         │
         v
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 2. DUPLICATE CHECK                                                          │
  │    - Load CLIP model (openai/clip-vit-base-patch32)                        │
  │    - Convert image to 512-dimensional vector                               │
  │    - Query Milvus: "Find vectors within distance 95"                       │
  │    - If match found → STOP, return existing S3 URL                         │
  │    - If no match → Continue                                                │
  └─────────────────────────────────────────────────────────────────────────────┘
         │
         v
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 3. S3 UPLOAD (DUAL BUCKET SUPPORT)                                          │
  │    - Generate unique filename: {timestamp}_{uuid}_{original_name}.jpg      │
  │    - Select bucket: "alya" → alyaimg, "blysk" → blyskimg                   │
  │    - Upload to selected S3 bucket                                          │
  │    - Get public URL: https://{bucket}.s3.amazonaws.com/{key}               │
  └─────────────────────────────────────────────────────────────────────────────┘
         │
         v
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 4. INDEX IMAGE                                                              │
  │    - Compute SHA256 hash of image bytes                                    │
  │    - Create integer ID from hash (for Milvus)                              │
  │    - Insert vector + ID into Milvus collection                             │
  │    - Update indexed_hashes.json: {hash: s3_key}                            │
  │    - Update int_hash_to_path.json: {int_id: s3_url}                        │
  └─────────────────────────────────────────────────────────────────────────────┘
         │
         v
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 5. AI ANALYSIS (Gemini 2.0)                                                 │
  │    - Load prompt from prompts.py (marketplace + product type)              │
  │    - Send image + prompt to Gemini API                                     │
  │    - Receive structured JSON response:                                     │
  │      {                                                                     │
  │        "quality": "A",                                                     │
  │        "name": "Elegant Diamond Studs",                                    │
  │        "description": "Premium quality diamond earrings...",               │
  │        "attributes": "diamonds, gold, studs, formal",                      │
  │        "color": "gold"                                                     │
  │      }                                                                     │
  └─────────────────────────────────────────────────────────────────────────────┘
         │
         v
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 6. CACHE RESULT                                                             │
  │    - Store in data_store.json keyed by SKU (if provided)                   │
  │    - Prevents duplicate AI calls for same product                          │
  └─────────────────────────────────────────────────────────────────────────────┘
         │
         v
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 7. RETURN RESPONSE                                                          │
  │    {                                                                       │
  │      "s3_url": "https://alyaimg.s3.amazonaws.com/...",                     │
  │      "name": "Elegant Diamond Studs",                                      │
  │      "description": "Premium quality diamond earrings...",                 │
  │      "attributes": "diamonds, gold, studs, formal",                        │
  │      "color": "gold",                                                      │
  │      "quality": "A"                                                        │
  │    }                                                                       │
  └─────────────────────────────────────────────────────────────────────────────┘


+=====================================================================================+
|                     CATALOG GENERATION: BULK FLOW                                   |
+=====================================================================================+

  USER SUBMITS CATALOG REQUEST
         │
         │ POST /catalog-ai
         │ {
         │   "image_url": "url1,url2,url3",
         │   "sku": "SKU1,SKU2,SKU3",
         │   "marketplace_type": "ama_ear",  // Amazon Earrings
         │   "product_type": "earrings"
         │ }
         v
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 1. PARSE INPUT                                                              │
  │    - Split URLs by comma                                                   │
  │    - Split SKUs by comma                                                   │
  │    - Validate marketplace type                                             │
  └─────────────────────────────────────────────────────────────────────────────┘
         │
         v
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 2. LOAD MARKETPLACE CONFIG                                                  │
  │    - Get prompt from prompts.py (e.g., ama_ear_prompt)                     │
  │    - Get target_fields from excel_fields.py                                │
  │    - Get fixed_values from excel_fields.py                                 │
  │    - Load Excel template from S3                                           │
  └─────────────────────────────────────────────────────────────────────────────┘
         │
         v
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 3. FOR EACH IMAGE:                                                          │
  │                                                                            │
  │    ┌──────────────────────────────────────────────────────────────────┐    │
  │    │ a. Download image from URL                                       │    │
  │    │ b. Check cache: fetch_data_for_sku(sku)                         │    │
  │    │    - If exists: use cached title, bullets, description          │    │
  │    │    - If not: call GPT-4 Vision for marketing copy               │    │
  │    │ c. Call Gemini with marketplace prompt                          │    │
  │    │ d. Merge all data:                                              │    │
  │    │    - Gemini attributes (color, material, gemstones)             │    │
  │    │    - GPT-4 descriptions (title, bullets)                        │    │
  │    │    - Fixed values (brand, shipping, dimensions)                 │    │
  │    │ e. Map to Excel columns                                         │    │
  │    └──────────────────────────────────────────────────────────────────┘    │
  └─────────────────────────────────────────────────────────────────────────────┘
         │
         v
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 4. GENERATE EXCEL                                                           │
  │    - Open template workbook                                                │
  │    - For each product:                                                     │
  │      - Find correct row                                                    │
  │      - Map each field to correct column                                    │
  │      - Write values                                                        │
  │    - Save to temp file                                                     │
  │    - Upload to S3: excel_files/{timestamp}_{marketplace}.xlsx             │
  └─────────────────────────────────────────────────────────────────────────────┘
         │
         v
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 5. RETURN RESPONSE                                                          │
  │    {                                                                       │
  │      "results": [                                                          │
  │        {"sku": "SKU1", "name": "...", "status": "success"},               │
  │        {"sku": "SKU2", "name": "...", "status": "success"},               │
  │        {"sku": "SKU3", "name": "...", "status": "success"}                │
  │      ],                                                                    │
  │      "excel_file_url": "https://alyaimg.s3.../excel_files/catalog.xlsx"   │
  │    }                                                                       │
  └─────────────────────────────────────────────────────────────────────────────┘


+=====================================================================================+
|                     SEMANTIC JEWELRY FILTERING: CHROMADB FLOW                       |
+=====================================================================================+

  USER SUBMITS SEMANTIC FILTER REQUEST
         │
         │ POST /semantic_filter_jewelry
         │ {
         │   "color": "gold",
         │   "category": "earrings",
         │   "target_audience": "women",
         │   "price_range": "1000-5000",
         │   "date_range": "2024-01-01 to 2024-12-31"
         │ }
         v
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 1. FETCH PRODUCTS FROM EXTERNAL API                                         │
  │    - Query: staging.blyskjewels.com/api/get-all-products                   │
  │    - Get product catalog with metadata                                     │
  └─────────────────────────────────────────────────────────────────────────────┘
         │
         v
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 2. EMBED PRODUCTS IN CHROMADB                                               │
  │    - Use Gemini embed_content() for each product                           │
  │    - Store embeddings + metadata in ChromaDB                               │
  │    - Metadata: color, category, price, date, style descriptions            │
  └─────────────────────────────────────────────────────────────────────────────┘
         │
         v
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 3. CONVERT USER QUERY TO EMBEDDING                                          │
  │    - Combine user preferences into query string                            │
  │    - Match against 100+ jewelry style descriptions                         │
  │    - Styles: Elegance, Heritage, Minimalist, Bohemian, etc.               │
  └─────────────────────────────────────────────────────────────────────────────┘
         │
         v
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 4. SEMANTIC SEARCH IN CHROMADB                                              │
  │    - Vector similarity search                                              │
  │    - Filter by metadata constraints (price, date, category)               │
  │    - Return top-k matching jewelry IDs                                     │
  └─────────────────────────────────────────────────────────────────────────────┘
         │
         v
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │ 5. RETURN RESPONSE                                                          │
  │    {                                                                       │
  │      "jewelry_ids": [123, 456, 789, ...],                                  │
  │      "collection_title": "Elegant Gold Earrings Collection",               │
  │      "collection_description": "Curated selection of gold earrings..."    │
  │    }                                                                       │
  └─────────────────────────────────────────────────────────────────────────────┘


+=====================================================================================+
|                           MARKETPLACE FIELD DETAILS                                 |
+=====================================================================================+

  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                              AMAZON EARRINGS                                     │
  ├──────────────────────────────────────────────────────────────────────────────────┤
  │  TOTAL EXCEL TEMPLATE COLUMNS:  267 columns (full Amazon flat file)             │
  │  ──────────────────────────────────────────────────────────────────────────────  │
  │  AI-GENERATED (Gemini):         44 fields                                       │
  │    • stone_color, style_name, stones_color, item_type_name                      │
  │    • occasion_type1-5, clasp_type, color_map, material_type                     │
  │    • stone_shape, stone_clarity, collection_name, setting_type                  │
  │    • stones_number_of_stones, theme, back_finding, gem_type, etc.               │
  │  ──────────────────────────────────────────────────────────────────────────────  │
  │  OPENAI GPT-4.1 GENERATED:      12 fields                                       │
  │    • item_name (title, max 160 chars, SEO optimized)                            │
  │    • bullet_point1 through bullet_point10 (10 bullet points)                    │
  │    • product_description                                                        │
  │  ──────────────────────────────────────────────────────────────────────────────  │
  │  FIXED VALUES:                  34 fields                                       │
  │    • feed_product_type, brand_name, manufacturer, target_gender                 │
  │    • country_of_origin, HSN (71171900), tax_code, currency (INR)                │
  │    • condition_type, fulfillment_latency, etc.                                  │
  └──────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                              AMAZON NECKLACE                                     │
  ├──────────────────────────────────────────────────────────────────────────────────┤
  │  AI-GENERATED (Gemini):         31 fields                                       │
  │  OPENAI GPT-4.1:                12 fields (title + 10 bullets + description)    │
  │  FIXED VALUES:                  31 fields                                       │
  └──────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                              AMAZON BRACELET                                     │
  ├──────────────────────────────────────────────────────────────────────────────────┤
  │  AI-GENERATED (Gemini):         37 fields                                       │
  │  OPENAI GPT-4.1:                12 fields (title + 10 bullets + description)    │
  │  FIXED VALUES:                  25 fields                                       │
  └──────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                              FLIPKART EARRINGS                                   │
  ├──────────────────────────────────────────────────────────────────────────────────┤
  │  AI-GENERATED (Gemini):         33 fields                                       │
  │    • Seller SKU ID, Model Number, Model Name, Type, Color, Base Material        │
  │    • Gemstone, Pearl Type, Collection, Occasion, Piercing Required              │
  │    • Number of Gemstones, Earring Back Type, Finish, Design, Metal Color        │
  │    • Pearl Shape/Grade/Color, Search Keywords, Key Features, Trend              │
  │    • Width(mm), Height(mm), Diameter(mm), Description                           │
  │  ──────────────────────────────────────────────────────────────────────────────  │
  │  OPENAI GPT-4.1:                Description only (Gemini handles most fields)   │
  │  ──────────────────────────────────────────────────────────────────────────────  │
  │  FIXED VALUES:                  27 fields                                       │
  │    • Listing Status, MRP, Stock, Dimensions, Brand (Alya Jewels)                │
  │    • HSN, Tax Code, Shipping provider, etc.                                     │
  └──────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                              FLIPKART NECKLACE / BRACELET                        │
  ├──────────────────────────────────────────────────────────────────────────────────┤
  │  NECKLACE:  AI = 35 fields, Fixed = 17 fields                                   │
  │  BRACELET:  AI = 29 fields, Fixed = 22 fields                                   │
  └──────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                              MEESHO (All Products)                               │
  ├──────────────────────────────────────────────────────────────────────────────────┤
  │  AI-GENERATED (Gemini):         11 fields                                       │
  │    • Product Name, Base Metal, Color, Occasion, Plating                         │
  │    • Sizing, Stone Type, Trend, Type, Product Description                       │
  │  ──────────────────────────────────────────────────────────────────────────────  │
  │  OPENAI GPT-4.1:                Title + Description                             │
  │  ──────────────────────────────────────────────────────────────────────────────  │
  │  FIXED VALUES:                  7-9 fields                                      │
  │    • variation (Free size), GST% (3), HSN ID, Country of origin                 │
  │    • Manufacturer Details, Packer Details, Brand Name                           │
  └──────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                              SHOPSY EARRINGS                                     │
  ├──────────────────────────────────────────────────────────────────────────────────┤
  │  Uses Flipkart earrings prompts and fixed values                                │
  │  Separate Excel template: earrings_shopsy.xlsx                                  │
  └──────────────────────────────────────────────────────────────────────────────────┘


+=====================================================================================+
|                           API USAGE SUMMARY                                         |
+=====================================================================================+

  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │  API                    │  PURPOSE                        │  MODEL / VERSION     │
  ├──────────────────────────────────────────────────────────────────────────────────┤
  │  Google Gemini          │  Jewelry attribute extraction   │  gemini-2.0-flash    │
  │                         │  (color, material, gemstones,   │                      │
  │                         │   occasion, style, dimensions)  │                      │
  ├──────────────────────────────────────────────────────────────────────────────────┤
  │  OpenAI GPT-4.1 Vision  │  Marketing copy generation:     │  gpt-4.1             │
  │                         │  • Title (160 chars, SEO)       │                      │
  │                         │  • 5-10 Bullet Points           │                      │
  │                         │  • Product Description          │                      │
  ├──────────────────────────────────────────────────────────────────────────────────┤
  │  OpenAI GPT Image       │  Product mockup generation      │  gpt-image-1         │
  ├──────────────────────────────────────────────────────────────────────────────────┤
  │  Gemini (Image Gen)     │  Image variations               │  gemini-2.5-flash-   │
  │                         │                                 │  image               │
  ├──────────────────────────────────────────────────────────────────────────────────┤
  │  CLIP                   │  512-dim image embeddings       │  clip-vit-base-      │
  │                         │  for duplicate detection        │  patch32             │
  ├──────────────────────────────────────────────────────────────────────────────────┤
  │  Gemini Embeddings      │  Semantic search embeddings     │  embed_content()     │
  │                         │  for ChromaDB filtering         │                      │
  └──────────────────────────────────────────────────────────────────────────────────┘


+=====================================================================================+
|                           DATA CACHING SYSTEM                                       |
+=====================================================================================+

  The system caches OpenAI-generated data (title, bullet points, description)
  by SKU to avoid redundant API calls:

  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  FILE: data_store.json                                                      │
  │  LOCATION: json_storage.py                                                  │
  │                                                                             │
  │  STRUCTURE:                                                                 │
  │  {                                                                          │
  │      "SKU-101": {                                                           │
  │          "title": "Elegant Gold Tone Diamond Studs...",                     │
  │          "bullet_points": {                                                 │
  │              "bullet_point1": "Premium quality...",                         │
  │              "bullet_point2": "Lightweight design...",                      │
  │              ...                                                            │
  │          },                                                                 │
  │          "description": "Beautiful handcrafted earrings..."                 │
  │      },                                                                     │
  │      "SKU-102": { ... }                                                     │
  │  }                                                                          │
  │                                                                             │
  │  FUNCTIONS:                                                                 │
  │  • store_data_for_sku(sku_id, data) - Saves/updates SKU data               │
  │  • fetch_data_for_sku(sku_id) - Retrieves cached data                      │
  │  • compress_bullet_points(data) - Compresses for storage                   │
  │  • expand_bullet_points(data) - Expands for use                            │
  └─────────────────────────────────────────────────────────────────────────────┘

  CACHE LOGIC (img_to_csv.py:987-1018):
  ┌─────────────────────────────────────────────────────────────────────────────┐
  │  1. Check if SKU exists in cache: fetch_data_for_sku(skuid)                │
  │  2. If NOT exists → Call GPT-4.1 for all fields                            │
  │  3. If EXISTS:                                                              │
  │     • Amazon: Use cached description, check title/bullets                  │
  │     • Flipkart/Shopsy: Use cached description only                         │
  │     • Meesho: Use cached description, check title                          │
  │  4. Store result back: store_data_for_sku(skuid, gpt_dict)                 │
  └─────────────────────────────────────────────────────────────────────────────┘


+=====================================================================================+
|                           CODE LOCATIONS REFERENCE                                  |
+=====================================================================================+

  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                          PROMPTS CONFIGURATION                                   │
  │                          FILE: prompts.py                                        │
  ├──────────────────────────────────────────────────────────────────────────────────┤
  │  GEMINI PROMPTS (Attribute Extraction):                                         │
  │  ──────────────────────────────────────────────────────────────────────────────  │
  │  prompt_questions_earrings_flipkart    Line 1      (33 fields)                  │
  │  prompt_description_earrings_flipkart  Line 145                                 │
  │  prompt_questions_necklace_flipkart    Line 191    (35 fields)                  │
  │  prompt_questions_bracelet_flipkart    Line 400    (29 fields)                  │
  │  ──────────────────────────────────────────────────────────────────────────────  │
  │  prompt_questions_earrings_amz         Line 544    (44 fields)                  │
  │  prompt_questions_necklace_amz         Line 698    (31 fields)                  │
  │  prompt_questions_bracelet_amz         Line 826    (37 fields)                  │
  │  ──────────────────────────────────────────────────────────────────────────────  │
  │  prompt_questions_earrings_meesho      Line 989    (11 fields)                  │
  │  prompt_questions_necklace_meesho      Line 1048   (11 fields)                  │
  │  prompt_questions_bracelet_meesho      Line 1106   (12 fields)                  │
  │  ──────────────────────────────────────────────────────────────────────────────  │
  │  prompt_questions_earrings_shopsy      Line 1595   (uses Flipkart)              │
  │                                                                                  │
  │  OPENAI GPT PROMPTS (Marketing Copy):                                           │
  │  ──────────────────────────────────────────────────────────────────────────────  │
  │  gpt_amz_earrings_prompt               Line 1239   (title + 5 bullets + desc)   │
  │  gpt_amz_necklace_prompt               Line 1289   (title + 5 bullets + desc)   │
  │  gpt_amz_bracelet_prompt               Line 1337   (title + 5 bullets + desc)   │
  │  ──────────────────────────────────────────────────────────────────────────────  │
  │  gpt_flipkart_earrings_prompt          Line 1390   (description only)           │
  │  gpt_flipkart_necklace_prompt          Line 1396   (description only)           │
  │  gpt_flipkart_bracelet_prompt          Line 1400   (description only)           │
  │  ──────────────────────────────────────────────────────────────────────────────  │
  │  gpt_meesho_earrings_prompt            Line 1407   (title + description)        │
  │  gpt_meesho_necklace_prompt            Line 1435   (title + description)        │
  │  gpt_meesho_bracelet_prompt            Line 1462   (title + description)        │
  │  ──────────────────────────────────────────────────────────────────────────────  │
  │  gpt_prompt_title_bp                   Line 1492   (fallback: title + bullets)  │
  │  gpt_prompt_title                      Line 1537   (fallback: title only)       │
  │  gpt_prompt_bp                         Line 1561   (fallback: bullets only)     │
  └──────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                          FIELD CONFIGURATIONS                                    │
  │                          FILE: excel_fields.py                                   │
  ├──────────────────────────────────────────────────────────────────────────────────┤
  │  FLIPKART:                                                                       │
  │  target_fields_earrings                Line 41     (33 AI fields)               │
  │  fixed_values_earrings                 Line 50     (27 fixed values)            │
  │  target_fields_necklace_flipkart       Line 207    (35 AI fields)               │
  │  fixed_values_necklace_flipkart        Line 187    (17 fixed values)            │
  │  target_fields_bracelet_flipkart       Line 180    (29 AI fields)               │
  │  fixed_values_bracelet_flipkart        Line 153    (22 fixed values)            │
  │  ──────────────────────────────────────────────────────────────────────────────  │
  │  AMAZON:                                                                         │
  │  target_fields_earrings_amz            Line 367    (44 AI fields)               │
  │  fixed_values_earrings_amz             Line 317    (34 fixed values)            │
  │  target_fields_necklace_amz            Line 278    (31 AI fields)               │
  │  fixed_values_necklace_amz             Line 244    (31 fixed values)            │
  │  target_fields_bracelet_amz            Line 112    (37 AI fields)               │
  │  fixed_values_bracelet_amz             Line 84     (25 fixed values)            │
  │  ──────────────────────────────────────────────────────────────────────────────  │
  │  MEESHO:                                                                         │
  │  target_fields_earrings_meesho         Line 422    (11 AI fields)               │
  │  fixed_values_earrings_meesho          Line 412    (7 fixed values)             │
  │  target_fields_necklace_meesho         Line 463    (11 AI fields)               │
  │  fixed_values_necklace_meesho          Line 450    (9 fixed values)             │
  │  target_fields_bracelet_meesho         Line 488    (12 AI fields)               │
  │  fixed_values_bracelet_meesho          Line 477    (8 fixed values)             │
  │  ──────────────────────────────────────────────────────────────────────────────  │
  │  Amazon Column Index Map               Line 365    (267 columns total)          │
  └──────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                          MAIN API ENDPOINTS                                      │
  │                          FILE: img_to_csv.py                                     │
  ├──────────────────────────────────────────────────────────────────────────────────┤
  │  startup_event()                       Line 81     (orphan cleanup, reindex)    │
  │  /semantic_filter_jewelry              Line 162    (ChromaDB search)            │
  │  /generate_caption                     Line 571    (single image analysis)      │
  │  /catalog-ai                           Line 864    (bulk catalog generation)    │
  │  /clear_excel_file                     Line 1459   (reset Excel templates)      │
  │  /generate-images                      Line 1680   (GPT image generation)       │
  │  ──────────────────────────────────────────────────────────────────────────────  │
  │  input_prompt_map (marketplace config) Line 899    (maps format to prompts)     │
  │  dims_prompt_map (dimensions)          Line 912    (dimension extraction)       │
  │  filename_map (Excel files)            Line 1065   (output Excel filenames)     │
  │  ──────────────────────────────────────────────────────────────────────────────  │
  │  SKU Cache Logic                       Line 986-1024 (fetch/store GPT data)     │
  │  Rate Limiting                         Line 1062   (sleep every 5 images)       │
  └──────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                          IMAGE SEARCH & INDEXING                                 │
  │                          FILE: image_search_engine.py                            │
  ├──────────────────────────────────────────────────────────────────────────────────┤
  │  get_image_paths_from_s3()             Line 56     (list S3 images)             │
  │  delete_images_from_s3()               Line 114    (batch delete from S3)       │
  │  hash_image_from_s3()                  Line 142    (SHA256 hash)                │
  │  load_image_from_s3()                  Line 147    (fetch image to PIL)         │
  │  index_images_from_s3()                Line 151    (bulk indexing)              │
  │  index_single_image_from_s3()          Line 244    (single image index)         │
  │  hash_to_int64()                       Line 324    (hash to Milvus ID)          │
  │  init_clip()                           Line 329    (load CLIP model)            │
  │  embed_image()                         Line 335    (512-dim embedding)          │
  │  init_milvus()                         Line 343    (connect to Zilliz)          │
  │  ──────────────────────────────────────────────────────────────────────────────  │
  │  INDEX_FILE = "indexed_hashes.json"    Line 39     (hash → S3 key map)          │
  │  INT_HASH_MAP_FILE = "int_hash_to_path.json" Line 40 (int_id → S3 URL)          │
  └──────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                          GEMINI & OPENAI UTILITIES                               │
  ├──────────────────────────────────────────────────────────────────────────────────┤
  │  FILE: utils.py                                                                  │
  │  get_gemini_responses()                Line 36     (temp=0.5, single image)     │
  │  get_gemini_responses_high_temp()      Line 17     (temp=1.5, creative)         │
  │  get_gemini_dims_responses()           Line 53     (temp=1.0, dimensions)       │
  │  get_gemini_responses_multi_image()    Line 82     (multiple images)            │
  │  input_image_setup()                   Line 137    (prepare image for API)      │
  │  write_to_excel_meesho()               Line 158    (Meesho Excel writer)        │
  │  write_to_excel_flipkart()             Line 235    (Flipkart Excel writer)      │
  │  write_to_excel_amz_xl()               Line 442    (Amazon Excel writer)        │
  │  ──────────────────────────────────────────────────────────────────────────────  │
  │  FILE: utils2.py                                                                 │
  │  API_KEYS (4 Gemini keys)              Line 30-35  (key rotation pool)          │
  │  _is_quota_exceeded()                  Line 49     (detect 429/quota errors)    │
  │  generate_with_failover()              Line 69     (auto-rotate on failure)     │
  │  get_gemini_responses() [image gen]    Line 139    (gemini-2.5-flash-image)     │
  │  generate_images_from_gpt()            Line 201    (gpt-image-1 model)          │
  │  resize_img()                          Line 130    (2000x2000 square)           │
  │  resize_img2()                         Line 120    (1080x1440 portrait)         │
  │  pil_to_base64()                       Line 42     (image encoding)             │
  │  ──────────────────────────────────────────────────────────────────────────────  │
  │  FILE: openai_utils.py                                                           │
  │  ask_gpt_with_image()                  Line 12     (GPT-4.1 Vision call)        │
  │                                                    (returns JSON response)      │
  │  ──────────────────────────────────────────────────────────────────────────────  │
  │  FILE: json_storage.py                                                           │
  │  JSON_PATH = "data_store.json"         Line 4      (cache file path)            │
  │  compress_bullet_points()              Line 6      (optimize storage)           │
  │  expand_bullet_points()                Line 26     (restore structure)          │
  │  store_data_for_sku()                  Line 48     (save to cache)              │
  │  fetch_data_for_sku()                  Line 71     (load from cache)            │
  └──────────────────────────────────────────────────────────────────────────────────┘

  ┌──────────────────────────────────────────────────────────────────────────────────┐
  │                          CONFIGURATION & INITIALIZATION                          │
  │                          FILE: img_to_csv.py                                     │
  ├──────────────────────────────────────────────────────────────────────────────────┤
  │  FastAPI app creation                  Line 40                                  │
  │  Gemini model init                     Line 55     (gemini-2.0-flash)           │
  │  BUCKET_MAP (dual S3)                  Line 61-64  (alya→alyaimg, blysk→blyskimg)│
  │  CLIP model init                       Line 69     (init_clip())                │
  │  Milvus/Zilliz connection              Line 70-74  (init_milvus())              │
  │  REINDEX_ON_STARTUP flag               Line 77                                  │
  │  CLEAN_ORPHANS_ON_STARTUP flag         Line 79                                  │
  └──────────────────────────────────────────────────────────────────────────────────┘


+=====================================================================================+
|                           TROUBLESHOOTING QUICK GUIDE                               |
+=====================================================================================+

  PROBLEM                           CAUSE                          SOLUTION
  ──────────────────────────────────────────────────────────────────────────────
  s3_url: "unknown"                 Orphan entry in Milvus         Run clean_orphan_entries()
                                    (ID exists but no JSON map)    or set CLEAN_ORPHANS_ON_STARTUP=true
  ──────────────────────────────────────────────────────────────────────────────
  Slow /image_similarity_search     Full re-index happening        Check REINDEX_ON_STARTUP=false
                                    on every request               (was fixed, should be false)
  ──────────────────────────────────────────────────────────────────────────────
  Windows emoji error               Unicode chars in logs          Emojis replaced with [OK], [ERROR]
  'charmap' codec error                                            in image_search_engine.py
  ──────────────────────────────────────────────────────────────────────────────
  Duplicate not detected            Threshold too high             Threshold is 95, should catch
                                    or different image format      95%+ similar images
  ──────────────────────────────────────────────────────────────────────────────
  API quota exceeded                Too many Gemini calls          utils2.py rotates API keys
                                                                   Add more keys to .env
  ──────────────────────────────────────────────────────────────────────────────
  Wrong S3 bucket used              Bucket name mismatch           Use "alya" or "blysk" as keys
                                    in config                      Maps to alyaimg/blyskimg
  ──────────────────────────────────────────────────────────────────────────────
  ChromaDB semantic search          Products not embedded          Check external API connection
  returns empty results             in ChromaDB                    staging.blyskjewels.com
  ──────────────────────────────────────────────────────────────────────────────
  Gemini failover not working       All API keys exhausted         Add more GOOGLE_API_KEY entries
                                    or _is_quota_exceeded()        Check utils2.py failover logic
                                    not detecting error
  ──────────────────────────────────────────────────────────────────────────────


+=====================================================================================+
|                           TECHNOLOGY STACK SUMMARY                                   |
+=====================================================================================+

  CATEGORY              TECHNOLOGY                    PURPOSE
  ──────────────────────────────────────────────────────────────────────────────
  Backend Framework     FastAPI                       Async web server
                        Uvicorn                       ASGI server
  ──────────────────────────────────────────────────────────────────────────────
  Frontend              Streamlit                     Interactive UI apps
  ──────────────────────────────────────────────────────────────────────────────
  AI/ML Services        Google Gemini 2.0 Flash       Image analysis, attribute extraction
                        OpenAI GPT-4.1 Vision         Description generation
                        OpenAI GPT-4 Image Gen        Product mockup generation
                        CLIP (ViT-base-patch32)       512-dim image embeddings
  ──────────────────────────────────────────────────────────────────────────────
  Vector Databases      Milvus / Zilliz Cloud         Image similarity search (L2)
                        ChromaDB                      Semantic jewelry filtering
  ──────────────────────────────────────────────────────────────────────────────
  Cloud Storage         AWS S3 (Dual Bucket)          alyaimg, blyskimg
                        boto3                         AWS SDK for Python
  ──────────────────────────────────────────────────────────────────────────────
  Web Scraping          Playwright                    Async browser automation
                        pandas                        Data handling & Excel export
  ──────────────────────────────────────────────────────────────────────────────
  Data Processing       Pillow                        Image processing
                        openpyxl                      Excel file handling
                        pydantic                      Data validation
  ──────────────────────────────────────────────────────────────────────────────
  Other                 python-multipart              File uploads
                        python-dotenv                 Environment config
                        tqdm                          Progress bars
                        transformers                  HuggingFace model hub
  ──────────────────────────────────────────────────────────────────────────────


+=====================================================================================+
|                           ENVIRONMENT VARIABLES REFERENCE                            |
+=====================================================================================+

  VARIABLE                          PURPOSE
  ──────────────────────────────────────────────────────────────────────────────
  GOOGLE_API_KEY                    Gemini API authentication
  HF_TOKEN                          HuggingFace token for models
  OPENAI_API_KEY                    OpenAI services (GPT-4, Image Gen)
  ──────────────────────────────────────────────────────────────────────────────
  ZILLIZ_URL                        Managed Milvus cloud endpoint
  ZILLIZ_TOKEN                      Zilliz authentication
  MILVUS_HOST, MILVUS_PORT          Local Milvus (optional)
  COLLECTION_NAME                   Milvus collection name
  ──────────────────────────────────────────────────────────────────────────────
  AWS_ACCESS_KEY_ID                 AWS S3 access
  AWS_SECRET_ACCESS_KEY             AWS S3 secret
  S3_BUCKET                         Primary bucket (alya or blysk)
  ──────────────────────────────────────────────────────────────────────────────
  LOCAL_IMAGE_DIR                   Local image storage path
  BATCH_SIZE                        Processing batch size (default: 64)
  REINDEX_ON_STARTUP                Full S3 re-index flag (true/false)
  CLEAN_ORPHANS_ON_STARTUP          Cleanup stale Milvus entries (true/false)
  ──────────────────────────────────────────────────────────────────────────────


+=====================================================================================+
|                                    END OF DIAGRAM                                   |
+=====================================================================================+
```

---

## How to Use This Diagram

1. **Print it** - Keep a physical copy at your desk
2. **Search it** - Use Ctrl+F to find specific terms
3. **Reference during calls** - Quick lookup when colleagues ask questions
4. **Onboard new team members** - Walk them through section by section

---

## Recent Changes

- **Dual S3 Bucket Support** - Added support for alya (alyaimg) and blysk (blyskimg) buckets
- **Web Scrapers** - Amazon, Flipkart, Myntra product scrapers using Playwright
- **ChromaDB Semantic Search** - Jewelry filtering by style, color, category, price
- **API Failover** - Multiple Gemini API key rotation with quota detection
- **GPT-4 Image Generation** - Product mockup generation capability
- **Streamlit Frontends** - Bill extractor and jewelry description generator UIs
- **Shopsy Platform** - Added as 4th marketplace (uses Flipkart templates)
- **SKU Data Caching** - OpenAI responses cached in data_store.json by SKU

---

## Quick Reference: Field Counts

| Platform  | Product   | Gemini AI | OpenAI GPT | Fixed | Total |
|-----------|-----------|-----------|------------|-------|-------|
| Amazon    | Earrings  | 44        | 12         | 34    | ~267  |
| Amazon    | Necklace  | 31        | 12         | 31    | ~267  |
| Amazon    | Bracelet  | 37        | 12         | 25    | ~267  |
| Flipkart  | Earrings  | 33        | 1 (desc)   | 27    | ~100  |
| Flipkart  | Necklace  | 35        | 1 (desc)   | 17    | ~100  |
| Flipkart  | Bracelet  | 29        | 1 (desc)   | 22    | ~100  |
| Meesho    | All       | 11        | 2 (t+d)    | 7-9   | ~80   |
| Shopsy    | Earrings  | 33        | 1 (desc)   | 27    | ~100  |

---

**Last Updated:** 2026-02-03
**Version:** Reflects dual-bucket support (commit 67602a0) + Marketplace field details
**Author:** Generated with Claude Code
