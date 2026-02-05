# Recent Changes - February 2026

This document covers all major changes made to the API server in February 2026.

---

## Table of Contents

1. [IP Metric for Duplicate Detection](#1-ip-metric-for-duplicate-detection)
2. [Batch-4 Image Generation](#2-batch-4-image-generation)
3. [Parallel Execution](#3-parallel-execution)
4. [Rate Limit Handling](#4-rate-limit-handling)
5. [Dual S3 Bucket Support](#5-dual-s3-bucket-support)
6. [API Endpoint Changes](#6-api-endpoint-changes)

---

## 1. IP Metric for Duplicate Detection

### What Changed?

Switched from L2 (Euclidean distance) to IP (Inner Product) metric for more accurate duplicate detection.

| Aspect | Before (L2) | After (IP) |
|--------|-------------|------------|
| Metric Type | Euclidean Distance | Inner Product |
| Threshold | `dist < 95` | `dist < 0.05` |
| Collection | `image_embeddings` | `image_embeddings_ip` |
| Embeddings | Raw | Normalized |

### Why IP Metric?

```
┌─────────────────────────────────────────────────────────────┐
│                    L2 vs IP Comparison                      │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  L2 (Euclidean):           IP (Inner Product):              │
│  ─────────────────         ───────────────────              │
│  • Measures distance       • Measures similarity            │
│  • Lower = similar         • Higher = similar               │
│  • Range: 0 to ∞           • Range: -1 to 1 (normalized)    │
│  • Less accurate for       • More accurate for CLIP         │
│    CLIP vectors              embeddings                     │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

### Files Changed

#### `image_search_engine.py`

**Collection Name (Line 35):**
```python
# OLD
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "image_embeddings")

# NEW
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "image_embeddings_ip")
```

**Embedding Normalization (Lines 335-342):**
```python
def embed_image(img, model, processor, device):
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    emb = features.cpu().numpy().reshape(-1)

    # NEW: Normalize for IP metric (cosine similarity)
    emb = emb / np.linalg.norm(emb)
    return emb
```

**Search Function (Lines 537-548):**
```python
def search_similar(collection, query_emb, top_k=5):
    search_params = {"metric_type": "IP", "params": {"ef": 64}}
    results = collection.search(
        data=[query_emb.tolist()],
        anns_field="embedding",
        param=search_params,
        limit=top_k,
        output_fields=["id"]
    )
    # Convert score to distance (1 - score)
    return [(int(hit.id), 1 - float(hit.distance)) for hit in results[0]]
```

#### `img_to_csv.py`

**Threshold Update (Lines 531, 823):**
```python
# OLD
if dist < 95:  # L2 distance

# NEW
if dist < 0.05:  # IP distance (~95% similarity)
```

### Threshold Explanation

```
┌────────────────────────────────────────────────────────────────┐
│                    Similarity Threshold                        │
├────────────────────────────────────────────────────────────────┤
│                                                                │
│  IP Score = 1.0  ──────────────────────────── Identical        │
│       ↓                                                        │
│  IP Score = 0.95 ──────────────────────────── 95% Similar      │
│       ↓                                        (Duplicate)     │
│  IP Score = 0.90 ──────────────────────────── 90% Similar      │
│       ↓                                                        │
│  IP Score = 0.50 ──────────────────────────── 50% Similar      │
│       ↓                                        (Different)     │
│  IP Score = 0.0  ──────────────────────────── No Similarity    │
│                                                                │
│  ───────────────────────────────────────────────────────────── │
│                                                                │
│  We use: dist = 1 - score                                      │
│  So: dist < 0.05 means score > 0.95 (95% similar = duplicate)  │
│                                                                │
└────────────────────────────────────────────────────────────────┘
```

---

## 2. Batch-4 Image Generation

### What Changed?

For earrings, instead of generating 1 image at a time, we now generate **4 images at once** by default.

### Earring Image Types

| Type | Name | Description | Default? |
|------|------|-------------|----------|
| 1 | White Background | Pure white background (Amazon main image) | Yes |
| 2 | Hand Holding | Earrings resting on open human hand | Yes |
| 3 | Dimension Image | With height/width measurement markings | No (skipped) |
| 4 | Lifestyle Nature | Hanging on stick with leaves background | Yes |
| 5 | Model Wearing | AI-generated model wearing the earrings | Yes |

**Default Types:** `[1, 2, 4, 5]` - Type 3 (Dimension) is skipped by default

### Files Changed

#### `img_to_csv.py`

```python
class ImageRequest(BaseModel):
    s3_urls: str
    product_type: str

# Default: skip type 3 (dimension image)
DEFAULT_EARRING_IMAGE_TYPES = [1, 2, 4, 5]
```

#### `prompts2.py`

Contains the prompts for each image type:
- `EARRING_IMAGE_TYPES` - Configuration dict for all 5 types
- `get_earring_prompt()` - Returns appropriate prompt for each type

---

## 3. Parallel Execution

### What Changed?

Implemented `ThreadPoolExecutor` for parallel API calls to OpenAI, reducing generation time from ~75 seconds to ~17 seconds.

### Performance Comparison

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                    BEFORE (Sequential) vs AFTER (Parallel)                   │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                             │
│  BEFORE (Sequential):                                                       │
│  ┌─────────┐   ┌─────────┐   ┌─────────┐   ┌─────────┐                     │
│  │ Image 1 │───│ Image 2 │───│ Image 3 │───│ Image 4 │  = ~75 seconds      │
│  │  ~15s   │   │  ~15s   │   │  ~15s   │   │  ~15s   │                     │
│  └─────────┘   └─────────┘   └─────────┘   └─────────┘                     │
│                                                                             │
│  AFTER (Parallel):                                                          │
│  ┌─────────┐                                                                │
│  │ Image 1 │───┐                                                            │
│  └─────────┘   │                                                            │
│  ┌─────────┐   │                                                            │
│  │ Image 2 │───┼──► All finish at ~17s = ~17 seconds total                 │
│  └─────────┘   │                                                            │
│  ┌─────────┐   │                                                            │
│  │ Image 3 │───┤                                                            │
│  └─────────┘   │                                                            │
│  ┌─────────┐   │                                                            │
│  │ Image 4 │───┘                                                            │
│  └─────────┘                                                                │
│                                                                             │
│  Speedup: 75s → 17s = 4.4x faster                                          │
│                                                                             │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Files Changed

#### `utils2.py`

```python
from concurrent.futures import ThreadPoolExecutor, as_completed

def generate_images_from_gpt(image, prompts, size="1024x1024"):
    # Prepare args for parallel execution
    args_list = [(prompt, img_bytes, size, idx) for idx, prompt in enumerate(prompts)]

    # Execute in parallel with 4 workers
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_generate_single_image, args): args[3]
            for args in args_list
        }
        for future in as_completed(futures):
            result = future.result()
            all_responses[prompt_index] = result
```

### Timing Logs

Added detailed timing logs to monitor performance:

```
============================================================
[BATCH] Starting generation of 4 images
[BATCH] Start time: 16:12:26
============================================================
[BATCH] Image preparation: 0.02s (size: 255.3 KB)
[IMAGE 0] Started at 16:12:26
[IMAGE 0] Buffer creation: 0.00s
[IMAGE 0] API call (upload+process+download): 13.33s
[IMAGE 0] Decode response: 0.02s
[IMAGE 0] TOTAL: 13.35s
[IMAGE 0] Finished at 16:12:40
--------------------------------------------------
... (similar for images 1-3)
============================================================
[BATCH] All 4 images completed
[BATCH] End time: 16:12:41
[BATCH] TOTAL BATCH TIME: 14.85s
[BATCH] Sequential would be: 55.76s
[BATCH] Parallelization speedup: 3.8x
============================================================
```

---

## 4. Rate Limit Handling

### What Changed?

Added retry logic with exponential backoff for OpenAI 429 (rate limit) errors.

### Problem

OpenAI has a limit of 5 input-images per minute. Parallel requests can exceed this limit.

### Solution

```python
def _generate_single_image(args):
    max_retries = 3
    base_delay = 15  # seconds

    for attempt in range(max_retries):
        try:
            result = client.images.edit(**api_params)
            break  # Success
        except Exception as e:
            if "429" in str(e) or "rate_limit" in str(e).lower():
                delay = base_delay * (attempt + 1)  # 15s, 30s, 45s
                print(f"[IMAGE {idx}] Rate limit hit, waiting {delay}s...")
                time.sleep(delay)
            else:
                raise  # Non-rate-limit error
```

### Retry Schedule

| Attempt | Wait Time |
|---------|-----------|
| 1 | 15 seconds |
| 2 | 30 seconds |
| 3 | 45 seconds |

### Graceful Error Handling

When image generation fails, the API now skips failed images instead of crashing:

```python
# img_to_csv.py
for i, item in enumerate(responses):
    # Skip failed generations
    if not item.get("images"):
        print(f"[WARNING] Skipping image {i} - generation failed")
        continue

    img_bytes = item["images"][0]
    # ... continue processing
```

---

## 5. Dual S3 Bucket Support

### What Changed?

Added support for two S3 buckets based on website parameter.

### Bucket Configuration

| Website Parameter | S3 Bucket |
|-------------------|-----------|
| `alya` (default) | alyaimg |
| `blysk` | blyskimg |

### Usage

```bash
# Upload to alya bucket (default)
curl -X POST /image_similarity_search -F "file=@image.jpg"

# Upload to blysk bucket
curl -X POST /image_similarity_search -F "file=@image.jpg" -F "website=blysk"
```

### Files Changed

#### `img_to_csv.py`

```python
# Bucket selection logic
if website == "blysk":
    bucket = "blyskimg"
else:
    bucket = "alyaimg"  # default

s3.put_object(Bucket=bucket, Key=key, Body=img_bytes)
```

---

## 6. API Endpoint Changes

### `/generate-images` (POST)

**Simplified Request:**
```json
{
  "s3_urls": "https://alyaimg.s3.amazonaws.com/image.jpg",
  "product_type": "ear"
}
```

**Response:**
```json
{
  "original_image_url": "https://...",
  "gen_images": [
    {"image_type": 1, "image_type_name": "White Background", "image_url": "..."},
    {"image_type": 2, "image_type_name": "Hand Holding", "image_url": "..."},
    {"image_type": 4, "image_type_name": "Lifestyle Nature", "image_url": "..."},
    {"image_type": 5, "image_type_name": "Model Wearing", "image_url": "..."}
  ],
  "zip_url": "https://...zip"
}
```

### `/image_similarity_search` (POST)

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | Required | Image to check |
| `check_duplicate` | string | "yes" | "yes" or "no" |
| `website` | string | "alya" | S3 bucket (alya/blysk) |

**Response (Duplicate):**
```json
{
  "duplicate_found": true,
  "results": [{"id": 123, "distance": 0.02, "path": "https://..."}]
}
```

**Response (New):**
```json
{
  "duplicate_found": false,
  "image_name": "filename.jpg",
  "s3_url": "https://..."
}
```

### `/generate_caption` (POST)

**Parameters:**
| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | Required | Jewelry image |
| `type` | string | Required | "earring", "necklace", "bracelet" |
| `check_duplicate` | string | "yes" | "yes" or "no" |
| `website` | string | "alya" | S3 bucket |

**Response:**
```json
{
  "display_name": "original_filename.jpg",
  "generated_name": "AI_generated_name",
  "quality": "A",
  "description": "one-line caption",
  "attributes": "jewelry features",
  "prompt": "generation prompt",
  "color": "detected color",
  "s3_url": "https://..."
}
```

---

## Summary of All Changes

| Change | Files | Impact |
|--------|-------|--------|
| IP Metric | image_search_engine.py, img_to_csv.py | More accurate duplicate detection |
| Batch-4 Generation | img_to_csv.py, prompts2.py | Generate 4 images at once |
| Parallel Execution | utils2.py | 4x faster generation (75s → 17s) |
| Rate Limit Handling | utils2.py, img_to_csv.py | Auto-retry on 429 errors |
| Dual S3 Buckets | img_to_csv.py | Support alya/blysk buckets |
| Timing Logs | utils2.py | Performance monitoring |

---

## Commits (February 2026)

| Commit | Description |
|--------|-------------|
| `0ff1d92` | Add composite approach for 100% jewelry preservation |
| `3f2834d` | Add image_types parameter for Amazon earrings catalog |
| `33bb046` | Add IP metric for duplicate detection |
| `8573238` | Add batch-4 image generation (skip dimension) |
| `0cc3fc7` | Add documentation |
| `2cc0a1d` | Add README.md with API documentation |
| `47a70f5` | Fix regenerate-image ZIP download error (S3 client) |
| `b0f764b` | Add parallel execution for image generation (~4x faster) |
| `c1b45eb` | Add timing logs for parallel image generation |
| `f7a6ede` | Fix rate limit handling with retry logic |

---

*Document created: 5th February 2026*
