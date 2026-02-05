# API Server Test Plan

Before deploying to live AWS, we need to test the following endpoints:
1. `/image_similarity_search`
2. `/generate_caption`
3. `/generate-images`

---

## Table of Contents

1. [Prerequisites](#1-prerequisites)
2. [Test 1: Image Similarity Search](#2-test-1-image-similarity-search)
3. [Test 2: Generate Caption](#3-test-2-generate-caption)
4. [Test 3: Generate Images (Batch-4)](#4-test-3-generate-images-batch-4)
5. [Test 4: IP Metric Verification](#5-test-4-ip-metric-verification)
6. [Test 5: Error Handling](#6-test-5-error-handling)
7. [Test 6: Edge Cases](#7-test-6-edge-cases)
8. [Test Results Template](#8-test-results-template)
9. [Minimum Tests Before Deployment](#9-minimum-tests-before-deployment)

---

## 1. Prerequisites

### Start the Server

```bash
uvicorn img_to_csv:app --reload --host 0.0.0.0 --port 8000
```

### Environment Variables (.env)

```env
GOOGLE_API_KEY=sk-***
AWS_ACCESS_KEY_ID=AKIA***
AWS_SECRET_ACCESS_KEY=***
S3_BUCKET=alyaimg
ZILLIZ_URL=https://***
ZILLIZ_TOKEN=***
OPENAI_API_KEY=sk-***
```

### Prepare Test Images

```
test_images/
├── test_earring_1.jpg    (new image)
├── test_earring_2.jpg    (different image)
└── test_earring_1_copy.jpg (copy for duplicate test)
```

---

## 2. Test 1: Image Similarity Search

**Endpoint:** `POST /image_similarity_search`

**Purpose:** Check if duplicate detection works with IP metric

### Test Cases

| Test # | Test Case | Expected Result |
|--------|-----------|-----------------|
| 1.1 | Upload NEW image with `check_duplicate="yes"` | `duplicate_found: false`, new `s3_url` returned |
| 1.2 | Upload SAME image again with `check_duplicate="yes"` | `duplicate_found: true`, existing URL in results |
| 1.3 | Upload SAME image with `check_duplicate="no"` | `duplicate_found: false`, `duplicate_check: "skipped"`, new URL |
| 1.4 | Upload to "alya" bucket `website="alya"` | `s3_url` contains "alyaimg" |
| 1.5 | Upload to "blysk" bucket `website="blysk"` | `s3_url` contains "blyskimg" |
| 1.6 | Invalid website `website="invalid"` | Error: "Invalid website" |
| 1.7 | Upload slightly modified image (resized) | Should detect as duplicate if >95% similar |
| 1.8 | Upload completely different image | `duplicate_found: false` |

### CURL Commands

#### Test 1.1 - New Image
```bash
curl -X POST "http://localhost:8000/image_similarity_search" \
  -F "file=@test_images/test_earring_1.jpg" \
  -F "check_duplicate=yes" \
  -F "website=alya"
```

#### Test 1.2 - Same Image (Duplicate)
```bash
curl -X POST "http://localhost:8000/image_similarity_search" \
  -F "file=@test_images/test_earring_1.jpg" \
  -F "check_duplicate=yes" \
  -F "website=alya"
```

#### Test 1.3 - Bypass Duplicate Check
```bash
curl -X POST "http://localhost:8000/image_similarity_search" \
  -F "file=@test_images/test_earring_1.jpg" \
  -F "check_duplicate=no" \
  -F "website=alya"
```

#### Test 1.5 - Different Bucket
```bash
curl -X POST "http://localhost:8000/image_similarity_search" \
  -F "file=@test_images/test_earring_2.jpg" \
  -F "check_duplicate=yes" \
  -F "website=blysk"
```

---

## 3. Test 2: Generate Caption

**Endpoint:** `POST /generate_caption`

**Purpose:** Check caption generation + duplicate detection

### Test Cases

| Test # | Test Case | Expected Result |
|--------|-----------|-----------------|
| 2.1 | Upload NEW image with `check_duplicate="yes"` | Full response with all fields |
| 2.2 | Upload DUPLICATE image with `check_duplicate="yes"` | Response with `duplicate: "duplicate found"` |
| 2.3 | Upload DUPLICATE with `check_duplicate="no"` | Full response (treats as new) |
| 2.4 | Test `type="earring"` | Should work |
| 2.5 | Test `type="necklace"` | Should work |
| 2.6 | Test `type="bracelet"` | Should work |
| 2.7 | Test quality detection (A/B/C on image) | Should detect A/B/C |
| 2.8 | Test color detection | Should return correct color |
| 2.9 | Invalid file (not image) | Error response |

### Expected Response (New SKU)

```json
{
  "display_name": "original_filename.jpg",
  "generated_name": "AI_generated_name",
  "quality": "A",
  "description": "one-line caption",
  "attributes": "jewelry features",
  "prompt": "generation prompt",
  "color": "gold",
  "s3_url": "https://alyaimg.s3.amazonaws.com/..."
}
```

### Expected Response (Duplicate)

```json
{
  "display_name": "original_name.jpg",
  "s3_url": "https://existing_image...",
  "duplicate": "duplicate found"
}
```

### CURL Commands

#### Test 2.1 - New Image Caption
```bash
curl -X POST "http://localhost:8000/generate_caption" \
  -F "file=@test_images/test_earring_1.jpg" \
  -F "type=earring" \
  -F "check_duplicate=yes" \
  -F "website=alya"
```

#### Test 2.2 - Duplicate Image
```bash
curl -X POST "http://localhost:8000/generate_caption" \
  -F "file=@test_images/test_earring_1.jpg" \
  -F "type=earring" \
  -F "check_duplicate=yes" \
  -F "website=alya"
```

#### Test 2.3 - Bypass Duplicate
```bash
curl -X POST "http://localhost:8000/generate_caption" \
  -F "file=@test_images/test_earring_1.jpg" \
  -F "type=earring" \
  -F "check_duplicate=no" \
  -F "website=alya"
```

---

## 4. Test 3: Generate Images (Batch-4)

**Endpoint:** `POST /generate-images`

**Purpose:** Check batch image generation for earrings

### Test Cases

| Test # | Test Case | Expected Result |
|--------|-----------|-----------------|
| 3.1 | Earring with default types (no `image_types`) | 4 images: Type 1, 2, 4, 5 + zip_url |
| 3.2 | Earring with custom `image_types=[1,2]` | Only 2 images generated |
| 3.3 | Earring with `image_types=[1,3]` without dimensions | Error: requires height & width |
| 3.4 | Earring with type 3 + dimensions | Dimension image with measurements |
| 3.5 | Bracelet `product_type="bracelet"` | Uses old prompt system |
| 3.6 | Necklace `product_type="necklace"` | Uses old prompt system |
| 3.7 | Invalid product type | Error: "Invalid product_type" |
| 3.8 | Invalid image type `image_types=[6]` | Error: "Invalid image_type" |
| 3.9 | Check response structure | Each image has: prompt_index, image_url, image_type, image_type_name |
| 3.10 | Check ZIP file | ZIP contains all generated images |

### Image Types Reference

| Type | Name | Description |
|------|------|-------------|
| 1 | White Background | Pure white background (Amazon main) |
| 2 | Hand Holding | Earrings on open human hand |
| 3 | Dimension Image | With height/width markings |
| 4 | Lifestyle Nature | Hanging on stick with leaves |
| 5 | Model Wearing | AI model wearing earrings |

### CURL Commands

#### Test 3.1 - Default Batch-4 (Earring)
```bash
curl -X POST "http://localhost:8000/generate-images" \
  -H "Content-Type: application/json" \
  -d '{
    "s3_urls": "https://alyaimg.s3.amazonaws.com/YOUR_IMAGE.jpg",
    "product_type": "earring"
  }'
```

#### Test 3.2 - Custom Types
```bash
curl -X POST "http://localhost:8000/generate-images" \
  -H "Content-Type: application/json" \
  -d '{
    "s3_urls": "https://alyaimg.s3.amazonaws.com/YOUR_IMAGE.jpg",
    "product_type": "earring",
    "image_types": [1, 2]
  }'
```

#### Test 3.4 - With Dimension Image
```bash
curl -X POST "http://localhost:8000/generate-images" \
  -H "Content-Type: application/json" \
  -d '{
    "s3_urls": "https://alyaimg.s3.amazonaws.com/YOUR_IMAGE.jpg",
    "product_type": "earring",
    "image_types": [1, 3],
    "height": "2.5 cm",
    "width": "1.8 cm"
  }'
```

#### Test 3.5 - Bracelet
```bash
curl -X POST "http://localhost:8000/generate-images" \
  -H "Content-Type: application/json" \
  -d '{
    "s3_urls": "https://alyaimg.s3.amazonaws.com/YOUR_IMAGE.jpg",
    "product_type": "bracelet",
    "num_images": 3
  }'
```

---

## 5. Test 4: IP Metric Verification

**Purpose:** Verify IP metric is working correctly

### Test Cases

| Test # | Test Case | Expected Result |
|--------|-----------|-----------------|
| 4.1 | Check distance value in server logs | Distance between 0 and 1 (not 0-100+) |
| 4.2 | Identical image | Distance ~ 0.0 (very similar) |
| 4.3 | Slightly different image (same product, different angle) | Distance ~ 0.01 - 0.05 (still duplicate) |
| 4.4 | Completely different image | Distance > 0.1 (not duplicate) |
| 4.5 | Check collection name in Milvus/Zilliz | Should be "image_embeddings_ip" |

### Distance Reference

```
┌─────────────────────────────────────────────┐
│           IP Distance Values                │
├─────────────────────────────────────────────┤
│  0.00        = Identical (100% similar)     │
│  0.01 - 0.05 = Very similar (duplicate)     │
│  0.05 - 0.10 = Similar (borderline)         │
│  0.10 - 0.50 = Different                    │
│  0.50+       = Very different               │
└─────────────────────────────────────────────┘

Threshold: dist < 0.05 = Duplicate
```

---

## 6. Test 5: Error Handling

### Test Cases

| Test # | Test Case | Expected Result |
|--------|-----------|-----------------|
| 5.1 | Missing file parameter | Error response |
| 5.2 | Invalid file format (not image) | Error: "Invalid image" |
| 5.3 | Invalid S3 URL | Error: "Failed to fetch image" |
| 5.4 | Missing type parameter (generate_caption) | Error response |
| 5.5 | Server without API keys | Clear error message |

---

## 7. Test 6: Edge Cases

### Test Cases

| Test # | Test Case | Expected Result |
|--------|-----------|-----------------|
| 6.1 | Very large image (>10MB) | Should handle or show error |
| 6.2 | Very small image (<10KB) | Should work normally |
| 6.3 | Different formats: JPG, JPEG, PNG, WEBP | All should work |
| 6.4 | Image with transparency (PNG) | Should handle correctly |
| 6.5 | Concurrent requests (multiple uploads) | Should handle without crash |

---

## 8. Test Results Template

Copy this template to record your test results:

### Test 1: Image Similarity Search

| Test # | Status | Notes |
|--------|--------|-------|
| 1.1 | ⬜ | |
| 1.2 | ⬜ | |
| 1.3 | ⬜ | |
| 1.4 | ⬜ | |
| 1.5 | ⬜ | |
| 1.6 | ⬜ | |
| 1.7 | ⬜ | |
| 1.8 | ⬜ | |

### Test 2: Generate Caption

| Test # | Status | Notes |
|--------|--------|-------|
| 2.1 | ⬜ | |
| 2.2 | ⬜ | |
| 2.3 | ⬜ | |
| 2.4 | ⬜ | |
| 2.5 | ⬜ | |
| 2.6 | ⬜ | |
| 2.7 | ⬜ | |
| 2.8 | ⬜ | |
| 2.9 | ⬜ | |

### Test 3: Generate Images

| Test # | Status | Notes |
|--------|--------|-------|
| 3.1 | ⬜ | |
| 3.2 | ⬜ | |
| 3.3 | ⬜ | |
| 3.4 | ⬜ | |
| 3.5 | ⬜ | |
| 3.6 | ⬜ | |
| 3.7 | ⬜ | |
| 3.8 | ⬜ | |
| 3.9 | ⬜ | |
| 3.10 | ⬜ | |

### Test 4: IP Metric

| Test # | Status | Notes |
|--------|--------|-------|
| 4.1 | ⬜ | |
| 4.2 | ⬜ | |
| 4.3 | ⬜ | |
| 4.4 | ⬜ | |
| 4.5 | ⬜ | |

### Test 5: Error Handling

| Test # | Status | Notes |
|--------|--------|-------|
| 5.1 | ⬜ | |
| 5.2 | ⬜ | |
| 5.3 | ⬜ | |
| 5.4 | ⬜ | |
| 5.5 | ⬜ | |

### Test 6: Edge Cases

| Test # | Status | Notes |
|--------|--------|-------|
| 6.1 | ⬜ | |
| 6.2 | ⬜ | |
| 6.3 | ⬜ | |
| 6.4 | ⬜ | |
| 6.5 | ⬜ | |

**Status Legend:** ✅ Pass | ❌ Fail | ⬜ Not Tested | ⏭️ Skipped

---

## 9. Minimum Tests Before Deployment

If time is limited, run at least these **critical tests**:

| Priority | Test | Description |
|----------|------|-------------|
| 🔴 Critical | 1.1 | New image upload works |
| 🔴 Critical | 1.2 | Duplicate detection works |
| 🔴 Critical | 2.1 | Caption generation works |
| 🔴 Critical | 3.1 | Batch-4 image generation works |
| 🟡 Important | 4.1 | IP distance values are 0-1 range |
| 🟡 Important | 2.2 | Caption duplicate detection works |

### Quick Test Script

```bash
# 1. Start server
uvicorn img_to_csv:app --reload --port 8000

# 2. Test similarity search (new image)
curl -X POST "http://localhost:8000/image_similarity_search" \
  -F "file=@test_images/test_earring_1.jpg" \
  -F "check_duplicate=yes"

# 3. Test duplicate detection (same image)
curl -X POST "http://localhost:8000/image_similarity_search" \
  -F "file=@test_images/test_earring_1.jpg" \
  -F "check_duplicate=yes"

# 4. Test caption generation
curl -X POST "http://localhost:8000/generate_caption" \
  -F "file=@test_images/test_earring_2.jpg" \
  -F "type=earring"

# 5. Test batch-4 generation
curl -X POST "http://localhost:8000/generate-images" \
  -H "Content-Type: application/json" \
  -d '{"s3_urls": "YOUR_S3_URL", "product_type": "earring"}'
```

---

*Test Plan created: 5th February 2026*
