# Jewelry AI API Server

AI-powered jewelry e-commerce automation system with duplicate detection, caption generation, and batch image generation.

---

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Configure Environment

Create `.env` file:

```env
GOOGLE_API_KEY=your_gemini_api_key
AWS_ACCESS_KEY_ID=your_aws_key
AWS_SECRET_ACCESS_KEY=your_aws_secret
S3_BUCKET=alyaimg
ZILLIZ_URL=your_milvus_url
ZILLIZ_TOKEN=your_milvus_token
OPENAI_API_KEY=your_openai_key
```

### 3. Start Server

```bash
uvicorn img_to_csv:app --reload --host 0.0.0.0 --port 8000
```

### 4. Open API Docs

Go to: http://localhost:8000/docs

---

## Main Endpoints

### 1. `/image_similarity_search` (POST)

Check if image is duplicate, upload if new.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | Required | Image to check |
| `check_duplicate` | string | "yes" | "yes" or "no" |
| `website` | string | "alya" | S3 bucket (alya/blysk) |

**Response (New Image):**
```json
{
  "duplicate_found": false,
  "image_name": "image.jpg",
  "s3_url": "https://alyaimg.s3.amazonaws.com/..."
}
```

**Response (Duplicate):**
```json
{
  "duplicate_found": true,
  "results": [{"id": 123, "distance": 0.02, "path": "https://..."}]
}
```

---

### 2. `/generate_caption` (POST)

Analyze jewelry image and generate AI caption.

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `file` | File | Required | Jewelry image |
| `type` | string | Required | "earring", "necklace", "bracelet" |
| `check_duplicate` | string | "yes" | "yes" or "no" |
| `website` | string | "alya" | S3 bucket |

**Response:**
```json
{
  "display_name": "image.jpg",
  "generated_name": "Golden Cascade Earrings",
  "quality": "A",
  "description": "Elegant gold earrings...",
  "attributes": "Gold-toned discs cascade...",
  "prompt": "...",
  "color": "Gold",
  "s3_url": "https://..."
}
```

---

### 3. `/generate-images` (POST)

Generate 4 product images from jewelry image.

| Parameter | Type | Description |
|-----------|------|-------------|
| `s3_urls` | string | S3 URL of original image |
| `product_type` | string | "ear", "bra", or "nec" |

**Request:**
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
  "zip_url": "https://..."
}
```

**Image Types (Earrings):**
| Type | Name | Description |
|------|------|-------------|
| 1 | White Background | Pure white background (Amazon main) |
| 2 | Hand Holding | Earrings on open hand |
| 4 | Lifestyle Nature | Hanging on stick with leaves |
| 5 | Model Wearing | AI model wearing earrings |

---

## Key Features

### IP Metric Duplicate Detection

- Uses Inner Product (IP) metric for accurate similarity search
- Normalized CLIP embeddings for cosine similarity
- Threshold: `distance < 0.05` (~95% similarity = duplicate)
- Collection: `image_embeddings_ip`

### Dual S3 Bucket Support

| Website | Bucket |
|---------|--------|
| `alya` | alyaimg |
| `blysk` | blyskimg |

### Batch-4 Image Generation

For earrings, generates 4 images at once:
- Type 1: White Background
- Type 2: Hand Holding
- Type 4: Lifestyle Nature
- Type 5: Model Wearing

---

## Technology Stack

| Component | Technology |
|-----------|------------|
| API | FastAPI |
| AI Models | Gemini 2.0, GPT-4 Vision, DALL-E |
| Vector DB | Milvus/Zilliz |
| Storage | AWS S3 |
| Embeddings | CLIP (openai/clip-vit-base-patch32) |

---

## Testing

See [TEST_PLAN.md](TEST_PLAN.md) for comprehensive test cases.

**Test Summary (5th Feb 2026):**
| Test | Result |
|------|--------|
| Image Similarity Search | 8/8 ✅ |
| Generate Caption | 8/9 ✅ |
| Generate Images | 3/10 (6 skipped) ✅ |

---

## Documentation

For detailed documentation, see [DOCUMENTATION/README.md](DOCUMENTATION/README.md)

- [Project Structure Guide](DOCUMENTATION/01_PROJECT_STRUCTURE_GUIDE.md)
- [Code Explanation](DOCUMENTATION/02_CODE_EXPLANATION.md)
- [File Modification Guide](DOCUMENTATION/03_FILE_MODIFICATION_GUIDE.md)
- [Debugging Checklist](DOCUMENTATION/04_DEBUGGING_CHECKLIST.md)
- [Testing Guide](DOCUMENTATION/05_TESTING_GUIDE.md)

---

## Recent Changes (5th Feb 2026)

See [CHANGES_2026_02_05.md](CHANGES_2026_02_05.md) for details.

- Switched from L2 to IP metric for duplicate detection
- Added batch-4 image generation for earrings
- Simplified `/generate-images` API (only requires s3_urls + product_type)
- Added distance/similarity logging

---

## License

Proprietary - All rights reserved.
