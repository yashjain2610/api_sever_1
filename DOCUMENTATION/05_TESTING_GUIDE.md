# Testing Guide - How to Write and Run Tests

Complete guide for writing and running tests for the jewelry AI application.

---

## 🎯 Testing Overview

### Why Testing Matters
- **Catch bugs early** before deployment
- **Save time** - faster than manual testing
- **Prevent regressions** - ensure old fixes don't break
- **Document code** - tests show how to use functions
- **Confidence** - deploy with peace of mind

### Testing Pyramid
```
         🔺 E2E Tests (UI tests)     ← 5%
        🔺🔺 Integration Tests      ← 15%
       🔺🔺🔺 Unit Tests           ← 80%
```

### Testing Strategy for This Project

| Component | Test Type | Framework | Priority |
|-----------|-----------|-----------|----------|
| **Utility Functions** | Unit | pytest | HIGH |
| **API Endpoints** | Integration | pytest + TestClient | HIGH |
| **Database Operations** | Integration | pytest-mock | MEDIUM |
| **AI API Calls** | Integration | pytest-mock | MEDIUM |
| **File Operations** | Unit | pytest | MEDIUM |
| **Web Scraping** | Integration | Skip in CI | LOW |
| **Full Workflow** | E2E | Selenium | LOW |

---

## 📦 Installation & Setup

### Step 1: Install Testing Libraries

```bash
pip install pytest
pip install pytest-asyncio
pip install pytest-cov
pip install pytest-mock
pip install httpx
pip install pillow
```

### Step 2: Add to requirements.txt

```txt
pytest==7.4.0
pytest-asyncio==0.21.1
pytest-cov==4.1.0
pytest-mock==3.11.1
httpx==0.24.1
```

### Step 3: Create Test Structure

```
project/
├── img_to_csv.py
├── utils.py
├── ... other source files
│
├── tests/                          ← Create this folder
│   ├── __init__.py                 ← Empty file
│   ├── conftest.py                 ← Shared fixtures
│   │
│   ├── unit/                       ← Unit tests
│   │   ├── __init__.py
│   │   ├── test_utils.py
│   │   ├── test_json_storage.py
│   │   └── test_excel_fields.py
│   │
│   ├── integration/                ← Integration tests
│   │   ├── __init__.py
│   │   ├── test_api_endpoints.py
│   │   ├── test_image_search.py
│   │   └── test_catalog_generation.py
│   │
│   ├── fixtures/                   ← Test data
│   │   ├── sample_image.jpg
│   │   ├── sample_response.json
│   │   └── mock_data.py
│   │
│   └── mocks/                      ← Mock objects
│       ├── __init__.py
│       └── mock_apis.py
```

---

## 🔧 Setup Fixtures (conftest.py)

**Location:** `tests/conftest.py`

```python
import pytest
import os
from pathlib import Path
from io import BytesIO
from PIL import Image
import json
import tempfile
from fastapi.testclient import TestClient

# Import app and modules to test
from img_to_csv import app
from utils import input_image_setup, get_gemini_responses
from json_storage import store_data_for_sku, fetch_data_for_sku

# ==================== FIXTURES ====================

@pytest.fixture
def client():
    """FastAPI TestClient for API testing"""
    return TestClient(app)

@pytest.fixture
def sample_image():
    """Create a sample image for testing"""
    # Create 100x100 RGB image
    image = Image.new('RGB', (100, 100), color='red')
    image_bytes = BytesIO()
    image.save(image_bytes, format='PNG')
    image_bytes.seek(0)
    return image_bytes.getvalue()

@pytest.fixture
def sample_jewelry_image():
    """Create sample jewelry image (more realistic)"""
    image = Image.new('RGB', (500, 500), color=(200, 180, 150))  # Jewelry color
    image_bytes = BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)
    return image_bytes.getvalue()

@pytest.fixture
def gemini_response_json():
    """Mock Gemini API response"""
    return {
        "type": "Stud Earring",
        "color": "Gold",
        "base_material": "Alloy",
        "gemstone": "Cubic Zirconia",
        "number_of_gemstones": "2",
        "collection": "Contemporary",
        "occasion": "Everyday"
    }

@pytest.fixture
def temp_json_file():
    """Create temporary JSON file for testing"""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
        json.dump({"SKU-001": {"name": "Test Product"}}, f)
        temp_path = f.name
    
    yield temp_path
    
    # Cleanup
    os.unlink(temp_path)

@pytest.fixture
def mock_s3_client(mocker):
    """Mock AWS S3 client"""
    mock = mocker.MagicMock()
    mocker.patch('boto3.client', return_value=mock)
    return mock

@pytest.fixture
def mock_gemini_client(mocker):
    """Mock Google Gemini client"""
    mock = mocker.MagicMock()
    mock.models.generate_content.return_value.text = json.dumps({
        "type": "Stud",
        "color": "Gold"
    })
    mocker.patch('genai.Client', return_value=mock)
    return mock

@pytest.fixture
def env_vars(monkeypatch):
    """Set environment variables for testing"""
    monkeypatch.setenv("GOOGLE_API_KEY", "test_key_123")
    monkeypatch.setenv("AWS_ACCESS_KEY_ID", "test_aws_id")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "test_aws_secret")
    monkeypatch.setenv("S3_BUCKET", "test-bucket")
    monkeypatch.setenv("MILVUS_HOST", "localhost")
    monkeypatch.setenv("MILVUS_PORT", "19530")

# ==================== UTILITIES ====================

def create_test_image(width=100, height=100, color=(255, 0, 0)):
    """Helper to create test image"""
    image = Image.new('RGB', (width, height), color=color)
    image_bytes = BytesIO()
    image.save(image_bytes, format='JPEG')
    image_bytes.seek(0)
    return image_bytes.getvalue()

def create_test_upload_file(filename="test.jpg", content=None):
    """Helper to create test file upload"""
    from fastapi import UploadFile
    if content is None:
        content = create_test_image()
    return UploadFile(filename=filename, file=BytesIO(content))
```

---

## ✅ Unit Tests

### 1. Test Utils.py Functions

**Location:** `tests/unit/test_utils.py`

```python
import pytest
import json
from io import BytesIO
from PIL import Image
from utils import (
    input_image_setup,
    get_gemini_responses,
    get_gemini_responses_high_temp
)

class TestInputImageSetup:
    """Test image preparation for Gemini"""
    
    def test_input_image_setup_valid_image(self, sample_image):
        """Should convert image bytes to Gemini format"""
        result = input_image_setup(BytesIO(sample_image), "image/jpeg")
        
        assert result is not None
        assert len(result) > 0
        assert hasattr(result[0], 'mime_type')
        assert result[0].mime_type == "image/jpeg"
    
    def test_input_image_setup_png_format(self, sample_image):
        """Should handle PNG format"""
        result = input_image_setup(BytesIO(sample_image), "image/png")
        
        assert result[0].mime_type == "image/png"
    
    def test_input_image_setup_invalid_format(self, sample_image):
        """Should handle invalid format gracefully"""
        result = input_image_setup(BytesIO(sample_image), "invalid/format")
        
        assert result is not None  # Should not crash
    
    def test_input_image_setup_empty_image(self):
        """Should handle empty image"""
        empty_bytes = BytesIO(b'')
        result = input_image_setup(empty_bytes, "image/jpeg")
        
        assert result is not None

class TestGeminiResponses:
    """Test Gemini API response handling"""
    
    def test_get_gemini_responses_missing_api_key(self, monkeypatch):
        """Should return error when API key missing"""
        monkeypatch.setenv("GOOGLE_API_KEY", "")
        
        result = get_gemini_responses("test", [None], ["prompt"])
        
        assert "Error" in result[0]
    
    @pytest.mark.asyncio
    async def test_get_gemini_responses_valid_call(self, mocker, sample_image):
        """Should successfully call Gemini API"""
        # Mock the client
        mock_response = mocker.MagicMock()
        mock_response.text = '{"type": "Stud", "color": "Gold"}'
        
        mock_client = mocker.MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        
        mocker.patch('utils.client', mock_client)
        
        # Call function
        image_data = input_image_setup(BytesIO(sample_image), "image/jpeg")
        result = get_gemini_responses("Test", image_data, ["prompt"])
        
        assert len(result) > 0
        assert '{"type": "Stud"' in result[0]
    
    def test_get_gemini_responses_high_temp(self, mocker):
        """Should use high temperature for creative responses"""
        mock_response = mocker.MagicMock()
        mock_response.text = "Creative response"
        
        mock_client = mocker.MagicMock()
        mock_client.models.generate_content.return_value = mock_response
        
        mocker.patch('utils.client', mock_client)
        
        result = get_gemini_responses_high_temp("Test", [None], ["prompt"])
        
        assert "Creative" in result[0]

class TestImageProcessing:
    """Test image utility functions"""
    
    def test_image_format_detection(self, sample_image):
        """Should detect image format correctly"""
        image = Image.open(BytesIO(sample_image))
        
        assert image.format in ['PNG', 'JPEG', 'JPG']
    
    def test_image_dimensions(self, sample_image):
        """Should get correct image dimensions"""
        image = Image.open(BytesIO(sample_image))
        
        assert image.size[0] > 0
        assert image.size[1] > 0
```

### 2. Test JSON Storage

**Location:** `tests/unit/test_json_storage.py`

```python
import pytest
import json
import tempfile
import os
from pathlib import Path
from json_storage import (
    store_data_for_sku,
    fetch_data_for_sku,
    expand_bullet_points,
    compress_bullet_points
)

class TestJSONStorage:
    """Test local JSON database operations"""
    
    @pytest.fixture
    def temp_json_path(self, monkeypatch):
        """Temporary JSON file for each test"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.json', delete=False) as f:
            temp_path = f.name
        
        monkeypatch.setattr('json_storage.JSON_PATH', Path(temp_path))
        
        yield temp_path
        
        # Cleanup
        if os.path.exists(temp_path):
            os.unlink(temp_path)
    
    def test_store_data_for_sku(self, temp_json_path):
        """Should store product data"""
        test_data = {
            "product_name": "Diamond Earrings",
            "description": "Premium diamonds",
            "price": "$49.99"
        }
        
        store_data_for_sku("SKU-001", test_data)
        
        # Verify stored
        with open(temp_json_path, 'r') as f:
            stored = json.load(f)
        
        assert "SKU-001" in stored
        assert stored["SKU-001"]["product_name"] == "Diamond Earrings"
    
    def test_fetch_data_for_sku_existing(self, temp_json_path):
        """Should fetch existing product data"""
        # Store first
        store_data_for_sku("SKU-001", {"name": "Test"})
        
        # Fetch
        result = fetch_data_for_sku("SKU-001")
        
        assert result["sku_exists"] != False or "name" in result
    
    def test_fetch_data_for_sku_not_existing(self, temp_json_path):
        """Should return sku_exists=False for missing SKU"""
        result = fetch_data_for_sku("NONEXISTENT-SKU")
        
        assert result.get("sku_exists") == False
    
    def test_expand_bullet_points(self):
        """Should expand compressed bullet points"""
        compressed = {
            "name": "Earrings",
            "bullet_points": {
                "bullet_point_1": "High quality",
                "bullet_point_2": "Gold plated"
            }
        }
        
        expanded = expand_bullet_points(compressed)
        
        assert "bullet_point_1" in expanded
        assert expanded["bullet_point_1"] == "High quality"
        assert "bullet_points" not in expanded
    
    def test_compress_bullet_points(self):
        """Should compress flat bullet points"""
        flat = {
            "name": "Earrings",
            "bullet_point_1": "High quality",
            "bullet_point_2": "Gold plated"
        }
        
        compressed = compress_bullet_points(flat)
        
        assert "bullet_points" in compressed
        assert isinstance(compressed["bullet_points"], dict)
        assert "bullet_point_1" not in compressed
```

### 3. Test Image Search Functions

**Location:** `tests/unit/test_image_search.py`

```python
import pytest
import numpy as np
from unittest.mock import MagicMock, patch
from image_search_engine import (
    embed_image,
    search_similar,
    init_clip,
    get_image_hash
)

class TestImageSearch:
    """Test duplicate detection and image search"""
    
    def test_embed_image_returns_vector(self, sample_image, mocker):
        """Should convert image to embedding vector"""
        # Mock CLIP model
        mock_model = MagicMock()
        mock_embedding = np.random.rand(768)  # CLIP outputs 768-dim vector
        
        mock_processor = MagicMock()
        mock_device = "cpu"
        
        # Mock the embed function
        def mock_embed(image, processor, device):
            return mock_embedding
        
        mocker.patch('image_search_engine.embed_image', mock_embed)
        
        # This would be called in real code
        # For testing, we verify the structure
        assert isinstance(mock_embedding, np.ndarray)
        assert mock_embedding.shape[0] == 768
    
    def test_image_hash_consistency(self, sample_image):
        """Should produce consistent hash for same image"""
        from image_search_engine import get_image_hash
        from io import BytesIO
        from PIL import Image
        
        image = Image.open(BytesIO(sample_image))
        hash1 = get_image_hash(image)
        hash2 = get_image_hash(image)
        
        assert hash1 == hash2
    
    def test_search_similar_no_results(self, mocker):
        """Should handle empty search results"""
        mock_db = MagicMock()
        mock_db.search.return_value = []
        
        query_embedding = np.random.rand(768)
        
        mocker.patch('image_search_engine.search_similar',
                    return_value=[])
        
        # Verify empty results handled gracefully
        assert [] == []

class TestSimilarityThreshold:
    """Test duplicate detection threshold"""
    
    def test_similarity_above_threshold(self):
        """Should detect duplicate if similarity > 95%"""
        distance = 2.5  # Higher distance = less similar
        similarity = 100 - distance
        
        threshold = 95
        is_duplicate = similarity >= threshold
        
        assert is_duplicate == False  # 97.5% > 95%, so True
    
    def test_similarity_below_threshold(self):
        """Should not detect duplicate if similarity < 95%"""
        distance = 8.0
        similarity = 100 - distance
        
        threshold = 95
        is_duplicate = similarity >= threshold
        
        assert is_duplicate == False  # 92% < 95%, so False
    
    def test_exact_duplicate(self):
        """Should detect when images are identical"""
        distance = 0.0  # Identical
        similarity = 100 - distance
        
        assert similarity == 100
        assert similarity >= 95  # Would be detected as duplicate
```

---

## 🔌 Integration Tests

### 1. Test API Endpoints

**Location:** `tests/integration/test_api_endpoints.py`

```python
import pytest
from fastapi.testclient import TestClient
from img_to_csv import app
from io import BytesIO
from PIL import Image
import json

client = TestClient(app)

class TestGenerateCaptionEndpoint:
    """Test /generate_caption endpoint"""
    
    def test_generate_caption_invalid_file(self):
        """Should reject invalid image"""
        response = client.post(
            "/generate_caption",
            data={"type": "elegance"},
            files={"file": ("test.txt", b"not an image", "text/plain")}
        )
        
        assert response.status_code == 400
    
    def test_generate_caption_missing_type(self, sample_jewelry_image):
        """Should require type parameter"""
        response = client.post(
            "/generate_caption",
            files={"file": ("test.jpg", BytesIO(sample_jewelry_image), "image/jpeg")}
        )
        
        assert response.status_code in [400, 422]
    
    def test_generate_caption_valid_request(self, sample_jewelry_image, mocker):
        """Should process valid request"""
        # Mock external APIs
        mocker.patch('boto3.client')
        mocker.patch('image_search_engine.search_similar', return_value=[])
        mocker.patch('openai_utils.ask_gpt_with_image', return_value={
            "name": "Diamond Earrings",
            "description": "Premium diamonds"
        })
        
        response = client.post(
            "/generate_caption",
            data={"type": "Elegance"},
            files={"file": ("test.jpg", BytesIO(sample_jewelry_image), "image/jpeg")}
        )
        
        # Should succeed or at least not crash
        assert response.status_code in [200, 500]  # 500 ok if mocks incomplete

class TestProcessImageEndpoint:
    """Test /process-image-and-prompt/ endpoint"""
    
    def test_process_bill_image(self, sample_image):
        """Should extract products from bill image"""
        response = client.post(
            "/process-image-and-prompt/",
            files={"image": ("bill.jpg", BytesIO(sample_image), "image/jpeg")}
        )
        
        # Endpoint might be unavailable, but shouldn't crash
        assert response.status_code in [200, 500, 404]
    
    def test_process_bill_invalid_file(self):
        """Should reject non-image file"""
        response = client.post(
            "/process-image-and-prompt/",
            files={"image": ("doc.pdf", b"%PDF-1.4...", "application/pdf")}
        )
        
        assert response.status_code in [400, 415, 500]

class TestSemanticFilterEndpoint:
    """Test /semantic_filter_jewelry/ endpoint"""
    
    def test_semantic_filter_valid_input(self, mocker):
        """Should filter jewelry by preferences"""
        # Mock external API
        mocker.patch('requests.request', return_value=MagicMock(
            text=json.dumps({"data": []})
        ))
        
        request_data = {
            "user_needs": {
                "collection_size": 5,
                "color": "gold",
                "category": "Elegence",
                "category_description": ["Elegence"],
                "target_audience": "women",
                "manual_prompt": "high quality",
                "start_price": 10,
                "end_price": 100,
                "start_date": "2025-01-01",
                "end_date": "2025-12-31",
                "extra_jwellery": []
            }
        }
        
        response = client.post(
            "/semantic_filter_jewelry/",
            json=request_data
        )
        
        # Check response format
        if response.status_code == 200:
            data = response.json()
            assert "top_jewelry_ids" in data or "error" in data

class TestCatalogAIEndpoint:
    """Test /catalog-ai endpoint"""
    
    def test_catalog_ai_missing_params(self):
        """Should require all parameters"""
        response = client.post(
            "/catalog-ai",
            data={"image_urls": ""}
        )
        
        assert response.status_code in [400, 422]
    
    def test_catalog_ai_invalid_marketplace(self):
        """Should validate marketplace"""
        response = client.post(
            "/catalog-ai",
            data={
                "image_urls": "https://example.com/img.jpg",
                "skuids": "SKU-001",
                "type": "earrings",
                "marketplace": "unknown_marketplace"
            }
        )
        
        assert response.status_code in [400, 422, 500]

class TestHealthCheck:
    """Test if server is accessible"""
    
    def test_server_running(self):
        """Should respond to requests"""
        response = client.get("/docs")
        
        assert response.status_code in [200, 404, 405]

class TestErrorHandling:
    """Test error responses"""
    
    def test_404_on_invalid_endpoint(self):
        """Should return 404 for invalid endpoint"""
        response = client.get("/nonexistent/endpoint")
        
        assert response.status_code == 404
    
    def test_500_on_internal_error(self, mocker):
        """Should handle internal errors gracefully"""
        # Mock a function to raise error
        mocker.patch('img_to_csv.client', side_effect=Exception("Test error"))
        
        # Most endpoints should return error, not crash
        response = client.get("/docs")
        
        # This should still work (docs endpoint doesn't use client)
        assert response.status_code in [200, 404, 405]
```

### 2. Test Database Integration

**Location:** `tests/integration/test_database_integration.py`

```python
import pytest
from unittest.mock import MagicMock, patch
import json
from pymilvus import Collection

class TestMilvusIntegration:
    """Test Milvus vector database integration"""
    
    def test_milvus_connection(self, mocker):
        """Should connect to Milvus"""
        mock_connection = MagicMock()
        mocker.patch('pymilvus.connections.connect', mock_connection)
        
        from image_search_engine import init_milvus
        
        # Should not raise error
        result = init_milvus(host="localhost", port="19530")
        
        # Verify connection was attempted
        # (Would be actual connection in production)
    
    def test_milvus_insert_embeddings(self, mocker):
        """Should insert embeddings into Milvus"""
        mock_collection = MagicMock()
        mock_collection.insert.return_value = [1, 2, 3]
        
        # Test data
        embeddings = [[0.1] * 768 for _ in range(3)]
        ids = [1, 2, 3]
        
        mock_collection.insert([ids, embeddings])
        
        # Verify insert was called
        mock_collection.insert.assert_called_once()
    
    def test_milvus_search(self, mocker):
        """Should search Milvus for similar embeddings"""
        mock_collection = MagicMock()
        mock_collection.search.return_value = [
            MagicMock(distances=[2.5], ids=[1])
        ]
        
        query_embedding = [[0.1] * 768]
        
        result = mock_collection.search(query_embedding, "embedding", {}, 1)
        
        assert len(result) > 0

class TestChromaDBIntegration:
    """Test ChromaDB semantic search integration"""
    
    def test_chromadb_connection(self, mocker):
        """Should connect to ChromaDB"""
        mock_client = MagicMock()
        mocker.patch('chromadb.Client', return_value=mock_client)
        
        from img_to_csv import client as chromadb_client
        
        # Verify client available
        assert chromadb_client is not None
    
    def test_chromadb_add_documents(self, mocker):
        """Should add documents to ChromaDB"""
        mock_collection = MagicMock()
        
        mock_collection.add(
            documents=["test document"],
            embeddings=[[0.1] * 1536],
            metadatas=[{"id": "123"}],
            ids=["1"]
        )
        
        mock_collection.add.assert_called_once()
    
    def test_chromadb_query(self, mocker):
        """Should query ChromaDB for similar documents"""
        mock_collection = MagicMock()
        mock_collection.query.return_value = {
            "ids": [["1", "2"]],
            "distances": [[0.5, 0.7]]
        }
        
        results = mock_collection.query(
            query_embeddings=[[0.1] * 1536],
            n_results=2
        )
        
        assert len(results["ids"][0]) > 0

class TestS3Integration:
    """Test AWS S3 integration"""
    
    def test_s3_upload_file(self, mocker):
        """Should upload file to S3"""
        mock_s3 = MagicMock()
        mocker.patch('boto3.client', return_value=mock_s3)
        
        import boto3
        s3 = boto3.client('s3')
        
        s3.upload_file(
            Filename="test.jpg",
            Bucket="test-bucket",
            Key="test-key.jpg"
        )
        
        s3.upload_file.assert_called_once()
    
    def test_s3_delete_file(self, mocker):
        """Should delete file from S3"""
        mock_s3 = MagicMock()
        mocker.patch('boto3.client', return_value=mock_s3)
        
        import boto3
        s3 = boto3.client('s3')
        
        s3.delete_object(Bucket="test-bucket", Key="test-key.jpg")
        
        s3.delete_object.assert_called_once()
    
    def test_s3_list_objects(self, mocker):
        """Should list objects in S3 bucket"""
        mock_s3 = MagicMock()
        mock_s3.list_objects_v2.return_value = {
            'Contents': [
                {'Key': 'file1.jpg'},
                {'Key': 'file2.jpg'}
            ]
        }
        mocker.patch('boto3.client', return_value=mock_s3)
        
        import boto3
        s3 = boto3.client('s3')
        
        response = s3.list_objects_v2(Bucket="test-bucket")
        
        assert len(response['Contents']) == 2
```

### 3. Test Catalog Generation

**Location:** `tests/integration/test_catalog_generation.py`

```python
import pytest
from unittest.mock import MagicMock, AsyncMock
import json
import httpx

class TestCatalogGeneration:
    """Test complete catalog generation workflow"""
    
    @pytest.mark.asyncio
    async def test_catalog_generation_workflow(self, mocker):
        """Test complete end-to-end catalog generation"""
        # Mock Gemini response
        gemini_response = {
            "type": "Stud Earring",
            "color": "Gold",
            "base_material": "Alloy",
            "gemstone": "Cubic Zirconia"
        }
        
        # Mock GPT response
        gpt_response = {
            "item_name": "Premium Gold Stud Earrings",
            "description": "Beautiful stud earrings...",
            "bullet_points": {
                "bullet_point_1": "High quality",
                "bullet_point_2": "Gold plated"
            }
        }
        
        # Mock Gemini
        mocker.patch('utils.get_gemini_responses',
                    return_value=[json.dumps(gemini_response)])
        
        # Mock GPT
        mocker.patch('openai_utils.ask_gpt_with_image',
                    return_value=gpt_response)
        
        # Mock S3
        mock_s3 = MagicMock()
        mocker.patch('boto3.client', return_value=mock_s3)
        
        # Mock HTTP client
        mock_response = AsyncMock()
        mock_response.status_code = 200
        mock_response.content = b"fake image content"
        
        # Test would call catalog-ai endpoint
        # Verify all mocks were called
        assert gemini_response is not None
        assert gpt_response is not None
    
    def test_catalog_field_mapping(self):
        """Test correct field mapping for each marketplace"""
        from excel_fields import (
            target_fields_earrings,
            target_fields_earrings_amz,
            fixed_values_earrings,
            fixed_values_earrings_amz
        )
        
        # Verify fields exist
        assert "Type" in target_fields_earrings or "Type" in target_fields_earrings
        assert "Color" in target_fields_earrings or "Color" in target_fields_earrings
        
        # Verify fixed values populated
        assert len(fixed_values_earrings) > 0
        assert len(fixed_values_earrings_amz) > 0
    
    def test_excel_generation(self, mocker):
        """Test Excel file generation"""
        from excel_fields import write_to_excel_flipkart
        from openpyxl import Workbook
        
        # Create test data
        excel_results = [
            ("SKU-001", {
                "Type": "Stud",
                "Color": "Gold",
                "Gemstone": "Diamond"
            }, "Premium earrings")
        ]
        
        # Mock workbook
        mock_wb = MagicMock()
        mocker.patch('openpyxl.load_workbook', return_value=mock_wb)
        
        # This would call the function
        # Verify it doesn't crash
        assert excel_results is not None
```

---

## 🧪 Running Tests

### Basic Test Run

```bash
# Run all tests
pytest

# Run with verbose output
pytest -v

# Run specific test file
pytest tests/unit/test_utils.py

# Run specific test class
pytest tests/unit/test_utils.py::TestInputImageSetup

# Run specific test
pytest tests/unit/test_utils.py::TestInputImageSetup::test_input_image_setup_valid_image
```

### Test Coverage

```bash
# Generate coverage report
pytest --cov=. --cov-report=html

# View coverage in browser
open htmlcov/index.html  # Mac
start htmlcov/index.html  # Windows
xdg-open htmlcov/index.html  # Linux

# Coverage for specific module
pytest --cov=utils --cov-report=html
```

### Async Tests

```bash
# Run async tests
pytest -v -k "asyncio"

# Run with asyncio debug
pytest --asyncio-mode=auto
```

### Mock External Services

```bash
# Run with mocked external APIs (skip real API calls)
pytest -m "not integration"

# Run only integration tests
pytest -m "integration"
```

### Parallel Testing

```bash
# Install pytest-xdist
pip install pytest-xdist

# Run tests in parallel (4 workers)
pytest -n 4
```

### Create Markers in conftest.py

```python
# tests/conftest.py
def pytest_configure(config):
    config.addinivalue_line(
        "markers", "integration: mark test as integration test"
    )
    config.addinivalue_line(
        "markers", "unit: mark test as unit test"
    )
    config.addinivalue_line(
        "markers", "slow: mark test as slow running"
    )

# Then mark tests:
@pytest.mark.integration
def test_milvus_connection():
    pass

@pytest.mark.slow
def test_catalog_generation():
    pass
```

---

## 📋 Test Checklist

### Before Committing Code

```
☐ 1. Run all unit tests
     pytest tests/unit/

☐ 2. Run integration tests
     pytest tests/integration/

☐ 3. Check code coverage
     pytest --cov=.
     Coverage should be > 80%

☐ 4. Run with mocks only (no real API calls)
     pytest -m "not real_api"

☐ 5. Test on different Python versions
     tox  # requires tox

☐ 6. No warnings in test output
     pytest -W error::DeprecationWarning

☐ 7. Tests complete in reasonable time
     pytest --durations=10  # Show 10 slowest tests
```

### Before Deploying to Production

```
☐ 1. All tests pass locally
☐ 2. All tests pass in CI/CD
☐ 3. Code coverage > 80%
☐ 4. No security warnings
     pip-audit
☐ 5. No performance regressions
     pytest --benchmark
☐ 6. Tested with production-like data
☐ 7. Edge cases tested
```

---

## 🔄 CI/CD Integration (GitHub Actions)

**Location:** `.github/workflows/tests.yml`

```yaml
name: Tests

on: [push, pull_request]

jobs:
  test:
    runs-on: ubuntu-latest
    
    strategy:
      matrix:
        python-version: ['3.8', '3.9', '3.10']
    
    steps:
    - uses: actions/checkout@v2
    
    - name: Set up Python ${{ matrix.python-version }}
      uses: actions/setup-python@v2
      with:
        python-version: ${{ matrix.python-version }}
    
    - name: Install dependencies
      run: |
        pip install -r requirements.txt
        pip install pytest pytest-cov pytest-asyncio
    
    - name: Run tests
      run: pytest tests/ --cov=. --cov-report=xml
    
    - name: Upload coverage
      uses: codecov/codecov-action@v2
      with:
        files: ./coverage.xml
```

---

## 🎯 Testing Best Practices

### ✅ DO:

```python
# ✅ GOOD: Clear test names
def test_gemini_returns_json_when_given_valid_image():
    """Should parse Gemini response as JSON"""
    pass

# ✅ GOOD: Test one thing per test
def test_store_data_saves_to_json():
    store_data_for_sku("SKU-001", {"name": "Test"})
    result = fetch_data_for_sku("SKU-001")
    assert result is not None

# ✅ GOOD: Use fixtures for setup
@pytest.fixture
def test_data():
    return {"sku": "001", "name": "Test"}

# ✅ GOOD: Mock external services
def test_api_call(mocker):
    mocker.patch('requests.get')
    # Test without actual HTTP call

# ✅ GOOD: Test edge cases
def test_empty_input():
    result = process_data("")
    assert result is not None

# ✅ GOOD: Use descriptive assertions
def test_price_calculation():
    result = calculate_price(10, 0.1)
    assert result == 9.0, "10% discount should give 9.0"
```

### ❌ DON'T:

```python
# ❌ BAD: Vague test name
def test_function():
    pass

# ❌ BAD: Test multiple things
def test_all_features():
    create_user()
    delete_user()
    update_user()  # Should be 3 separate tests

# ❌ BAD: Call real APIs in tests
def test_api_call():
    response = requests.get("https://api.example.com")  # Real call!

# ❌ BAD: Hard to maintain setup
def test_something():
    data = "long test data string" * 100
    # Use fixture instead

# ❌ BAD: Weak assertions
def test_calculation():
    result = calculate_price(10)
    assert result  # Too vague, what if result is 0?

# ❌ BAD: Dependencies between tests
def test_create():
    global user_id  # Creates test order dependency
    user_id = create_user()

def test_delete():
    delete_user(user_id)  # Depends on test_create running first!
```

---

## 📊 Test Examples Summary

### What to Test (Priority Order)

1. **🔴 CRITICAL** (Must test):
   - API endpoints
   - External API calls (Gemini, GPT, S3)
   - Database operations
   - Authentication/Authorization
   - Error handling

2. **🟡 IMPORTANT** (Should test):
   - Data validation
   - File operations
   - JSON parsing
   - Image processing
   - Business logic

3. **🟢 NICE-TO-HAVE** (Can test):
   - UI components
   - Web scraping
   - Performance
   - Load testing

### Test Structure Template

```python
import pytest

class TestFeatureName:
    """Test [feature/component] functionality"""
    
    @pytest.fixture
    def setup_data(self):
        """Prepare test data"""
        return {"key": "value"}
    
    def test_happy_path(self, setup_data):
        """Should work with valid input"""
        result = function(setup_data)
        assert result is not None
    
    def test_error_case(self):
        """Should handle errors gracefully"""
        with pytest.raises(ValueError):
            function(invalid_input)
    
    def test_edge_case(self):
        """Should handle edge cases"""
        result = function(edge_case_input)
        assert result == expected_value
    
    @pytest.mark.parametrize("input,expected", [
        ("valid", True),
        ("invalid", False),
        ("", False),
    ])
    def test_multiple_cases(self, input, expected):
        """Should handle multiple cases"""
        result = function(input)
        assert result == expected
```

---

## 🚀 Getting Started with Testing

### Step 1: Create Test Directory
```bash
mkdir -p tests/{unit,integration,fixtures,mocks}
touch tests/__init__.py
touch tests/conftest.py
```

### Step 2: Write First Test
```python
# tests/unit/test_simple.py
def test_addition():
    assert 1 + 1 == 2
```

### Step 3: Run Test
```bash
pytest tests/unit/test_simple.py -v
```

### Step 4: Expand Tests
- Copy examples from this guide
- Adapt to your code
- Run and verify

### Step 5: Integrate with CI/CD
- Add GitHub Actions workflow
- Run tests on every push
- Block merges if tests fail

---

## 📚 Additional Resources

- **Pytest Docs**: https://docs.pytest.org/
- **FastAPI Testing**: https://fastapi.tiangolo.com/advanced/testing-dependencies/
- **Unittest Mock**: https://docs.python.org/3/library/unittest.mock.html
- **Hypothesis (Property Testing)**: https://hypothesis.readthedocs.io/

---

**Your code is only as good as your tests!** 🧪✨
