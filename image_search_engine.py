import os
import sys
import json
import hashlib
from PIL import Image
from tqdm import tqdm
import numpy as np
import torch
from transformers import CLIPProcessor, CLIPModel
import streamlit as st
import urllib.parse
from pymilvus import (
    connections,
    FieldSchema,
    CollectionSchema,
    DataType,
    Collection,
    utility
)
from dotenv import load_dotenv
import boto3
from io import BytesIO
import time

load_dotenv()

S3_BUCKET = os.getenv("S3_BUCKET", "alyaimg")
S3_PREFIX = os.getenv("S3_IMAGE_PREFIX", "")


LOCAL_IMAGE_DIR = os.getenv("LOCAL_IMAGE_DIR", "./images")

MILVUS_HOST = os.getenv("MILVUS_HOST", "localhost")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")
COLLECTION_NAME = os.getenv("COLLECTION_NAME", "image_embeddings")

BATCH_SIZE = int(os.getenv("BATCH_SIZE", "64"))
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
INDEX_FILE = "indexed_hashes.json"
INT_HASH_MAP_FILE = "int_hash_to_path.json"

URL   = os.getenv("ZILLIZ_URL")
TOKEN = os.getenv("ZILLIZ_TOKEN")

# IMAGE_EXTS = (".jpg", ".jpeg", ".png", ".webp")
# EXCLUDED_TOP_LEVEL = {"gen_images/", "excel_files/"} 

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.getenv("AWS_ACCESS_KEY_ID"),
    aws_secret_access_key=os.getenv("AWS_SECRET_ACCESS_KEY"),
    )


# ------------------------ Initialize Model ------------------------
def get_image_paths_from_s3(bucket_name: str, prefix: str = ""):
    paginator = s3.get_paginator("list_objects_v2")
    image_paths = []

    for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
        for obj in page.get("Contents", []):
            key = obj["Key"]
            if key.startswith("gen_images/") or key.startswith("excel_files/"):
                continue
            if key.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                image_paths.append(key)
    #print(image_paths)
    return sorted(image_paths)

# def list_images_under_prefix(bucket_name: str, prefix: str):
#     keys = []
#     paginator = s3.get_paginator("list_objects_v2")
#     try:
#         for page in paginator.paginate(Bucket=bucket_name, Prefix=prefix):
#             for obj in page.get("Contents", []):
#                 k = obj["Key"]
#                 if k.lower().endswith(IMAGE_EXTS):
#                     keys.append(k)
#     except ClientError as e:
#         # optional: log or handle errors (throttling/network)
#         print(f"Error listing {prefix}: {e}")
#     return keys

# def get_image_paths_from_s3_parallel(bucket_name: str, root_prefix: str = "", max_workers: int = 4):
#     paginator = s3.get_paginator("list_objects_v2")
#     allowed_prefixes = []
#     image_paths = []

#     # get top-level prefixes using Delimiter to avoid excluded folders
#     for page in paginator.paginate(Bucket=bucket_name, Prefix=root_prefix, Delimiter="/"):
#         for obj in page.get("Contents", []):
#             key = obj["Key"]
#             if any(key.startswith(excl) for excl in EXCLUDED_TOP_LEVEL):
#                 continue
#             if key.lower().endswith(IMAGE_EXTS):
#                 image_paths.append(key)
#         for cp in page.get("CommonPrefixes", []):
#             prefix = cp["Prefix"]
#             if prefix in EXCLUDED_TOP_LEVEL:
#                 continue
#             allowed_prefixes.append(prefix)

#     # parallelize listing of allowed prefixes
#     with ThreadPoolExecutor(max_workers=max_workers) as ex:
#         futures = {ex.submit(list_images_under_prefix, bucket_name, p): p for p in allowed_prefixes}
#         for fut in as_completed(futures):
#             image_paths.extend(fut.result())

#     return sorted(image_paths)




def delete_images_from_s3(bucket_name: str, keys, verbose: bool = True):
    """
    Delete the specified object keys from the given S3 bucket.

    Args:
        bucket_name: name of the S3 bucket (e.g. "alyaimg").
        keys: list of object keys to delete (e.g. ["SKU-107.jpg", "a.jpg", ...]).
        verbose: if True, prints progress and errors.
    """
    # S3 delete_objects can handle up to 1000 keys per call
    BATCH_SIZE = 1000
    total = len(keys)
    for i in range(0, total, BATCH_SIZE):
        batch = keys[i : i + BATCH_SIZE]
        delete_payload = {"Objects": [{"Key": k} for k in batch]}
        try:
            response = s3.delete_objects(Bucket=bucket_name, Delete=delete_payload)
            deleted = response.get("Deleted", [])
            errors = response.get("Errors", [])
            if verbose:
                print(f"Batch {i//BATCH_SIZE + 1}: requested {len(batch)} deletions, succeeded {len(deleted)}.")
                if errors:
                    print("Errors:")
                    for err in errors:
                        print(f"  - {err['Key']}: {err['Message']}")
        except Exception as e:
            print(f"Failed to delete batch starting at index {i}: {e}")

def hash_image_from_s3(bucket: str, key: str):
    obj = s3.get_object(Bucket=bucket, Key=key)
    data = obj['Body'].read()
    return hashlib.sha256(data).hexdigest()

def load_image_from_s3(bucket: str, key: str) -> Image.Image:
    obj = s3.get_object(Bucket=bucket, Key=key)
    return Image.open(BytesIO(obj['Body'].read())).convert("RGB")

def index_images_from_s3(
    collection,
    image_keys,
    model,
    processor,
    device,
    batch_size=128,
    index_file=INDEX_FILE,
    int_hash_file=INT_HASH_MAP_FILE
):
    indexed = {}
    int_hash_to_path = {}

    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            indexed = json.load(f)

    batch_ids, batch_embeddings = [], []
    current_hashes = {}
    new_embeddings_inserted = False

    for key in tqdm(image_keys, desc="Indexing S3 images"):
        try:
            h = hash_image_from_s3(S3_BUCKET, key)
            if h in indexed:
                continue
            int_h = hash_to_int64(h)
            current_hashes[h] = key

            encoded_key = urllib.parse.quote(key)
            s3_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{encoded_key}"
            int_hash_to_path[str(int_h)] = s3_url

            img = load_image_from_s3(S3_BUCKET, key)
            emb = embed_image(img, model, processor, device)

            batch_ids.append(int_h)
            batch_embeddings.append(emb.tolist())
        except Exception as e:
            print(f"Skipping {key}: {e}")
            continue

        if len(batch_ids) >= batch_size:
            ids_array = np.array(batch_ids, dtype=np.int64)
            emb_array = np.array(batch_embeddings, dtype=np.float32)
            entities = [
                {"id": i, "embedding": e}
                for i, e in zip(ids_array.tolist(), emb_array.tolist())
            ]
            collection.insert(entities)
            new_embeddings_inserted = True
            batch_ids, batch_embeddings = [], []

    if batch_ids:
        ids_array = np.array(batch_ids, dtype=np.int64)
        emb_array = np.array(batch_embeddings, dtype=np.float32)
        entities = [
            {"id": i, "embedding": e}
            for i, e in zip(ids_array.tolist(), emb_array.tolist())
        ]
        collection.insert(entities)
        new_embeddings_inserted = True

    deleted_hashes = set(indexed.keys()) - set(current_hashes.keys())
    if deleted_hashes:
        deleted_ids = [hash_to_int64(h) for h in deleted_hashes]
        expr = f"id in [{', '.join(map(str, deleted_ids))}]"
        try:
            collection.delete(expr=expr)
            new_embeddings_inserted = True
        except Exception as e:
            print(f"Error deleting stale embeddings: {e}")

    if new_embeddings_inserted:
        try:
            collection.flush()
            collection.load()
            print("✅ S3 collection flushed and reloaded.")
        except Exception as e:
            print(f"❌ Error during S3 flush/load: {e}")
    else:
        print("ℹ️ No changes to flush from S3.")

    with open(index_file, 'w') as f:
        json.dump(current_hashes, f)

    with open(int_hash_file, 'w') as f:
        json.dump(int_hash_to_path, f)

    print(f"✅ Indexed {len(current_hashes)} S3 images into Milvus.")



def get_image_paths(image_dir: str):
    paths = []
    for root, _, files in os.walk(image_dir):
        for fname in files:
            if fname.lower().endswith((".jpg", ".jpeg", ".png", ".webp")):
                paths.append(os.path.join(root, fname))
    return sorted(paths)


def hash_image(path):
    with open(path, 'rb') as f:
        return hashlib.sha256(f.read()).hexdigest()

def hash_to_int64(hash_str: str) -> int:
    unsigned = int(hash_str[:16], 16)
    return unsigned & 0x7FFFFFFFFFFFFFFF


def init_clip(device: str = None):
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
    return model, processor, device

def embed_image(img: Image.Image, model, processor, device) -> np.ndarray:
    """Preprocess and embed an image using CLIP."""
    inputs = processor(images=img, return_tensors="pt").to(device)
    with torch.no_grad():
        features = model.get_image_features(**inputs)
    return features.cpu().numpy().reshape(-1)

# ------------------------ Connect to Milvus ------------------------
def init_milvus(host: str, port: str, collection_name: str, dim: int = 512):
    #connections.connect("default", host=host, port=port)
    connections.connect(
    alias="default",
    uri=URL,
    token=TOKEN,
    db_name="default"      # or your custom database
    )
    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=dim)
    ]
    schema = CollectionSchema(fields, description="Image embeddings collection")
    if collection_name in utility.list_collections():
        col = Collection(name=collection_name)
    else:
        col = Collection(name=collection_name, schema=schema)
    if not any(index.field_name == "embedding" for index in col.indexes):
        col.create_index(
            field_name="embedding",
            index_params={
                "index_type": "HNSW",
                "params": {"M": 48, "efConstruction": 200},
                "metric_type": "L2"
            }
        )

    col.load()
    return col

def index_images(collection, image_paths, model, processor, device, batch_size=128, index_file=INDEX_FILE, int_hash_file=INT_HASH_MAP_FILE):
    indexed = {}
    int_hash_to_path = {}

    if os.path.exists(index_file):
        with open(index_file, 'r') as f:
            indexed = json.load(f)

    batch_ids, batch_embeddings = [], []
    current_hashes = {}

    new_embeddings_inserted = False

    # Compute hashes for current images
    for path in tqdm(image_paths, desc="Indexing images"):
        try:
            h = hash_image(path)
            int_h = hash_to_int64(h)
            current_hashes[h] = path
            int_hash_to_path[str(int_h)] = path
            if h in indexed:
                continue

            img = Image.open(path).convert("RGB")
            emb = embed_image(img, model, processor, device)

            batch_ids.append(int_h)
            batch_embeddings.append(emb.tolist())
        except Exception as e:
            print(f"Skipping {path}: {e}")
            continue

        if len(batch_ids) >= batch_size:
            ids_array = np.array(batch_ids, dtype=np.int64)
            emb_array = np.array(batch_embeddings, dtype=np.float32)
            ids_py = [int(x) for x in ids_array.tolist()]
            emb_py = emb_array.tolist()
            entities = [
                {"id": i, "embedding": e}
                for i, e in zip(ids_py, emb_py)
            ]
            collection.insert(entities)
            new_embeddings_inserted = False
            batch_ids, batch_embeddings = [], []

    if batch_ids:
        ids_array = np.array(batch_ids, dtype=np.int64)
        emb_array = np.array(batch_embeddings, dtype=np.float32)
        ids_py = [int(x) for x in ids_array.tolist()]
        emb_py = emb_array.tolist()
        entities = [
                {"id": i, "embedding": e}
                for i, e in zip(ids_py, emb_py)
            ]
        collection.insert(entities)
        new_embeddings_inserted = False

    # Identify deleted hashes
    deleted_hashes = set(indexed.keys()) - set(current_hashes.keys())
    if deleted_hashes:
        deleted_ids = [hash_to_int64(h) for h in deleted_hashes]
        expr = f"id in [{', '.join(map(str, deleted_ids))}]"
        print(expr)
        collection.delete(expr=expr)
        print(f"\U0001f5d1️ Deleted {len(deleted_hashes)} stale embeddings from Milvus.")

    if new_embeddings_inserted:
        try:
            collection.flush()
            collection.load()
            print("✅ Collection flushed and reloaded.")
        except Exception as e:
            print(f"❌ Error during flush/load: {e}")
    else:
        print("ℹ️ No changes to flush; collection unchanged.")

    with open(index_file, 'w') as f:
        json.dump(current_hashes, f)

    with open(int_hash_file, 'w') as f:
        json.dump(int_hash_to_path, f)

    print(f"✅ Indexed {len(current_hashes)} images into Milvus.")


# ------------------------ Search Function ------------------------
def search_similar(collection, query_emb, top_k: int = 5):
    collection.load()
    results = collection.search(
        [query_emb.tolist()], "embedding", {"metric_type": "L2"}, limit=top_k
    )
    return [(int(hit.id), float(hit.distance)) for hit in results[0]]


# def main_index():
#     image_dir = os.getenv("LOCAL_IMAGE_DIR", "./images")
#     host = os.getenv("MILVUS_HOST", "localhost")
#     port = os.getenv("MILVUS_PORT", "19530")
#     coll_name = os.getenv("COLLECTION_NAME", "image_embeddings")
#     batch_size = int(os.getenv("BATCH_SIZE", "128"))
#     model, processor, device = init_clip()
#     collection = init_milvus(host, port, coll_name)
#     print("hello")
#     paths = get_image_paths_from_s3(bucket_name=S3_BUCKET)
#     print(paths)
#     index_images_from_s3(collection, paths, model, processor, device, batch_size)

def main_index():
    # paths = get_image_paths_from_s3(bucket_name=S3_BUCKET)
    # print(paths)
    # paths = ['A40.jpg']
    # delete_images_from_s3(S3_BUCKET, paths)
    start = time.time()
    paths = get_image_paths_from_s3(bucket_name=S3_BUCKET)
    end = time.time()
    print(end - start)
    print(paths)

if __name__ == "__main__":
    main_index()