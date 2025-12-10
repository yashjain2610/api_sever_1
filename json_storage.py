import json
from pathlib import Path

JSON_PATH = Path("data_store.json")

def compress_bullet_points(data: dict):
    """
    Convert keys like bullet_point_1, bullet_point_2 ... into
    a single field:
    { "bullet_points": { "bullet_point_1": "", ... } }
    """
    bullet_keys = {k: v for k, v in data.items() if k.startswith("bullet_point")}

    if not bullet_keys:
        return data

    # Remove old bullet_point fields
    for k in bullet_keys:
        data.pop(k)

    # Insert compressed structure
    data["bullet_points"] = bullet_keys
    return data


def expand_bullet_points(data: dict):
    """
    Convert compressed form:
    { "bullet_points": { "bullet_point_1": "", ... } }
    back into flat dict keys:
    { "bullet_point_1": "", "bullet_point_2": "" }
    """
    bp = data.get("bullet_points")
    if not isinstance(bp, dict):
        return data

    # Add flattened bullet points
    for k, v in bp.items():
        data[k] = v

    # Remove compressed key
    data.pop("bullet_points", None)

    return data



def store_data_for_sku(sku_id: str, data: dict):
    # Load existing file or create new structure
    if JSON_PATH.exists():
        with open(JSON_PATH, "r", encoding="utf-8") as f:
            db = json.load(f)
    else:
        db = {}

    # Compress bullet points
    data = compress_bullet_points(data)

        # Merge incoming data with existing data (do not delete existing fields)
    if sku_id not in db:
        db[sku_id] = {}

    # Update only fields that are coming in
    for key, value in data.items():
        db[sku_id][key] = value

    # Save back to file
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(db, f, indent=4, ensure_ascii=False)

def fetch_data_for_sku(sku_id: str):
    # If file doesn't exist
    print("here")
    sku_id = str(sku_id)
    if not JSON_PATH.exists():
        return {
            "sku_exists": False,
            "title": False,
            "bullet_points": False,
            "description": False,
        }
    print("here")
    with open(JSON_PATH, "r", encoding="utf-8") as f:
        db = json.load(f)

    print(db)
    print(sku_id in db)

    # Base response
    result = {
        "sku_exists": sku_id in db,
        "title": False,
        "bullet_points": False,
        "description": False,
    }

    # If SKU missing
    if not result["sku_exists"]:
        return result

    data = db[sku_id]

    # Return actual values or False
    result["title"] = data.get("title", False)
    result["bullet_points"] = data.get("bullet_points", False)
    result["description"] = data.get("description", False)

    return result