import streamlit as st
import os
import io
import base64
import openpyxl
from prompts import *
import random
import time
from PIL import Image, ImageOps
from rembg import remove
from io import BytesIO
from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions
import itertools
import grpc
from openai import OpenAI
from concurrent.futures import ThreadPoolExecutor, as_completed

from dotenv import load_dotenv

load_dotenv()

API_KEY = os.getenv("API_KEY_GPT")
proj_id = os.getenv("PROJ_ID")
client = OpenAI(
    api_key=API_KEY,
    project=proj_id
    )



API_KEYS = [
    os.getenv("GOOGLE_API_KEY"),
    os.getenv("GOOGLE_API_KEY2"),
    os.getenv("GOOGLE_API_KEY3"),
    os.getenv("GOOGLE_API_KEY4"),
]

#print(API_KEYS)
clients = [genai.Client(api_key=key) for key in API_KEYS]
client_cycle = itertools.cycle(clients)


def pil_to_base64(image: Image.Image) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("utf-8")



def _is_quota_exceeded(err) -> bool:
    # HTTP‐style 429
    if getattr(err, 'code', None) == 429:
        return True

    # grpc.RpcError
    if isinstance(err, grpc.RpcError) and err.code() == grpc.StatusCode.RESOURCE_EXHAUSTED:
        return True

    # Google API exception
    if isinstance(err, google_exceptions.ResourceExhausted):
        return True

    # Fallback: look for the string in the message
    msg = str(err).upper()
    if 'RESOURCE_EXHAUSTED' in msg or 'QUOTA' in msg:
        return True

    return False

def generate_with_failover(model: str, contents: list, config: types.GenerateContentConfig):
    last_exc = None

    for _ in range(len(clients)):
        client = next(client_cycle)
        try:
            return client.models.generate_content(
                model=model,
                contents=contents,
                config=config
            )
        except Exception as e:
            if _is_quota_exceeded(e):
                last_exc = e
                # optionally log which key failed here
                continue
            # if it wasn’t a quota problem, bail out
            raise

    # all keys hit quota
    raise last_exc

def get_random_model_image(folder_path="model images"):
    # List only image files with valid extensions
    valid_extensions = (".jpg", ".jpeg", ".png")
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]

    if not image_files:
        raise FileNotFoundError(f"No image files found in folder: {folder_path}")
    
    # Pick a random image
    chosen_file = random.choice(image_files)
    image_path = os.path.join(folder_path, chosen_file)

    # Open and return the image
    return Image.open(image_path)  # returning filename is optional but helpful

def get_all_model_images(folder_path="model images"):
    valid_extensions = (".jpg", ".jpeg", ".png")
    image_files = [f for f in os.listdir(folder_path) if f.lower().endswith(valid_extensions)]

    if not image_files:
        raise FileNotFoundError(f"No image files found in folder: {folder_path}")

    images = []
    for file in image_files:
        image_path = os.path.join(folder_path, file)
        images.append(Image.open(image_path))
    return images


def resize_img2(image):
# Make image square by padding (optional: crop instead)
    img_square = ImageOps.pad(image, (max(image.size), max(image.size)), color=(255, 255, 255))

# Resize to 2000x2000
    img_resized = img_square.resize((1080, 1440), Image.Resampling.LANCZOS)

# Save the result
    return img_resized

def resize_img(image):
    img_square = ImageOps.pad(image, (max(image.size), max(image.size)), color=(255, 255, 255))

# Resize to 2000x2000
    img_resized = img_square.resize((2000, 2000), Image.Resampling.LANCZOS)

# Save the result
    return img_resized

def get_gemini_responses(input_text, image, prompts):


    all_responses = []

    for prompt in prompts:
        content = [prompt, image]
        # if prompt == models_prompt:
        #     base_model_image = get_random_model_image("model images")
        #     #model_image = get_all_model_images("sample output model")
        #     content = [prompt] + [base_model_image] + [image]

        try:
            response = generate_with_failover(
                model="gemini-2.5-flash-image",
                contents=content,
                config=types.GenerateContentConfig(
                    response_modalities=['TEXT', 'IMAGE']
                )
            )
        except Exception as e:
            st.error(f"All API-keys exhausted or fatal error: {e}")
            break

        structured_response = {
            "prompt": prompt,
            "text": "",
            "images": []
        }

        for part in response.candidates[0].content.parts:
            if hasattr(part, "text") and part.text:
                structured_response["text"] += part.text.strip()
            elif hasattr(part, "inline_data") and part.inline_data:
                structured_response["images"].append(part.inline_data.data)

        all_responses.append(structured_response)

    return all_responses


def input_image_setup(uploaded_file):
    if uploaded_file is not None:
        bytes_data = uploaded_file.getvalue()
        image_parts = [{"mime_type": uploaded_file.type, "data": bytes_data}]
        return image_parts
    else:
        raise FileNotFoundError("No file uploaded")
    
def input_image_setup_local(image_path):
    with open(image_path, "rb") as img_file:
        bytes_data = img_file.read()
        mime_type = "image/jpeg" if image_path.endswith(".jpg") or image_path.endswith(".jpeg") else "image/png"
        image_parts = [{"mime_type": mime_type, "data": bytes_data}]
        return image_parts

def encode_image(uploaded_image):
    img_bytes = uploaded_image.read()
    return base64.b64encode(img_bytes).decode("utf-8")



def extract_jewelry(image: Image.Image) -> Image.Image:
    """
    Extract jewelry from image with transparent background using rembg.

    Args:
        image: PIL Image of jewelry with any background

    Returns:
        PIL Image of jewelry with transparent background (RGBA)
    """
    no_bg = remove(image)

    if no_bg.mode != 'RGBA':
        no_bg = no_bg.convert('RGBA')

    return no_bg


def composite_jewelry_on_background(
    jewelry_rgba: Image.Image,
    background: Image.Image
) -> Image.Image:
    """
    Composite the original jewelry onto a generated background.

    Args:
        jewelry_rgba: Jewelry image with transparent background (RGBA)
        background: Generated background image

    Returns:
        Final composited image with original jewelry on new background
    """
    # Ensure background is RGBA
    if background.mode != 'RGBA':
        background = background.convert('RGBA')

    # Resize jewelry to match background if needed
    if jewelry_rgba.size != background.size:
        jewelry_rgba = jewelry_rgba.resize(background.size, Image.Resampling.LANCZOS)

    # Composite: paste jewelry on top of background using jewelry's alpha as mask
    result = background.copy()
    result.paste(jewelry_rgba, (0, 0), jewelry_rgba)

    return result


def _generate_single_image(args):
    """
    Helper function to generate a single image (used for parallel execution).
    Includes retry logic for rate limit (429) errors.
    """
    prompt, img_bytes, size, prompt_index = args

    total_start = time.time()
    print(f"[IMAGE {prompt_index}] Started at {time.strftime('%H:%M:%S')}")

    # Retry settings for rate limit errors
    max_retries = 3
    base_delay = 15  # seconds

    for attempt in range(max_retries):
        # Step 1: Create buffer (must recreate for each attempt)
        buffer_start = time.time()
        img_buffer = io.BytesIO(img_bytes)
        buffer_time = time.time() - buffer_start
        if attempt == 0:
            print(f"[IMAGE {prompt_index}] Buffer creation: {buffer_time:.2f}s")

        # Step 2: Prepare API params
        api_params = {
            "model": "gpt-image-1.5",
            "image": ("image.png", img_buffer, "image/png"),
            "prompt": prompt,
            "size": size,
            "quality": "low",
            "n": 1
        }

        # Step 3: API call (upload + processing + download) with retry
        api_start = time.time()
        try:
            result = client.images.edit(**api_params)
            api_time = time.time() - api_start
            print(f"[IMAGE {prompt_index}] API call (upload+process+download): {api_time:.2f}s")
            break  # Success, exit retry loop
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate_limit" in error_str.lower():
                delay = base_delay * (attempt + 1)  # 15s, 30s, 45s
                print(f"[IMAGE {prompt_index}] Rate limit hit (attempt {attempt + 1}/{max_retries}), waiting {delay}s...")
                time.sleep(delay)
                if attempt == max_retries - 1:
                    raise  # Re-raise on final attempt
            else:
                raise  # Non-rate-limit error, don't retry

    # Step 4: Decode response
    decode_start = time.time()
    structured_response = {
        "prompt": prompt,
        "prompt_index": prompt_index,
        "images": []
    }

    for img in result.data:
        img_bytes_result = base64.b64decode(img.b64_json)
        structured_response["images"].append(img_bytes_result)
    decode_time = time.time() - decode_start
    print(f"[IMAGE {prompt_index}] Decode response: {decode_time:.2f}s")

    total_time = time.time() - total_start
    print(f"[IMAGE {prompt_index}] TOTAL: {total_time:.2f}s")
    print(f"[IMAGE {prompt_index}] Finished at {time.strftime('%H:%M:%S')}")
    print("-" * 50)

    # Add timing info for efficiency calculation
    structured_response["_generation_time"] = total_time

    return structured_response


def generate_images_from_gpt(
    image: Image.Image,
    prompts: list[str],
    size: str = "1024x1024"
):
    """
    Runs a single image with multiple prompts using gpt-image-1.5
    and returns structured responses.

    Uses PARALLEL execution for ~4x faster generation.

    Args:
        image: PIL Image to edit
        prompts: List of prompts for each variation
        size: Output image size
    """
    overall_start = time.time()
    print("=" * 60)
    print(f"[BATCH] Starting generation of {len(prompts)} images")
    print(f"[BATCH] Start time: {time.strftime('%H:%M:%S')}")
    print("=" * 60)

    # Save image to bytes (will be shared across threads)
    prep_start = time.time()
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    img_bytes = img_buffer.getvalue()
    prep_time = time.time() - prep_start
    print(f"[BATCH] Image preparation: {prep_time:.2f}s (size: {len(img_bytes)/1024:.1f} KB)")

    # Prepare arguments for parallel execution
    args_list = [
        (prompt, img_bytes, size, idx)
        for idx, prompt in enumerate(prompts)
    ]

    all_responses = [None] * len(prompts)  # Pre-allocate to maintain order

    # Execute in parallel with ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {
            executor.submit(_generate_single_image, args): args[3]  # Map future to prompt_index
            for args in args_list
        }

        for future in as_completed(futures):
            prompt_index = futures[future]
            try:
                result = future.result()
                all_responses[prompt_index] = result
            except Exception as e:
                print(f"[ERROR] Failed to generate image {prompt_index}: {str(e)}")
                all_responses[prompt_index] = {
                    "prompt": prompts[prompt_index],
                    "prompt_index": prompt_index,
                    "images": [],
                    "error": str(e)
                }

    # Final summary
    overall_time = time.time() - overall_start

    # Calculate sequential time (sum of all individual times)
    sequential_time = sum(
        r.get("_generation_time", 0)
        for r in all_responses
        if r and "_generation_time" in r
    )

    # Calculate efficiency: sequential_time / parallel_time
    efficiency = sequential_time / overall_time if overall_time > 0 else 1.0

    print("=" * 60)
    print(f"[BATCH] All {len(prompts)} images completed")
    print(f"[BATCH] End time: {time.strftime('%H:%M:%S')}")
    print(f"[BATCH] TOTAL BATCH TIME: {overall_time:.2f}s")
    print(f"[BATCH] Sequential would be: {sequential_time:.2f}s")
    print(f"[BATCH] Parallelization speedup: {efficiency:.1f}x")
    print("=" * 60)

    return all_responses


def generate_earring_catalog_image(
    image: Image.Image,
    image_type: int,
    height: str = None,
    width: str = None,
    size: str = "1024x1024"
):
    """
    Generate a specific catalog image type for Amazon earrings.

    Args:
        image: PIL Image of the earring
        image_type: Integer 1-5 representing the catalog image type:
            1 - White Background (main product image)
            2 - Lifestyle Background 1 (pastel gradient)
            3 - Dimension Image (with height/width markings)
            4 - Lifestyle Background 2 (moody/editorial)
            5 - Model Wearing (AI-generated model)
        height: Height dimension (e.g., "2.5 cm") - required for type 3
        width: Width dimension (e.g., "1.8 cm") - required for type 3
        size: Output image size

    Returns:
        dict with prompt and generated image bytes
    """
    from prompts2 import get_earring_prompt, EARRING_IMAGE_TYPES

    # Validate image type
    if image_type not in EARRING_IMAGE_TYPES:
        raise ValueError(f"Invalid image type: {image_type}. Must be 1-5.")

    # Get the prompt
    prompt = get_earring_prompt(image_type, height, width)
    config = EARRING_IMAGE_TYPES[image_type]

    # Save image to temp buffer
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    # Build API call parameters
    api_params = {
        "model": "gpt-image-1.5",
        "image": ("image.png", img_buffer, "image/png"),
        "prompt": prompt,
        "size": size,
        "quality": "medium",
        "n": 1
    }

    result = client.images.edit(**api_params)

    response = {
        "image_type": image_type,
        "image_type_name": config["name"],
        "prompt": prompt,
        "images": []
    }

    for img in result.data:
        img_bytes = base64.b64decode(img.b64_json)
        response["images"].append(img_bytes)

    return response


def print_earring_image_types():
    """Print available earring catalog image types."""
    from prompts2 import list_earring_image_types
    list_earring_image_types()
