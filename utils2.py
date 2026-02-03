import streamlit as st
import os
import io
import base64
import openpyxl
from prompts import *
import random
from PIL import Image, ImageOps
from rembg import remove
from io import BytesIO
from google import genai
from google.genai import types
from google.api_core import exceptions as google_exceptions
import itertools
import grpc
from openai import OpenAI

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


def generate_images_from_gpt(
    image: Image.Image,
    prompts: list[str],
    size: str = "1024x1024",
    use_composite: bool = True
):
    """
    Runs a single image with multiple prompts using gpt-image-1
    and returns structured responses.

    Uses COMPOSITE approach for 100% jewelry preservation:
    1. Extract original jewelry using rembg
    2. Generate new background/scene with OpenAI
    3. Composite original jewelry onto generated background

    Args:
        image: PIL Image to edit
        prompts: List of prompts for each variation
        size: Output image size
        use_composite: If True, composite original jewelry onto generated background (100% preservation)
    """
    all_responses = []

    # Extract original jewelry with transparent background (for compositing later)
    jewelry_rgba = None
    if use_composite:
        jewelry_rgba = extract_jewelry(image)

    # Save image to temp buffer
    img_buffer = io.BytesIO()
    image.save(img_buffer, format="PNG")
    img_buffer.seek(0)

    for prompt in prompts:
        # Build API call parameters
        api_params = {
            "model": "gpt-image-1",
            "image": ("image.png", img_buffer, "image/png"),
            "prompt": prompt,
            "size": size,
            "quality": "low"
        }

        result = client.images.edit(**api_params)

        structured_response = {
            "prompt": prompt,
            "images": []
        }

        for img in result.data:
            img_bytes = base64.b64decode(img.b64_json)

            if use_composite and jewelry_rgba is not None:
                # Open the generated image (this has AI-generated background + AI jewelry)
                generated_img = Image.open(io.BytesIO(img_bytes))

                # Composite: replace AI jewelry with ORIGINAL jewelry
                final_img = composite_jewelry_on_background(jewelry_rgba, generated_img)

                # Convert back to bytes
                final_buffer = io.BytesIO()
                final_img.save(final_buffer, format="PNG")
                img_bytes = final_buffer.getvalue()

            structured_response["images"].append(img_bytes)

        all_responses.append(structured_response)

        # Reset buffer pointer for next loop
        img_buffer.seek(0)

    return all_responses
