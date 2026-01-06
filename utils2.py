import streamlit as st
import os
import io
import base64
import openpyxl
from prompts import *
import random
from PIL import Image , ImageOps
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



def generate_images_from_gpt(
    image: Image.Image,
    prompts: list[str],
    size: str = "1024x1024"
):
    """
    Runs a single image with multiple prompts using gpt-image-1
    and returns structured responses.
    """

    base64_image = pil_to_base64(image)
    all_responses = []

    for prompt in prompts:
        result = client.images.generate(
            model="gpt-image-1",
            prompt=prompt,
            image=base64_image,
            size=size,
            quality="low"
        )

        structured_response = {
            "prompt": prompt,
            "text": "",
            "images": []
        }

        for img in result.data:
            img_bytes = base64.b64decode(img.b64_json)
            structured_response["images"].append(img_bytes)

        all_responses.append(structured_response)

    return all_responses