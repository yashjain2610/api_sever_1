import streamlit as st
import os
import base64

import google.generativeai as genai

from dotenv import load_dotenv

load_dotenv()

genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))



def get_gemini_responses(input, image, prompts):
    model = genai.GenerativeModel('gemini-1.5-flash',
                                  generation_config=genai.types.GenerationConfig(
                                      temperature=0.5,
                                      top_p=0.9,
                                      top_k=40,
                                      max_output_tokens=1024
                                  ))
    all_responses = []
    for prompt in prompts:
        response = model.generate_content([input, image[0], prompt])
        for part in response.parts:
            if part.text:
                all_responses.append(part.text)
    #print(all_responses)
    return all_responses

def get_gemini_dims_responses(input, image, prompts):
    model = genai.GenerativeModel('gemini-1.5-flash',
                                  generation_config=genai.types.GenerationConfig(
                                      temperature=0.5,
                                      top_p=0.9,
                                      top_k=40,
                                      max_output_tokens=512
                                  ))
    all_responses = []
    for prompt in prompts:
        response = model.generate_content([input, image[0], prompt])
        # print(response)
        for part in response.parts:
            if part.text:
                all_responses.append(part.text)
    # print(all_responses)
    return all_responses[0]

# def input_image_setup(uploaded_file):
#     if uploaded_file is not None:
#         bytes_data = uploaded_file.getvalue()
#         image_parts = [{"mime_type": uploaded_file.type, "data": bytes_data}]
#         return image_parts
#     else:
#         raise FileNotFoundError("No file uploaded")
    
def input_image_setup(file_bytes_io, mime_type):
    if file_bytes_io is not None:
        bytes_data = file_bytes_io.getvalue()
        image_parts = [{"mime_type": mime_type, "data": bytes_data}]
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