from fastapi import FastAPI, File, UploadFile, Form,Request,HTTPException,APIRouter
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse,StreamingResponse
import csv
import io
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
from google.generativeai.types import HarmCategory, HarmBlockThreshold
import json
import google.generativeai as genai
from typing import List, Dict, Any, Optional
import requests
import logging
import uuid
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from datetime import date,datetime,timezone
import uvicorn
from PIL import Image
from io import BytesIO
from pymilvus import Collection
from typing import List
from utils import *
from prompts import *
from excel_fields import *
import httpx
import tempfile
import re
from prompts2 import *
from utils2 import get_gemini_responses as gen_image_responses
from utils2 import resize_img,resize_img2,generate_images_from_gpt
from urllib.parse import urlparse
import zipfile
import time
from openai_utils import *
from json_storage import *

app = FastAPI()
load_dotenv()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Streamlit app's URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.mount("/static", StaticFiles(directory="static"), name="static")


API_KEY = os.getenv("GOOGLE_API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(model_name="gemini-2.0-flash")

S3_BUCKET = os.getenv("S3_BUCKET", "alyaimg")
GENERATED_FOLDER = "gen_images/"

# Dual bucket configuration
BUCKET_MAP = {
    "alya": "alyaimg",
    "blysk": "blyskimg"
}

from image_search_engine import *

# Load model and DB once
clipmodel, processor, device = init_clip()
collection_db = init_milvus(
    host=os.getenv("MILVUS_HOST", "localhost"),
    port=os.getenv("MILVUS_PORT", "19530"),
    collection_name=os.getenv("COLLECTION_NAME", "image_embeddings_ip")
)

# Flag to control whether to run full S3 indexing on startup
REINDEX_ON_STARTUP = os.getenv("REINDEX_ON_STARTUP", "false").lower() == "true"
# Flag to clean orphan entries from Milvus (fixes "unknown" path issue)
CLEAN_ORPHANS_ON_STARTUP = os.getenv("CLEAN_ORPHANS_ON_STARTUP", "false").lower() == "true"

@app.on_event("startup")
async def startup_event():
    """Run cleanup and/or indexing on startup based on env flags"""
    # First, clean orphan entries if enabled
    if CLEAN_ORPHANS_ON_STARTUP:
        print("Cleaning orphan entries from Milvus...")
        clean_orphan_entries(collection_db)
        print("Orphan cleanup complete.")

    # Then, run full re-index if enabled
    if REINDEX_ON_STARTUP:
        print("Starting full S3 image indexing...")
        paths = get_image_paths_from_s3(S3_BUCKET)
        index_images_from_s3(collection_db, paths, clipmodel, processor, device)
        print("Startup indexing complete.")

    if not CLEAN_ORPHANS_ON_STARTUP and not REINDEX_ON_STARTUP:
        print("Skipping startup tasks (set CLEAN_ORPHANS_ON_STARTUP=true or REINDEX_ON_STARTUP=true to enable)")

gl=None


mapp={
    "Elegence": "Jewelry that remains timeless, characterized by balanced designs, clean lines, and subtle ornamentation, ideal for formal or everyday wear.",
    "Simplicity": "Pieces designed with less complexity, focusing on form and functionality while maintaining a sophisticated charm.",
    "Heritage": "Designs that evoke a sense of nostalgia or cultural richness, often featuring intricate engravings or traditional motifs.",
    "Refinement": "Jewelry with flawless finishes and a harmonious design language, offering a versatile appeal for all occasions.",
    "Powerful": "Jewelry designed to stand out, featuring large, prominent forms and striking shapes that draw immediate attention.",
    "Vibrant": "Pieces with lively colors and playful designs, embodying energy and joy, perfect for casual or festive occasions.",
    "Geometric": "Jewelry inspired by bold geometric forms, including triangles, squares, and abstract shapes, often exuding a contemporary and urban vibe.",
    "Oversized": "Bigger pieces that dominate the look, ideal for making a fashion-forward impact or enhancing simpler outfits.",
    "Rustic": "Jewelry inspired by nature's imperfections, showcasing raw finishes, earthy tones, and natural forms like bark or stone textures.",
    "Botanical": "Pieces with intricate designs inspired by flowers, leaves, and vines, evoking a sense of connection to the natural world.",
    "Grounded": "Jewelry featuring warm tones, neutral palettes, and organic forms, creating a calming and understated aesthetic.",
    "Textured": "Designs that focus on tactile surfaces like hammered metal, embossed patterns, or layered finishes, offering a sensory experience.",
    "Opulent": "Jewelry designed with grandeur in mind, often featuring gold finishes, intricate detailing, and an aura of exclusivity.",
    "Sparkling": "Pieces with reflective elements, like rhinestones or metallic accents, to capture and play with light.",
    "Golden": "Designs showcasing gold tones prominently, emphasizing the warmth and richness of this classic metal.",
    "Regal": "Jewelry inspired by royalty, often featuring elaborate designs with gemstones or intricate goldwork.",
    "Whimsical": "Jewelry with playful designs like charms, unusual motifs, or bright colors, often adding a sense of humor or whimsy to the wearer's look.",
    "Cheerful": "Energetic designs filled with color and movement, making them perfect for casual or summer looks.",
    "Quirky": "Offbeat and unconventional designs that appeal to those looking for distinctive and expressive pieces.",
    "Colorful": "Jewelry incorporating a range of vibrant hues, often in playful combinations, suitable for fun or celebratory occasions.",
    "Contemporary": "Pieces that align with current design trends while maintaining a clean, streamlined aesthetic.",
    "Minimalist": "Jewelry that eliminates unnecessary details, focusing on core shapes and functionality for a modern, understated look.",
    "Edgy": "Designs with an avant-garde or rebellious vibe, often incorporating asymmetry, spikes, or mixed metals.",
    "Matte-finished": "Jewelry with matte surfaces, offering a modern and sophisticated alternative to traditional shiny finishes.",
    "Textured Metal": "Jewelry that uses various texturing techniques to create unique surfaces, adding depth and character to metals like silver and gold.",
    "Minimal Metal": "Jewelry that focuses on pure metal forms with simple lines, ideal for those who prefer elegance without ornate detail.",
    "Gold-Plated": "Jewelry made from base metals with a layer of gold plating, offering the appearance of solid gold at a more affordable price point.",
    "Delicate": "Small, intricate pieces with fine detailing, perfect for accentuating a romantic or feminine style.",
    "Floral": "Jewelry that incorporates floral and botanical motifs, often in soft or pastel tones.",
    "Soft": "Designs with gentle curves, soft colors, and a sense of harmony, suitable for intimate or elegant settings.",
    "Festive": "Jewelry designed to shine during celebrations, featuring eye-catching elements like tassels, glitter, or bold patterns.",
    "Radiant": "Pieces that focus on reflecting light, often with faceted stones, metallic finishes, or clear embellishments.",
    "Abstract": "Jewelry that emphasizes creative, unconventional shapes and patterns, appealing to those with artistic tastes.",
    "Artisan": "Pieces made by skilled artisans, showcasing unique craftsmanship and individuality.",
    "Eclectic": "Jewelry that combines various styles, materials, or themes, often resulting in strikingly unique combinations.",
    "Mixed Metals": "Designs that blend silver, gold, rose gold, or other metals for a versatile, modern aesthetic.",
    "Antique-inspired": "Jewelry that captures the charm of bygone eras, often featuring intricate engravings and vintage-style design elements.",
    "Retro": "Pieces with mid-20th century influences, often with bold, graphic designs or playful patterns.",
    "Victorian": "Jewelry with intricate designs, often featuring floral patterns, filigree work, and vintage-inspired motifs from the Victorian era.",
    "Gothic": "Jewelry with a dark and dramatic vibe, often featuring black metals, skulls, or Victorian Gothic elements for a mysterious look.",
    "Bohemian": "Pieces inspired by boho-chic style, often incorporating natural stones, beads, and ethnic patterns.",
    "Ethnic": "Jewelry with strong cultural influences, often made by artisans using techniques passed down through generations.",
    "Indian-inspired": "Jewelry influenced by traditional Indian designs, featuring bold colors, intricate details, and spiritual symbolism, often crafted with metals.",
    "Avant-garde": "Jewelry that pushes boundaries and explores new, experimental designs, often combining new materials or techniques with an artistic approach.",
    "Futuristic": "Pieces inspired by modern technology, space, and futuristic aesthetics, featuring sleek forms, metallic finishes, and modern materials.",
    "Street Style": "Jewelry that reflects the spirit of street fashion, often incorporating elements like oversized chains, charms, and urban graphics.",
    "Minimalist Chic": "Jewelry that embraces the 'less is more' philosophy, focusing on understated elegance and clean lines with a modern aesthetic.",
    "Handcrafted": "Jewelry made by skilled artisans with a focus on originality and craftsmanship, each piece reflecting personal touches and attention to detail.",
    "Mixed Media": "Jewelry that incorporates multiple materials, like metal, stone, fabric, or resin, offering a blend of textures and styles.",
    "Innovative": "Pieces that explore new techniques, materials, or design philosophies to create something entirely fresh and unique.",
    "Customizable": "Jewelry designed to be personalized with names, dates, or other custom elements, offering a deeply personal connection.",
    "Healing": "Jewelry designed with spiritual or healing intentions, often incorporating stones or symbols believed to have metaphysical properties.",
    "Mystical": "Pieces with a mystical feel, often featuring celestial motifs, moon and star designs, or gemstones associated with spiritual practices.",
    "Astrological": "Jewelry inspired by astrology, with designs centered around zodiac signs, planets, stars, and constellations.",
    "Symbolic": "Jewelry that incorporates symbols with personal or spiritual meaning, like crosses, hearts, or symbols of protection and love.",
    "Oceanic": "Jewelry inspired by the ocean, featuring designs that evoke waves, coral, and sea creatures, often in soft blues and greens.",
    "Forest-inspired": "Designs that reflect the serenity of forests, often incorporating leaf motifs, wood textures, and green tones that evoke a peaceful ambiance.",
    "Celestial": "Jewelry inspired by the cosmos, often featuring stars, moons, and other celestial elements with sparkly stones and sleek, polished metals.",
    "Animal-inspired": "Jewelry inspired by animals, from delicate butterfly wings to fierce lion motifs, using textures and forms that reflect nature's creatures."
}




# Define the models
class Collection(BaseModel):
    jwellery_id: str
    generated_name: str
    description: str
    attributes: str
    color: str

class UserNeeds(BaseModel):
    collection_size: int
    color: str
    category: str
    category_description: List[str]
    target_audience: str
    manual_prompt: str
    start_price:int
    end_price:int
    start_date:date
    end_date:date
    extra_jwellery:List[str]
class InputData(BaseModel):
    user_needs: UserNeeds

class RequestModel(BaseModel):
    data: List[Dict[str, Any]]  # Each item can have a flexible structure
    columns: str
    manual: str

# Initialize FastAPI app and ChromaDB

client = chromadb.Client()
collection_name = "jewelry_collection"
collection = client.get_or_create_collection(name=collection_name)

# Load a sentence transformer for embedding generation
# Lightweight semantic embedding model

@app.post("/semantic_filter_jewelry/")
async def semantic_filter_jewelry(data: InputData):
    payload={
            "start_date":data.user_needs.start_date.isoformat(),
            "end_date":data.user_needs.end_date.isoformat(),
            "start_price":str(data.user_needs.start_price),
            "end_price":str(data.user_needs.end_price),
            "category":data.user_needs.category
            }
    response = requests.request("POST","https://staging.blyskjewels.com/api/get-all-products",data=payload).text
    for item in json.loads(response)["data"]:
    # Step 1: Populate ChromaDB with jewelry descriptions and metadata
        combined_text = f"Description: {item['description']}. " \
                        f"Attributes: {item['attributes']}. " \
                        f"Color: {item['color']}."
        result = genai.embed_content(
        model="models/text-embedding-004",
        content=combined_text
    )
        embedding = result['embedding']
      # Generate combined embedding
        collection.add(
            documents=[combined_text],
            embeddings=[embedding],
            metadatas=[{
                "jwellery_id": item['id'],
                "jwellery_name_generated": item['product_name'],
                "jwellery_attributes": item['attributes'],
                "jwellery_color": item['color']
            }],
            ids=[str(item['id'])]
        )
    
    # Step 2: Combine user query with their needs
    user_query = f"Category: {mapp[data.user_needs.category_description[0]]}. " \
                 f"Target Audience: {data.user_needs.target_audience}. " \
                 f"Preferred Color: {data.user_needs.color}. " \
                 f"Additional Notes: {data.user_needs.manual_prompt}."
    # Generate embedding for the query
    query_result = genai.embed_content(
        model="models/text-embedding-004",
        content=user_query
    )
    query_embedding =query_result['embedding']
    # Step 3: Perform semantic search in ChromaDB
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=data.user_needs.collection_size
    )
    
    # Step 4: Extract and return the top-k jewelry IDs
    if "ids" not in results or not results["ids"]:
        raise HTTPException(status_code=404, detail="No matching jewelry found")
    
    top_jewelry_ids = [id for id in results["ids"]]
    for i in data.user_needs.extra_jwellery:
        if i not in top_jewelry_ids:
            top_jewelry_ids[0].append(i)

    # try:
    #     client.delete_collection(name=collection_name)
    # except Exception as e:
    #     raise HTTPException(status_code=500, detail=f"Error deleting collection: {str(e)}")
    
    response = model.generate_content(["Create an eye catching description for the following details of collection in 3-4 lines. This collection will be used for social media promotions :-\n"+user_query+"\n Return only the description, no preambles or postambles."],
                                      safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,

    }
    )
    print(response)
    # Get the response data from Gemini
    gemini_response = response.text
    print(gemini_response)
    coll_title = model.generate_content([f"Create an eye-catching collection title for the following details. This collection will be used for social media promotions :-\n {user_query}\n Return only the  title. No preambles or postambles"],safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,

    }).text
    return {"top_jewelry_ids": top_jewelry_ids[0],
            "collection_description":gemini_response,
            "collection_title":coll_title
            }


def manage_undo_file(new_data, file_path='undo.json'):
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)
            print(data)
    else:
        data = []

    data.append(new_data)

    if len(data) > 10:
        data.pop(0)

    with open(file_path, 'w') as file:
        json.dump(data, file, indent=4)


@app.post('/further_req')
async def further_req(request:RequestModel):
    new_prompt = "Modify the above response based on the following criteria:\n"

    prompt=str(request.data)+"\n"+new_prompt+f"Only select the following keys: {request.columns}\n"+f"Select only those entries which follow: {request.manual}\nreturn the final response in json format. No preambles or postambles. Keep all strings in double quotes strictly"

    response = model.generate_content([ prompt],
                                      safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,

    }
    )
    print(response)
    # Get the response data from Gemini
    gemini_response = response.text
    print(gemini_response)
    if gemini_response[0]=="`":
        try:
            gemini_response=gemini_response[7:-4]
            my_dict = json.loads(gemini_response)
        except:
            gemini_response=gemini_response[7:-3]
            my_dict=json.loads(gemini_response)
    else:
        my_dict=json.loads(gemini_response)
    manage_undo_file(my_dict)
    return my_dict


@app.post("/process-image-and-prompt/")
async def process_image_and_prompt(image: UploadFile = File(...)):
    global gl
    prompt = """
    You are given an image of a bill or invoice. Extract all the product entries and any additional transaction details from it.

Output Format Instructions:
1. Return a single valid JSON array (i.e., response must start with '[' and end with ']').
2. Each product should be represented as one JSON object inside this list. The keys should be the column headers (e.g., "QTY", "DESCRIPTION", "UNIT PRICE", "DISCOUNT (%)", "TOTAL").
3. Each row in the table corresponds to one JSON object. Do not duplicate or infer rows that do not exist in the image.
4. All values must be strings enclosed in double quotes — even for numeric or currency values (e.g., "00001", "$100", "10").
5. Preserve all prefixes, postfixes, and symbols exactly as they appear (do not modify "$", "%", commas, etc.).
6. After all product JSONs, add exactly one final JSON object containing other invoice-related details (like buyer name, transaction ID, total amount, date, etc.).  
   - This JSON should only include non-empty, meaningful key-value pairs.  
   - If there are no such details, include an empty JSON `{}` as the last element.
7. The entire output must be a single JSON list — no explanations, extra text, or formatting outside the list.

Example structure (not actual values):
[
  {
    "QTY": "2",
    "DESCRIPTION": "Modern Ergonomic Office Chair",
    "UNIT PRICE": "$250.00",
    "DISCOUNT (%)": "10",
    "TOTAL": "$450.00"
  },
  {
    "QTY": "5",
    "DESCRIPTION": "Minimalist Wooden Desk",
    "UNIT PRICE": "$400.00",
    "DISCOUNT (%)": "5",
    "TOTAL": "$1900.00"
  },
  {
    "Invoice Number": "INV-0001",
    "Buyer Name": "John Doe",
    "Total Amount": "$2350.00",
    "Date": "2025-10-09"
  }
]

"""
    # Save the uploaded image to a file
    image_content = await image.read()
    image_name = f"{uuid.uuid4()}_{image.filename}"
    save_path = f"./{image_name}"
    with open(save_path, "wb") as f:
        f.write(image_content)
    # Send the image and prompt to the Gemini API
    # sample_file = genai.upload_file(path=save_path,
    #                         display_name=image_name)
    

    # # Prompt the model with text and the previously uploaded image.
    # response = model.generate_content([sample_file, prompt],safety_settings={
    #     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
    #     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
    #     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
    #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,

    # })

    image_data = input_image_setup(io.BytesIO(image_content), "image/jpeg")
    response_list = get_gemini_responses("Analyze this image carefully.", image_data, [prompt])
    response = response_list[0]

    print(response)
    # Get the response idata from Gemini
    gemini_response = response
    if gemini_response[0]=="`":
        gemini_response=gemini_response[7:-3]
    print(gemini_response)
    my_dict = json.loads(gemini_response)
    extras=my_dict[-1]
    os.remove(save_path)
    # Extract relevant data from the response (adjust based on your needs)
    to_return= {"table":my_dict[:-1],
            "extras":extras}
    manage_undo_file(to_return)
    return to_return



@app.get("/undo_invoice")
async def undo():
    file_path="undo.json"
    if os.path.exists(file_path):
        with open(file_path, 'r') as file:
            data = json.load(file)

        if data:
            data.pop()
            try:
              last_element=data[-1]
            except:
              return "no file to undo"
            with open(file_path, 'w') as file:
                json.dump(data, file, indent=4)
            return last_element
        else:
            return "No actions to undo."
    else:
        return "Undo file does not exist."


ALLOWED_EXTENSIONS = (".jpg", ".jpeg", ".png", ".webp")



@app.post("/generate_caption")
async def generate_caption(
    file: UploadFile = File(...),
    type: str = Form(...),
    website: str = Form("alya"),  # "alya" or "blysk" - select S3 bucket
    check_duplicate: str = Form("yes")  # "yes" or "no" - bypass duplicate detection
):

        """
        This endpoint takes a file and a type parameter and generates a caption for the uploaded image
        based on the type and the features of the image. The caption is generated by the Gemini AI model.
        The type parameter is used to determine the style of the caption. The style options are :-
        "Elegance": "Jewelry that embodies sophistication, adding a refined touch suitable for formal settings.",
        "Charm": "Pieces that convey a sense of warmth and appeal, making them endearing and delightful.",
        "Luxury": "Reflects an aura of exclusivity and high quality, perfect for those seeking something special.",
        "Modernity": "Captures contemporary styles, appealing to those who appreciate current trends and innovation.",
        "Timelessness": "Designs that maintain appeal across eras, suitable for those who value enduring style.",
        "Delicacy": "Jewelry with a gentle and subtle quality, often associated with a soft, graceful appearance.",
        "Boldness": "Makes a statement with a strong presence, ideal for those who prefer impactful designs.",
        "Romance": "Pieces that evoke a sense of love and sentimentality, often carrying a romantic or heartfelt appeal.",
        "Earthiness": "Connects to natural elements or grounded aesthetics, appealing to those with a down-to-earth style.",
        "Artistry": "Showcases creativity and craftsmanship, appealing to individuals who appreciate unique designs."

        The output is a JSON object with the following keys:
        - display_name: the name of the file that was uploaded
        - generated_name: a unique name for the image that is generated by the model
        - quality: the quality of the image, which is either A, B or C
        - description: a one-line description of the image that is generated by the model
        - attributes: a list of attributes of the image that are detected by the model
        - prompt: the prompt that was used to generate the description
        - color: the color of the image that is detected by the model
        - s3_url: the URL of the image on Amazon S3
        - duplicate: if the image is a duplicate of an existing image, this key will be present with the value "duplicate found"
        - website: "alya" (default) or "blysk" - select which S3 bucket to use
        - check_duplicate: "yes" (default) to check for duplicates, "no" to bypass duplicate detection

        If the image is not a duplicate, the output will not contain the "duplicate" key.
        """
        start = time.time()

        if not file:
            return {"error": "No files received"}

        # Get bucket based on website selection
        selected_bucket = BUCKET_MAP.get(website.lower(), S3_BUCKET)
        if website.lower() not in BUCKET_MAP:
            return JSONResponse(status_code=400, content={"error": f"Invalid website. Use 'alya' or 'blysk'"})

        image_bytes = await file.read()

        duplicate = False
        s3_url = ""
        original_name = ""

        # Only perform duplicate check if check_duplicate is "yes"
        if check_duplicate.lower() == "yes":
            paths = get_image_paths_from_s3(S3_BUCKET)

            # Step 1: Get query image
            image_s = Image.open(BytesIO(image_bytes)).convert("RGB")
            query_emb = embed_image(image_s, clipmodel, processor, device)

            # Step 2: Search similar
            results = search_similar(collection_db, query_emb, 1)

            # Step 3: Load ID-to-path map
            int_hash_map_file = os.getenv("INT_HASH_MAP_FILE", "int_hash_to_path.json")
            try:
                with open(int_hash_map_file, 'r') as f:
                    id_to_path = json.load(f)
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": f"Failed to load hash map: {str(e)}"})

            # Step 4: Build response
            matches = []
            for _id, dist in results:
                print(f"[generate_caption] IP distance: {dist} (similarity: {(1-dist)*100:.2f}%)")
                print()
                if dist < 0.05:  # IP metric: 0.05 = ~95% similarity
                    path = id_to_path.get(str(_id), "unknown")
                    matches.append({"id": _id, "distance": dist, "path": path})

            # Only treat as duplicate if we have a valid path (not "unknown")
            if matches and matches[0]["path"] != "unknown":
                duplicate = True
                s3_url = matches[0]["path"]
                # Extract S3 key from URL
                s3_key = s3_url.split("/")[-1]
                # Remove timestamp_uuid_ prefix if present
                parts = s3_key.split("_")
                if len(parts) > 2:
                    original_name = "_".join(parts[2:])
                else:
                    original_name = s3_key
        

        # typ=json.loads(style)
        image=file
        image_name = image.filename
        
        # Create the path to save the file using the filename   
        save_path = f"./{image_name}"
        
        # Save the uploaded image directly to the disk
        with open(save_path, "wb") as f:
            f.write(image_bytes)  # Write the image data to the file

        if not duplicate:
            try:
                timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
                # Get file extension from original filename
                file_ext = os.path.splitext(image_name)[1] or ".jpg"
                unique_name = f"{timestamp}{file_ext}"

                s3.upload_file(
                    Filename=save_path,
                    Bucket=selected_bucket,
                    Key=unique_name,
                    ExtraArgs={"ContentType": "image/jpeg"}
                )
                index_single_image_from_s3(collection_db, unique_name, clipmodel, processor, device, s3_bucket=selected_bucket)
                s3_url = f"https://{selected_bucket}.s3.amazonaws.com/{unique_name}"
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": f"Failed to upload to S3: {str(e)}"})
        
        if duplicate:
            return {
                "display_name": original_name,
                "s3_url": s3_url,
                "duplicate": "duplicate found"
            }
            
        end = time.time()
        print(end - start)
        print()


        # Now you can use `save_path` to upload the file to Gemini or process further
        # sample_file = genai.upload_file(path=save_path, display_name=image_name)
        # #ADD PREV PROMPT STYLES OF USER
        pp=f"""please generate a short description for the given jwellery : {type}. STRICTLY MAKE SURE THE JWELLERY IS {type}. Focus only on the jwellery features and not on the background. The description should be one or more of the following types :- 
                                            "Elegance": "Jewelry that embodies sophistication, adding a refined touch suitable for formal settings.",
                                            "Charm": "Pieces that convey a sense of warmth and appeal, making them endearing and delightful.",
                                            "Luxury": "Reflects an aura of exclusivity and high quality, perfect for those seeking something special.",
                                            "Modernity": "Captures contemporary styles, appealing to those who appreciate current trends and innovation.",
                                            "Timelessness": "Designs that maintain appeal across eras, suitable for those who value enduring style.",
                                            "Delicacy": "Jewelry with a gentle and subtle quality, often associated with a soft, graceful appearance.",
                                            "Boldness": "Makes a statement with a strong presence, ideal for those who prefer impactful designs.",
                                            "Romance": "Pieces that evoke a sense of love and sentimentality, often carrying a romantic or heartfelt appeal.",
                                            "Earthiness": "Connects to natural elements or grounded aesthetics, appealing to those with a down-to-earth style.",
                                            "Artistry": "Showcases creativity and craftsmanship, appealing to individuals who appreciate unique designs."

                                              Chose the most suitable type/s according to jwellery physical features. 
                                              Example of description :- crisscross diamonds resembling earthiness and artistry. (describe how jwellery features are resembling the style)
                                              Examples of attributes :- diamonds arranged in a criss-cross pattern".
                                              Also generate jwellery name.
                                              Also detect the letter A/B/C written on the image.
                                              Also detect the color of the jwellery.
                                              final_caption : one-line caption of jwellery for e-commerce listing, according to the generated description and attributes

                                              Final Output:-
                                              JSON with keys -> quality : (A/B/C), name : unique jwellery name ,description : generated description, attributes: jwellery physical features, color: color of the jwellery, final_caption: one-line caption of jwellery for e-commerce listing.
                                              No preambles or postambles. Keep strings in double quotes , dont give options give only one output""" 

        image_data = input_image_setup(io.BytesIO(image_bytes), "image/jpeg")
        response_list = get_gemini_responses("Analyze this image carefully.", image_data, [pp])
        response0 = response_list[0]

        # response0 = model.generate_content([sample_file, pp], safety_settings={
        #     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        #     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        #     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        # }).text

        if response0[0] == "`":
            response0 = response0[7:-4]
        js = json.loads(response0)
        prom = f"Give an eye-catching, one-line description of jwellery for e-commerce listing, according to the given description : {js['description']}, jwellery features : {js['attributes']} , jwellery type : {type}"
        # r2 = model.generate_content([prom], safety_settings={
        #     HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        #     HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        #     HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        #     HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        # }).text

        
        js["prompt"] = prom
        js["s3_url"] = s3_url
        if js["quality"] != "B" or js["quality"] != "C":
            js["quality"] = "A"
        os.remove(save_path)

        end2 = time.time()
        print(end2 - end)
        
        return {
            "display_name": image_name,
            "generated_name": js["name"],
            "quality": js["quality"],
            "description": js["final_caption"],
            "attributes": js["attributes"],
            "prompt": js["prompt"],
            "color" : js["color"],
            "s3_url": js["s3_url"],
        }



class DeleteImageRequest(BaseModel):
    s3_url: str

def extract_s3_key(s3_url: str) -> str:
    """
    Converts:
    https://bucket.s3.amazonaws.com/folder/img.jpg
    → folder/img.jpg
    """
    return s3_url.split(".amazonaws.com/")[-1]

# ---------- DELETE API ----------
@app.post("/delete_image")
async def delete_image(request: DeleteImageRequest):

    s3_urls = [(url.strip()) for url in request.s3_url.split(",")]

    for s3_url in s3_urls:

        s3_key = extract_s3_key(s3_url)

        # Load hash map
        try:
            with open(INT_HASH_MAP_FILE, "r") as f:
                id_to_path = json.load(f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to load hash map: {str(e)}")

        # Find Milvus ID
        milvus_id = None
        for _id, path in id_to_path.items():
            if path == s3_url or path.endswith(s3_key):
                milvus_id = int(_id)
                break

        if milvus_id is None:
            raise HTTPException(status_code=404, detail="Image not found in index")

        # ---- Delete from S3 ----
        try:
            s3.delete_object(Bucket=S3_BUCKET, Key=s3_key)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"S3 delete failed: {str(e)}")

        # ---- Delete from Milvus ----
        try:
            collection_db.delete(expr=f"id in [{milvus_id}]")
            collection_db.flush()
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Milvus delete failed: {str(e)}")

        # ---- Update hash map ----
        del id_to_path[str(milvus_id)]

        try:
            with open(INT_HASH_MAP_FILE, "w") as f:
                json.dump(id_to_path, f)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Failed to update hash map: {str(e)}")

    return {
        "status": "success"
    }


@app.post("/regenerate")
async def regenerate(previous_prompt: str=Form(...),style: str=Form(...)):
    new_prompt = previous_prompt + f"""Keep the jwellery features same and the type of jwellery should not change, and change the description to {style} describing jwellery features. The change should be significant.
     you should not give any prambles or postambles and not give any option to users as I have to directly display it on a website                   
    """
    new_res = model.generate_content([new_prompt],safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,

    }).text
    print(new_prompt)
    return new_res
    
@app.post("/regenerate_title")
async def reg_title(previous_title:str=Form(...),color:str=Form(...),attributes:str=Form(...)):
    prompt=f"Regenerate the title name of jwellery based on the given description of jwellery. the new title should be different from the current title :-\nDescription:- {attributes},\n color:- {color}\ncurrent title :- {previous_title}\n Return only the title. No preambles or postambles"
    new_title= model.generate_content([prompt],safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,

    }).text
    return new_title


@app.post("/image_similarity_search")
async def image_searh(
    file: UploadFile = File(...),
    top_k: int = 1,
    website: str = "alya",  # "alya" or "blysk" - select S3 bucket
    check_duplicate: str = "yes"  # "yes" or "no" - bypass duplicate detection
):
    # Get bucket based on website selection
    selected_bucket = BUCKET_MAP.get(website.lower(), S3_BUCKET)
    if website.lower() not in BUCKET_MAP:
        return JSONResponse(status_code=400, content={"error": f"Invalid website. Use 'alya' or 'blysk'"})

    # Step 1: Read query image and embed it
    image_bytes = await file.read()
    try:
        image = Image.open(BytesIO(image_bytes)).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid image: {str(e)}"})

    # If check_duplicate is "no", skip similarity search and upload directly
    if check_duplicate.lower() == "no":
        image_name = file.filename
        save_path = f"./{image_name}"
        with open(save_path, "wb") as f:
            f.write(image_bytes)

        try:
            timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
            # Get file extension from original filename
            file_ext = os.path.splitext(image_name)[1] or ".jpg"
            unique_name = f"{timestamp}{file_ext}"

            s3.upload_file(
                Filename=save_path,
                Bucket=selected_bucket,
                Key=unique_name,
                ExtraArgs={"ContentType": "image/jpeg"}
            )
            index_single_image_from_s3(collection_db, unique_name, clipmodel, processor, device, s3_bucket=selected_bucket)
            s3_url = f"https://{selected_bucket}.s3.amazonaws.com/{unique_name}"
        except Exception as e:
            os.remove(save_path)
            return JSONResponse(status_code=500, content={"error": f"Failed to upload to S3: {str(e)}"})

        os.remove(save_path)
        return {
            "duplicate_found": False,
            "duplicate_check": "skipped",
            "image_name": image_name,
            "s3_url": s3_url
        }

    query_emb = embed_image(image, clipmodel, processor, device)

    # Step 2: Search similar
    results = search_similar(collection_db, query_emb, top_k)

    # Step 3: Load ID-to-path map
    int_hash_map_file = os.getenv("INT_HASH_MAP_FILE", "int_hash_to_path.json")
    try:
        with open(int_hash_map_file, 'r') as f:
            id_to_path = json.load(f)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to load hash map: {str(e)}"})

    # Step 4: Check for duplicates (IP metric: dist < 0.05 = ~95% similarity)
    matches = []
    for _id, dist in results:
        print(f"[image_similarity_search] IP distance: {dist} (similarity: {(1-dist)*100:.2f}%)")
        if dist < 0.05:  # IP metric: 0.05 = ~95% similarity
            path = id_to_path.get(str(_id), "unknown")
            # Only count as duplicate if path is valid (not "unknown")
            if path != "unknown":
                matches.append({"id": _id, "distance": dist, "path": path})

    # Step 5: If duplicate found, return match info (don't upload)
    if matches:
        return {
            "duplicate_found": True,
            "results": matches
        }

    # Step 6: No duplicate - upload new image to S3 and index it
    image_name = file.filename
    save_path = f"./{image_name}"

    # Save the uploaded image to disk temporarily
    with open(save_path, "wb") as f:
        f.write(image_bytes)

    try:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d%H%M%S%f")
        # Get file extension from original filename
        file_ext = os.path.splitext(image_name)[1] or ".jpg"
        unique_name = f"{timestamp}{file_ext}"

        s3.upload_file(
            Filename=save_path,
            Bucket=selected_bucket,
            Key=unique_name,
            ExtraArgs={"ContentType": "image/jpeg"}
        )
        index_single_image_from_s3(collection_db, unique_name, clipmodel, processor, device, s3_bucket=selected_bucket)
        s3_url = f"https://{selected_bucket}.s3.amazonaws.com/{unique_name}"
    except Exception as e:
        os.remove(save_path)
        return JSONResponse(status_code=500, content={"error": f"Failed to upload to S3: {str(e)}"})

    os.remove(save_path)

    return {
        "duplicate_found": False,
        "image_name": image_name,
        "s3_url": s3_url
    }

class CatalogRequest(BaseModel):
    image_urls: str
    type: str
    marketplace: str
    skuids: str



@app.post("/catalog-ai")
async def catalog_ai(req: CatalogRequest):

    """
    Generate a catalog for AI using the given image URLs and SKUs.

    The response will contain a list of dictionaries, where each dictionary
    contains the attributes for a single image. The attributes will include
    the original filename, SKU, and description, as well as additional
    attributes generated by the AI model.

    Additionally, the response will include a URL to a generated Excel file
    containing the same data.

    :param req: A CatalogRequest object containing the image URLs and SKUs.
    :return: A JSON object containing the list of attributes and the URL to the Excel file.
    """
    clear_excel_file(req.type,req.marketplace)
    format = req.marketplace.lower()[:3] + "_" + req.type.lower()[:3]


    input_prompt_map = {
        "fli_ear": ([prompt_questions_earrings_flipkart], fixed_values_earrings, gpt_flipkart_earrings_prompt),
        "ama_ear": ([prompt_questions_earrings_amz], fixed_values_earrings_amz, gpt_amz_earrings_prompt),
        "mee_ear": ([prompt_questions_earrings_meesho], fixed_values_earrings_meesho, gpt_meesho_earrings_prompt),
        "fli_nec": ([prompt_questions_necklace_flipkart], fixed_values_necklace_flipkart, gpt_flipkart_necklace_prompt),
        "ama_nec": ([prompt_questions_necklace_amz], fixed_values_necklace_amz , gpt_amz_necklace_prompt),
        "mee_nec": ([prompt_questions_necklace_meesho], fixed_values_necklace_meesho, gpt_meesho_necklace_prompt),
        "fli_bra": ([prompt_questions_bracelet_flipkart], fixed_values_bracelet_flipkart, gpt_flipkart_bracelet_prompt),
        "mee_bra": ([prompt_questions_bracelet_meesho], fixed_values_bracelet_meesho, gpt_meesho_bracelet_prompt),
        "ama_bra": ([prompt_questions_bracelet_amz], fixed_values_bracelet_amz, gpt_amz_bracelet_prompt),
        "sho_ear": ([prompt_questions_earrings_shopsy], fixed_values_earrings, gpt_flipkart_earrings_prompt),
    }

    dims_prompt_map = {
        "fli_ear": [prompt_dimensions_earrings_flipkart],
        "ama_ear": [prompt_dimensions_earrings_amz],
        "fli_nec": [prompt_dimensions_necklace_flipkart],
        "fli_bra": [prompt_dimensions_bracelet_flipkart],
        "ama_bra": [prompt_dimensions_bracelet_amz],
    }

    input_prompts , fixed_values , gpt_prompt = input_prompt_map.get(format)
    dims_prompts = dims_prompt_map.get(format, [])

    if not input_prompts:
        return JSONResponse(status_code=400, content={"error": "Invalid format"})
    
    excel_results = []
    results = []

    # print(req.image_urls)
    # print()
    # print(req.skuids)

    url_list = [(url.strip()) for url in req.image_urls.split(",")]
    skuid_list = [(sku.strip()) for sku in req.skuids.split(",")]

    # print(skuid_list)
    # print()
    # print(url_list)


    #https://alyaimg.s3.amazonaws.com/SKU-107.jpg
    #https://alyaimg.s3.amazonaws.com/SKU-113.jpg

    count = 1
    #print(skuid_list)
    async with httpx.AsyncClient() as client:
        for url, skuid in zip(url_list, skuid_list):
            r = await client.get(url)
            if r.status_code != 200:
                raise HTTPException(400, f"Failed to fetch {url}")
            image_bytes = r.content
            # image=file
            # image_name = image.filename

            # save_path = f"./{image_name}"
            
            # # Save the uploaded image directly to the disk
            # with open(save_path, "wb") as f:
            #     f.write(image_bytes)  # Write the image data to the file

            image_data = input_image_setup(io.BytesIO(image_bytes), "image/jpeg")
            response_list = get_gemini_responses("Analyze this image carefully.", image_data, input_prompts)

            #description = response_list[0] if len(response_list) > 0 else "No description"
            response_json = {}

            if len(response_list) > 0:
                try:
                    cleaned = response_list[0].strip().replace("```json", "").replace("```", "").strip()
                    response_json = json.loads(cleaned)
                    # print(response_json)
                    # print()
                    # print()
                except Exception as e:
                    response_json = {"error": f"Failed to parse JSON: {str(e)}"}

            if dims_prompts and response_json.get("is_scale") == "Yes":
                dims_response = get_gemini_dims_responses("Analyze this image carefully.", image_data, dims_prompts)
                try:
                    cleaned_dims = dims_response.strip().replace("```json", "").replace("```", "").strip()
                    dims_json = json.loads(cleaned_dims)
                    response_json.update(dims_json)
                except Exception as e:
                    response_json["dimensions_error"] = f"Failed to parse dimensions: {str(e)}"

            gpt_dict = {}
            available_dict = fetch_data_for_sku(skuid)
            
            if available_dict.get("sku_exists") == False:
                gpt_dict = ask_gpt_with_image(prompt = gpt_prompt , image_bytes = image_bytes)
            else:
                if req.marketplace.lower()[:3] == "ama":
                    gpt_dict["description"] = available_dict.get("description")
                    if available_dict.get("title") == False and available_dict.get("bullet_points") == False:
                        temp_dict = ask_gpt_with_image(prompt = gpt_prompt_title_bp , image_bytes = image_bytes)
                        gpt_dict["item_name"] = temp_dict.get("title")
                    elif available_dict.get("title") == False:
                        title_dict = ask_gpt_with_image(prompt = gpt_prompt_title , image_bytes = image_bytes)
                        gpt_dict["item_name"] = title_dict.get("title")
                        gpt_dict["bullet_points"] = available_dict.get("bullet_points")
                    elif available_dict.get("bullet_points") == False:
                        temp_dict = ask_gpt_with_image(prompt = gpt_prompt_bp , image_bytes = image_bytes)
                        gpt_dict["item_name"] = available_dict.get("title")
                        gpt_dict.update(temp_dict)
                    else:
                        available_dict.pop("sku_exists", None)
                        gpt_dict["item_name"] = available_dict.get("title")
                        gpt_dict["bullet_points"] = available_dict.get("bullet_points")
                        gpt_dict["description"] = available_dict.get("description")
                elif req.marketplace.lower()[:3] == "fli" or req.marketplace.lower()[:3] == "sho":
                    gpt_dict["description"] = available_dict.get("description")
                else:
                    gpt_dict["description"] = available_dict.get("description")
                    if available_dict.get("title") == False:
                        title_dict = ask_gpt_with_image(prompt = gpt_prompt_title , image_bytes = image_bytes)
                        gpt_dict["Product Name"] = title_dict.get("title")
                    else:
                        gpt_dict["Product Name"] = available_dict.get("title")




            gpt_dict = expand_bullet_points(gpt_dict)
            store_data_for_sku(skuid, gpt_dict)
            gpt_dict = expand_bullet_points(gpt_dict)
            #print(gpt_dict)

            description = gpt_dict.get("description", "No description")
            gpt_dict.pop("description", None)

            response_json.update(gpt_dict)

            dict = {
                "filename": url,
                "description": description,
                "skuid": skuid
            }

            if format == "ama_ear":
                temp_dict = {
                    "stones_id": "1",
                    "stones_type" : "No Gemstone",
                    "stones_treatment_method": "Not Treated",
                    "stones_creation_method" : "unknown"
                }

                if response_json.get("stones_number_of_stones",0) == 0:
                    response_json.update(temp_dict)
                
            final_response = {**dict ,**response_json, **fixed_values}
            # static_file_path = os.path.join("static", static_file_name)

            # print(static_file_path)
            # static_url = f"http://15.206.26.88:8000/static/{static_file_name}"  

            excel_results.append((skuid,response_json,description))

            results.append({
                "attributes": final_response,
            })
            count += 1
            if count%5 == 0:
                time.sleep(30)
    
    filename_map = {
        "fli_ear": "earrings_flipkart.xlsx",
        "ama_ear": "earrings_amz.xlsx",
        "mee_ear": "earrings_meesho.xlsx",
        "sho_ear": "earrings_shopsy.xlsx",
        "fli_nec": "necklace_flipkart.xlsx",
        "ama_nec": "necklace_amz.xlsx",
        "mee_nec": "necklace_meesho.xlsx",
        "fli_bra": "bracelet_flipkart.xlsx",
        "mee_bra": "bracelet_meesho.xlsx",
        "ama_bra": "bracelet_amz.xlsx"
    }


    static_file_name = filename_map.get(format)
    s3_url = f"https://alyaimg.s3.amazonaws.com/excel_files/{static_file_name}"

    s3_key = f"excel_files/{static_file_name}"

# Download the file using httpx
    response = httpx.get(s3_url)
    response.raise_for_status()

# Save to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name


    if format == "fli_ear":
        write_to_excel_flipkart(excel_results, filename=tmp_path, target_fields=target_fields_earrings, fixed_values=fixed_values_earrings)
    elif format == "ama_ear":
        write_to_excel_amz_xl(excel_results, filename=tmp_path, target_fields=target_fields_earrings_amz, fixed_values=fixed_values_earrings_amz)
    elif format == "mee_ear":
        write_to_excel_meesho(excel_results, filename=tmp_path, target_fields=target_fields_earrings_meesho, fixed_values=fixed_values_earrings_meesho)
    elif format == "sho_ear":
        write_to_excel_flipkart(excel_results, filename=tmp_path, target_fields=target_fields_earrings, fixed_values=fixed_values_earrings)
    elif format == "fli_nec":
        write_to_excel_flipkart(excel_results, filename=tmp_path, target_fields=target_fields_necklace_flipkart, fixed_values=fixed_values_necklace_flipkart)
    elif format == "ama_nec":
        write_to_excel_amz_xl(excel_results, filename=tmp_path, target_fields=target_fields_necklace_amz, fixed_values=fixed_values_necklace_amz)
    elif format == "ama_bra":
        write_to_excel_amz_xl(excel_results, filename=tmp_path, target_fields=target_fields_bracelet_amz, fixed_values=fixed_values_bracelet_amz)
    elif format == "mee_nec":
        write_to_excel_meesho(excel_results, filename=tmp_path, target_fields=target_fields_necklace_meesho, fixed_values=fixed_values_necklace_meesho)
    elif format == "fli_bra":
        write_to_excel_flipkart(excel_results, filename=tmp_path, target_fields=target_fields_bracelet_flipkart, fixed_values=fixed_values_bracelet_flipkart)
    elif format == "mee_bra":   
        write_to_excel_meesho(excel_results, filename=tmp_path, target_fields=target_fields_bracelet_meesho, fixed_values=fixed_values_bracelet_meesho)

    # Upload the modified file back to S3 (same key)
    s3.upload_file(
        Filename=tmp_path,
        Bucket=S3_BUCKET,
        Key=s3_key,
        ExtraArgs={'ContentType': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'}
    )

    os.remove(tmp_path)


    return {"results": results , "excel_file": s3_url}

class CatalogRequestVariation(BaseModel):
    number_of_grps: str
    group_sizes: str
    image_urls: str
    variations_category: str
    size_available: str
    type: str
    marketplace: str
    skuids: str


@app.post("/catalog_ai_variations")
async def catalog_ai_variations(req: CatalogRequestVariation):
    """
    Generate a catalog with variations for AI using the given image URLs and SKUs.

    This function processes images and generates a catalog with variations based on the
    specified categories. The response will contain a list of dictionaries, where each
    dictionary contains the attributes for a single image, including variations like size
    and color. Additionally, the response will include a URL to a generated Excel file
    containing the same data.

    :param req: A CatalogRequestVariation object containing the image URLs, SKUs, number
                of groups, group sizes, variations category, available sizes, type, and
                marketplace.
    :return: A JSON object containing the list of attributes and the URL to the Excel file.
    """

    format = req.marketplace.lower()[:3] + "_" + req.type.lower()[:3]


    input_prompt_map = {
        "fli_ear": ([prompt_description_earrings_flipkart, prompt_questions_earrings_flipkart], fixed_values_earrings),
        "ama_ear": ([prompt_description_earrings_amz, prompt_questions_earrings_amz], fixed_values_earrings_amz),
        "mee_ear": ([prompt_description_earrings_meesho, prompt_questions_earrings_meesho], fixed_values_earrings_meesho),
        "fli_nec": ([prompt_description_necklace_flipkart, prompt_questions_necklace_flipkart], fixed_values_necklace_flipkart),
        "ama_nec": ([prompt_description_necklace_amz, prompt_questions_necklace_amz], fixed_values_necklace_amz),
        "mee_nec": ([prompt_description_necklace_meesho, prompt_questions_necklace_meesho], fixed_values_necklace_meesho),
        "fli_bra": ([prompt_description_bracelet_flipkart, prompt_questions_bracelet_flipkart], fixed_values_bracelet_flipkart),
        "mee_bra": ([prompt_description_bracelet_meesho, prompt_questions_bracelet_meesho], fixed_values_bracelet_meesho),
    }

    dims_prompt_map = {
        "fli_ear": [prompt_dimensions_earrings_flipkart],
        "ama_ear": [prompt_dimensions_earrings_amz],
        "fli_nec": [prompt_dimensions_necklace_flipkart],
        "fli_bra": [prompt_dimensions_bracelet_flipkart],
    }

    input_prompts , fixed_values = input_prompt_map.get(format)
    dims_prompts = dims_prompt_map.get(format, [])

    if not input_prompts:
        return JSONResponse(status_code=400, content={"error": "Invalid format"})
    
    results = []

    # print(req.image_urls)
    # print()
    # print(req.skuids)

    group_sizes_list = list(map(int, req.group_sizes.split(",")))
    url_list = [(url.strip()) for url in req.image_urls.split(",")]
    skuid_list = [(sku.strip()) for sku in req.skuids.split(",")]
    category_list = [(cat.strip()) for cat in req.variations_category.split(",")]
    size_available_list = [(size.strip()) for size in req.size_available.split(",")]
    print(size_available_list)
    print()
    count = 1

    image_data_list = []
    image_data_list_2 = []

    #get image data list
    async with httpx.AsyncClient() as client:
        for url, skuid in zip(url_list, skuid_list):
            r = await client.get(url)
            if r.status_code != 200:
                raise HTTPException(400, f"Failed to fetch {url}")
            image_bytes = r.content
            # image=file
            # image_name = image.filename

            # save_path = f"./{image_name}"
            
            # # Save the uploaded image directly to the disk
            # with open(save_path, "wb") as f:
            #     f.write(image_bytes)  # Write the image data to the file

            image_data = input_image_setup(io.BytesIO(image_bytes), "image/jpeg")
            image_data_list.append(image_data[0])
            image_data_list_2.append(image_data)
    
    print("loaded images")

    # get excel file 
    filename_map = {
        "fli_ear": "earrings_flipkart.xlsx",
        "ama_ear": "earrings_amazon_variations.xlsx",
        "mee_ear": "earrings_meesho.xlsx",
        "fli_nec": "necklace_flipkart.xlsx",
        "ama_nec": "necklace_amz.xlsx",
        "mee_nec": "necklace_meesho.xlsx",
        "fli_bra": "bracelet_flipkart.xlsx",
        "mee_bra": "bracelet_meesho.xlsx"
    }

    static_file_name = filename_map.get(format)
    s3_url = f"https://alyaimg.s3.amazonaws.com/excel_files/{static_file_name}"

    s3_key = f"excel_files/{static_file_name}"

# Download the file using httpx
    response = httpx.get(s3_url)
    response.raise_for_status()

# Save to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name

    print("saved files")

    index = 0
    ci = 0
    url_num = 0
    for group_index, group_size in enumerate(group_sizes_list):
        category = category_list[ci]
        if(category == "size"):
            group_images = [image_data_list_2[i][0] for i in range(index, index + 1)]
            group_image_data_2 = image_data_list_2[index:index + 1]
        elif(category == "colorsize"):
            group_images = [image_data_list_2[i][0] for i in range(index, index + (group_size//len(size_available_list)))]
            group_image_data_2 = image_data_list_2[index:index + (group_size//len(size_available_list))]
        else:
            group_images = [image_data_list_2[i][0] for i in range(index, index + group_size)]
            group_image_data_2 = image_data_list_2[index:index + group_size]
        group_skuids = skuid_list[index:index + group_size]

        print(category)
        print()

        image_name = ""
        ## write variations row for various variaiton categories
        if(category == "color"):
            variations_prompt_list = [variations_prompt]
            variations_row = get_gemini_responses_multi_image("analyse the images carefully", group_images, variations_prompt_list)
            print(f"done\n")
            cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", variations_row[0].strip(), flags=re.IGNORECASE)
            variation_row_dict = json.loads(cleaned)
            image_name = variation_row_dict["item_sku"]
            variation_row_dict.pop("item_sku")
            results_dict = {**variation_row_dict,**fixed_values_variation_amz}
            results.append(results_dict)
            write_to_excel_amz_xl([(image_name,variation_row_dict,"")], tmp_path,target_values_variations_amz,fixed_values_variation_amz)
        elif(category == "size"):
            fixed_values_variation_amz["variation_theme"] = "SizeName"
            variations_prompt_list = [variations_prompt]
            variations_row = get_gemini_responses_multi_image("analyse the images carefully", group_images, variations_prompt_list)
            print(f"done\n")
            cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", variations_row[0].strip(), flags=re.IGNORECASE)
            variation_row_dict = json.loads(cleaned)
            image_name = variation_row_dict["item_sku"]
            variation_row_dict.pop("item_sku")
            results_dict = {**variation_row_dict,**fixed_values_variation_amz}
            results.append(results_dict)  
            write_to_excel_amz_xl([(image_name,variation_row_dict,"")], tmp_path,target_values_variations_amz,fixed_values_variation_amz)
        elif(category == "colorsize"):
            fixed_values_variation_amz["variation_theme"] = "colorSize"
            variations_prompt_list = [variations_prompt]
            variations_row = get_gemini_responses_multi_image("analyse the images carefully", group_images, variations_prompt_list)
            print(f"done\n")
            cleaned = re.sub(r"```(?:json)?\s*([\s\S]*?)\s*```", r"\1", variations_row[0].strip(), flags=re.IGNORECASE)
            variation_row_dict = json.loads(cleaned)
            image_name = variation_row_dict["item_sku"]
            variation_row_dict.pop("item_sku")
            results_dict = {**variation_row_dict,**fixed_values_variation_amz}
            results.append(results_dict)
            write_to_excel_amz_xl([(image_name,variation_row_dict,"")], tmp_path,target_values_variations_amz,fixed_values_variation_amz)


        fixed_values_variations_child["parent_sku"] = image_name
        fixed_values.update(fixed_values_variations_child)
        ## process images for other data fields
        excel_results = []
        for image_data , skuid in zip(group_image_data_2, group_skuids):
                response_list = get_gemini_responses("Analyze this image carefully.", image_data, input_prompts)

                description = response_list[0] if len(response_list) > 0 else "No description"
                response_json = {}

                if len(response_list) > 1:
                    try:
                        cleaned = response_list[1].strip().replace("```json", "").replace("```", "").strip()
                        response_json = json.loads(cleaned)
                    except Exception as e:
                        response_json = {"error": f"Failed to parse JSON: {str(e)}"}

                if dims_prompts:
                    dims_response = get_gemini_dims_responses("Analyze this image carefully.", image_data, dims_prompts)
                    try:
                        cleaned_dims = dims_response.strip().replace("```json", "").replace("```", "").strip()
                        dims_json = json.loads(cleaned_dims)
                        response_json.update(dims_json)
                    except Exception as e:
                        response_json["dimensions_error"] = f"Failed to parse dimensions: {str(e)}"

                dict = {
                    "filename": url_list[url_num],
                    "description": description,
                    "skuid": skuid
                }
                
                final_response = {**dict ,**response_json, **fixed_values }
                # static_file_path = os.path.join("static", static_file_name)

                # print(static_file_path)
                # static_url = f"http://15.206.26.88:8000/static/{static_file_name}"  

                excel_results.append((skuid,response_json,description))

                results.append({
                    "attributes": final_response,
                })
                url_num += 1
                count += 1
                if count%5 == 0:
                    time.sleep(30)
        
        if category == "size":
            skuid , response_json , description = excel_results[0]
            excel_results = []
            temp_result = results[len(results)-1]["attributes"]
            results.pop()
            for size , skuid in zip(size_available_list, group_skuids):
                print(size)
                print() 
                new_json = response_json.copy()
                new_json["size_name"] = size
                copy_results = temp_result.copy()
                copy_results["size_name"] = size
                copy_results["skuid"] = skuid
                results.append({
                    "attributes": copy_results,
                })
                excel_results.append((skuid, new_json, description))
        elif category == "colorsize":
            new_excel_results = []
            sku_index = 0
            temp_results = []
            for i in range(group_size//len(size_available_list)):
                temp_results.append(results.pop()["attributes"])
            for skuid , response_json , description in excel_results:
                skuids = group_skuids[sku_index: sku_index + len(size_available_list)]
                sku_index += len(size_available_list)
                temp_result = temp_results.pop()
                for size , skuid in zip(size_available_list, skuids):
                    print(size)
                    print() 
                    new_json = response_json.copy()
                    new_json["size_name"] = size
                    copy_results = temp_result.copy()
                    copy_results["size_name"] = size
                    copy_results["skuid"] = skuid
                    results.append({
                        "attributes": copy_results,
                    })
                    new_excel_results.append((skuid, new_json, description))

            excel_results = new_excel_results


        index += group_size
        ci += 1

        print(f"group {group_index} done\n")
        print()
        print(excel_results)
        #write to excel other data
        if format == "fli_ear":
            write_to_excel_flipkart(excel_results, filename=tmp_path, target_fields=target_fields_earrings, fixed_values=fixed_values_earrings)
        elif format == "ama_ear":
            fixed_values_variations_child["parent_sku"] = image_name
            #fixed_values_earrings_amz.update(fixed_values_variations_child)
            if(category == "size" or category == "colorsize"):
                fixed_values_earrings_amz.pop("size_name")
                target_fields_earrings_amz.append("size_name")
                if category == "size":
                    fixed_values_variations_child["variation_theme"] = "SizeName"
                else:
                    fixed_values_variations_child["variation_theme"] = "colorSize"
            fixed_values_earrings_amz.update(fixed_values_variations_child)
            write_to_excel_amz_xl(excel_results, filename=tmp_path, target_fields=target_fields_earrings_amz, fixed_values=fixed_values_earrings_amz)
        elif format == "mee_ear":
            write_to_excel_meesho(excel_results, filename=tmp_path, target_fields=target_fields_earrings_meesho, fixed_values=fixed_values_earrings_meesho)
        elif format == "fli_nec":
            write_to_excel_flipkart(excel_results, filename=tmp_path, target_fields=target_fields_necklace_flipkart, fixed_values=fixed_values_necklace_flipkart)
        elif format == "ama_nec":
            fixed_values_variations_child["parent_sku"] = image_name
            #fixed_values_earrings_amz.update(fixed_values_variations_child)
            if(category == "size" or category == "colorsize"):
                fixed_values_necklace_amz.pop("size_name")
                target_fields_necklace_amz.append("size_name")
                if category == "size":
                    fixed_values_variations_child["variation_theme"] = "SizeName"
                else:
                    fixed_values_variations_child["variation_theme"] = "colorSize"
            fixed_values_necklace_amz.update(fixed_values_variations_child)
            #write_to_excel_amz_xl(excel_results, filename=tmp_path, target_fields=target_fields_earrings_amz, fixed_values=fixed_values_earrings_amz)
            write_to_excel_amz_xl(excel_results, filename=tmp_path, target_fields=target_fields_necklace_amz, fixed_values=fixed_values_necklace_amz)
        elif format == "mee_nec":
            write_to_excel_meesho(excel_results, filename=tmp_path, target_fields=target_fields_necklace_meesho, fixed_values=fixed_values_necklace_meesho)
        elif format == "fli_bra":
            write_to_excel_flipkart(excel_results, filename=tmp_path, target_fields=target_fields_bracelet_flipkart, fixed_values=fixed_values_bracelet_flipkart)
        elif format == "mee_bra":   
            write_to_excel_meesho(excel_results, filename=tmp_path, target_fields=target_fields_bracelet_meesho, fixed_values=fixed_values_bracelet_meesho)


    s3.upload_file(
        Filename=tmp_path,
        Bucket=S3_BUCKET,
        Key=s3_key,
        ExtraArgs={'ContentType': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'}
    )

    os.remove(tmp_path)


    return {"results": results , "excel_file": s3_url}

#@app.post("/clear-excel")
def clear_excel_file(type: str = Form(...) , marketplace: str = Form(...)):
    static_dir = "static"
    format = marketplace.lower()[:3] + "_" + type.lower()[:3]
    filename_map = {
        "fli_ear": "earrings_flipkart.xlsx",
        "ama_ear": "earrings_amz.xlsx",
        "mee_ear": "earrings_meesho.xlsx",
        "sho_ear": "earrings_shopsy.xlsx",
        "fli_nec": "necklace_flipkart.xlsx",
        "ama_nec": "necklace_amz.xlsx",
        "mee_nec": "necklace_meesho.xlsx",
        "fli_bra": "bracelet_flipkart.xlsx",
        "mee_bra": "bracelet_meesho.xlsx",
        "ama_bra": "bracelet_amz.xlsx",
    }

    filename = filename_map[format]

    s3_url = f"https://alyaimg.s3.amazonaws.com/excel_files/{filename}"

    s3_key = f"excel_files/{filename}"

# Download the file using httpx
    print(s3_url)
    response = httpx.get(s3_url)
    response.raise_for_status()

# Save to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".xlsx") as tmp_file:
        tmp_file.write(response.content)
        tmp_path = tmp_file.name


    # base_dir = os.path.dirname(__file__)
    # file_path = file_path = os.path.join(base_dir, "static", filename)
    # print(file_path)

    # if not os.path.exists(file_path):
    #     raise HTTPException(status_code=404, detail="File not found")

    # Determine starting row based on filename
    filename_lower = filename.lower()
    sheet_number = 0
    if "amz" in filename_lower or "ama" in filename_lower:
        start_row = 4
        sheet_number = 0
    elif "fli" in filename_lower or "sho" in filename_lower:
        start_row = 5
        sheet_number = 2
    elif "mee" in filename_lower:
        start_row = 5
        sheet_number = 1
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        wb = openpyxl.load_workbook(tmp_path)
        ws = wb.worksheets[sheet_number]

        max_row = ws.max_row
        if max_row >= start_row:
            ws.delete_rows(start_row, max_row - start_row + 1)

        wb.save(tmp_path)
        wb.close()

        s3.upload_file(
        Filename=tmp_path,
        Bucket=S3_BUCKET,
        Key=s3_key,
        ExtraArgs={'ContentType': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'}
        )

        os.remove(tmp_path)


        return {"message": f"Cleared rows from row {start_row} in {filename}"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear Excel file: {str(e)}")
    

@app.post("/create_order")
async def create_order(file: UploadFile = File(...)):
    """
    Parse an uploaded PDF file and return the parsed data and products data in a JSON format.
    
    The PDF file is uploaded to Gemini servers, and then the Gemini model is used to extract the fields.
    The extracted fields are then used to generate a parsed data object and a products data array.
    The parsed data object includes all the fields that were extracted from the PDF file.
    The products data array includes an array of objects, each containing the product price, item type, sku id, and quantity.
    """
    if file.content_type != "application/pdf":
        return {"error": "Only PDF files are allowed"}

    # Save the PDF to a temp file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        temp_path = tmp.name
        tmp.write(await file.read())

    model_1 = genai.GenerativeModel('gemini-2.0-flash',
                                  generation_config=genai.types.GenerationConfig(
                                      temperature=0.5,
                                      top_p=0.9,
                                      top_k=40,
                                      max_output_tokens=1024
                                  ))

    # Load the file for Gemini
    pdf_part = genai.upload_file(path=temp_path)

    response = model_1.generate_content([order_prompt, pdf_part])
    response = response.text
    cleaned = response.strip().replace("```json", "").replace("```", "").strip()
    response_json = json.loads(cleaned)

    # Optional: delete the uploaded file from Gemini servers
    genai.delete_file(pdf_part.name)
    os.remove(temp_path)

    item_types = response_json["item_types"]
    sku_ids = response_json["sku_ids"]
    quantities = response_json["quantities"]
    product_price = response_json["product_price"]

    response_json.pop("product_price")
    response_json.pop("item_types")
    response_json.pop("sku_ids")
    response_json.pop("quantities")

    product_price_array = product_price.split(",")
    item_types_array = item_types.split(",")
    sku_ids_array = sku_ids.split(",")
    quantities_array = quantities.split(",")

    products_array = []

    for i in range(len(item_types_array)):
        product = {
            "product_price": product_price_array[i].strip(),
            "item_type": re.match(r'[A-Za-z]+', item_types_array[i].strip()).group(),
            "sku_id": sku_ids_array[i].strip(),
            "quantity": quantities_array[i].strip()
        }
        products_array.append(product)

    return {
        "parsed_data": response_json,
        "products_data": products_array
    } 

# Default image types for earrings (skip type 3 dimension image)
DEFAULT_EARRING_IMAGE_TYPES = [1, 2, 4, 5]  # White BG, Hand, Lifestyle, Model

class ImageRequest(BaseModel):
    s3_urls: str
    product_type: str

    class Config:
        schema_extra = {
            "example": {
                "s3_urls": "https://alyaimg.s3.amazonaws.com/your_image.jpg",
                "product_type": "ear"
            }
        }

def create_zip_from_s3_urls(image_urls, zip_filename):
    # Create an in-memory ZIP file
    """
    Downloads images from a list of S3 URLs, creates a ZIP file containing these images, 
    uploads the ZIP file to an S3 bucket, and returns the public URL of the uploaded ZIP file.

    Args:
        image_urls (list of str): List of S3 URLs pointing to images to be zipped.
        zip_filename (str): Desired name for the ZIP file to be created and uploaded.

    Returns:
        str: Public URL of the uploaded ZIP file in the S3 bucket.

    Raises:
        Exception: If there is an error downloading any of the images from the URLs.
    """

    zip_buffer = io.BytesIO()

    with zipfile.ZipFile(zip_buffer, 'w', zipfile.ZIP_DEFLATED) as zip_file:
        for url in image_urls:
            try:
                filename = url.split("/")[-1]
                response = requests.get(url)
                response.raise_for_status()
                zip_file.writestr(filename, response.content)
            except Exception as e:
                print(f"Error downloading {url}: {e}")

    # Upload ZIP to S3
    zip_buffer.seek(0)
    s3_key = f"{GENERATED_FOLDER}{zip_filename}"
    s3.upload_fileobj(zip_buffer, S3_BUCKET, s3_key, ExtraArgs={"ContentType": "application/zip"})

    # Return public URL
    public_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{s3_key}"
    return public_url

@app.post("/generate-images")
async def generate_images(request: ImageRequest):

    """
    Generate images based on a given image URL and product type, and return the generated images' URLs.

    This function accepts an S3 URL of an image and a product type, retrieves the image, and generates
    new images using predefined prompts based on the product type. The generated images are uploaded
    to S3, and their URLs are returned along with a ZIP file containing all the images.

    Args:
        request (ImageRequest): An object containing the S3 URL of the image and the product type.

    Returns:
        dict: A dictionary containing the original image URL, a list of generated image URLs, and the
        ZIP file URL containing all the generated images.

    Raises:
        JSONResponse: If the product type is invalid or if an error occurs while fetching or processing
        the image.
    """

    product_type = request.product_type[0:3].lower()

    # Track image types for response
    image_types_list = None

    # For earrings, use batch-4 generation with types [1,2,4,5]
    if product_type == "ear":
        # Always use default types [1,2,4,5] (skip type 3 dimension)
        image_types = DEFAULT_EARRING_IMAGE_TYPES

        # Build prompt list from image types
        prompt_list = []
        image_types_list = []
        for img_type in image_types:
            prompt = get_earring_prompt(img_type, None, None)
            prompt_list.append(prompt)
            image_types_list.append({
                "type": img_type,
                "name": EARRING_IMAGE_TYPES[img_type]["name"]
            })
    else:
        # Fallback to original behavior for bracelet/necklace (4 images)
        prompt_map = {
            "bra": [white_bgd_bracelet_prompt, multicolor_1_bracelet_prompt, multicolor_2_bracelet_prompt, hand_bracelet_prompt],
            "nec": [white_bgd_necklace_prompt, multicolor_1_necklace_prompt, multicolor_2_necklace_prompt, hand_necklace_prompt]
        }

        prompt_list = prompt_map.get(product_type)
        if not prompt_list:
            return JSONResponse(status_code=400, content={"error": "Invalid product_type. Use 'ear', 'bra', or 'nec'"})

    results = {}
    url = request.s3_urls
    async with httpx.AsyncClient() as client:
            try:
                response = await client.get(url)
                response.raise_for_status()
            except httpx.HTTPError as e:
                return JSONResponse(status_code=400, content={"error": f"Failed to fetch image from URL: {url}", "details": str(e)})

            try:
                image = Image.open(io.BytesIO(response.content))
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Failed to decode image from URL: {url}", "details": str(e)})

            responses = generate_images_from_gpt(image, prompt_list)
            image_urls = []
            parsed = urlparse(url)
            filename_base = os.path.basename(parsed.path) 
            name_no_ext = os.path.splitext(filename_base)[0]

            for i, item in enumerate(responses):
                # Skip failed generations (empty images list due to rate limit or other errors)
                if not item.get("images"):
                    print(f"[WARNING] Skipping image {i} - generation failed: {item.get('error', 'unknown error')}")
                    continue

                img_bytes = item["images"][0]

                gen_image = Image.open(io.BytesIO(img_bytes))
                gen_image = resize_img(gen_image)

                img_buffer = io.BytesIO()
                gen_image.save(img_buffer, format="PNG")
                resized_img_bytes = img_buffer.getvalue()

                # Use image_type in filename if specified, otherwise use prompt index
                if image_types_list is not None:
                    img_type = image_types_list[i]["type"]
                    filename = f"{name_no_ext}_type{img_type}.png"
                else:
                    filename = f"{name_no_ext}_prompt{i}.png"
                key = f"{GENERATED_FOLDER}{filename}"

                # Upload to S3
                s3.put_object(Bucket=S3_BUCKET, Key=key, Body=resized_img_bytes, ContentType="image/png")

                image_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{key}"
                image_info = {
                    "prompt_index": i,
                    "image_url": image_url
                }
                # Add image_type info if specified
                if image_types_list is not None:
                    image_info["image_type"] = image_types_list[i]["type"]
                    image_info["image_type_name"] = image_types_list[i]["name"]
                image_urls.append(image_info)
            
            urls = [img['image_url'] for img in image_urls]
            zip_url = create_zip_from_s3_urls(urls, f"{name_no_ext}.zip")
            results["zip_url"] = zip_url

            results["original_image_url"] = url
            results["gen_images"] = image_urls

    return results


def update_zip_on_s3(zip_url: str, new_image_bytes: bytes, new_filename: str):
    # Parse original zip key
    """
    Update a ZIP file on S3 by replacing one of its files and re-uploading the entire ZIP.

    Args:
        zip_url (str): Public URL of the original ZIP file on S3.
        new_image_bytes (bytes): Bytes of the new image to be added to the ZIP.
        new_filename (str): Desired filename for the new image in the ZIP.

    Returns:
        str: Same public URL as the original ZIP file, but with the updated contents.
    """

    parsed = urlparse(zip_url)
    zip_key = parsed.path.lstrip("/")
    zip_name = zip_key.split("/")[-1]

    # Extract bucket from URL
    bucket_name = parsed.netloc.split('.')[0]  # e.g., "alyaimg" from "alyaimg.s3.amazonaws.com"

    # Download original ZIP using S3 client (more reliable than requests)
    try:
        zip_obj = s3.get_object(Bucket=bucket_name, Key=zip_key)
        zip_content = zip_obj['Body'].read()
    except s3.exceptions.NoSuchKey:
        raise Exception(f"ZIP file not found in S3: {zip_key}")
    except Exception as e:
        # Fallback to requests if S3 client fails
        try:
            response = requests.get(zip_url, timeout=30)
            if response.status_code != 200:
                raise Exception(f"Failed to download ZIP: {zip_url} - Status: {response.status_code}")
            zip_content = response.content
        except requests.exceptions.RequestException as req_err:
            raise Exception(f"Failed to download ZIP from {zip_url} - S3 Error: {str(e)}, HTTP Error: {str(req_err)}")

    # Read ZIP into memory
    zip_buffer_in = io.BytesIO(zip_content)
    zip_buffer_out = io.BytesIO()

    with zipfile.ZipFile(zip_buffer_in, 'r') as zin:
        with zipfile.ZipFile(zip_buffer_out, 'w', zipfile.ZIP_DEFLATED) as zout:
            for item in zin.infolist():
                if item.filename != new_filename:
                    # Copy everything except the file to be replaced
                    zout.writestr(item.filename, zin.read(item.filename))
            # Add the new/updated image
            zout.writestr(new_filename, new_image_bytes)

    # Upload new ZIP to S3 (overwrite)
    zip_buffer_out.seek(0)
    s3.put_object(Bucket=bucket_name, Key=zip_key, Body=zip_buffer_out.read(), ContentType="application/zip")

    # Return same public URL
    return f"https://{bucket_name}.s3.amazonaws.com/{zip_key}"

@app.post("/regenerate-image")
async def regenerate_image(
    original_image_url: str = Form(...),
    old_generated_url: str = Form(...),
    old_zip_url: str = Form(...),
    product_type: str = Form(...),
    prompt_index: int = Form(...)
):
    
    # 1. Validate product_type and prompt_index
    """
    Regenerate a single image from the original image URL, replacing the old generated image URL
    and updating the ZIP file URL.

    Args:
        original_image_url (str): The original image URL on S3.
        old_generated_url (str): The old generated image URL on S3.
        old_zip_url (str): The original ZIP file URL on S3.
        product_type (str): The product type (ear, bra, or nec).
        prompt_index (int): The index of the prompt to use for regeneration.

    Returns:
        dict: A dictionary containing the same original image URL, the new generated image URL,
        and the updated ZIP file URL.
    """
    prompt_map = {
        "ear": [white_bgd_prompt, multicolor_1_prompt, multicolor_2_prompt, props_img_prompt, hand_prompt],
        "bra": [white_bgd_bracelet_prompt, multicolor_1_bracelet_prompt, multicolor_2_bracelet_prompt, props_img_bracelet_prompt, hand_bracelet_prompt],
        "nec": [white_bgd_necklace_prompt, multicolor_1_necklace_prompt, multicolor_2_necklace_prompt, props_img_necklace_prompt, hand_necklace_prompt, neck_necklace_prompt]
    }

    prompts = prompt_map.get(product_type[0:3].lower())
    if not prompts or not (0 <= prompt_index < len(prompts)):
        raise HTTPException(400, "Invalid product_type or prompt_index")

    # 2. Download original image from S3 URL
    try:
        async with httpx.AsyncClient() as client:
            response = await client.get(original_image_url)
            response.raise_for_status()
            image = Image.open(io.BytesIO(response.content))
    except Exception as e:
        raise HTTPException(400, f"Failed to download original image: {e}")

    # 3. Delete old image from S3
    try:
        parsed = urlparse(old_generated_url)
        old_key = parsed.path.lstrip("/")
        s3.delete_object(Bucket=S3_BUCKET, Key=old_key)
    except Exception as e:
        raise HTTPException(500, f"Failed to delete old image: {e}")

    # 4. Generate new image
    new_response = gen_image_responses("analyse the image", image, [prompts[prompt_index]])[0]
    img_bytes = new_response["images"][0]
    gen_image = Image.open(io.BytesIO(img_bytes))
    gen_image = resize_img(gen_image)

    img_buffer = io.BytesIO()
    gen_image.save(img_buffer, format="PNG")
    resized_img_bytes = img_buffer.getvalue()

    filename = os.path.basename(parsed.path)

    updated_zip_url = update_zip_on_s3(old_zip_url, resized_img_bytes, filename)

    # 5. Upload new image to same S3 path
    try:
        s3.put_object(
            Bucket=S3_BUCKET,
            Key=old_key,
            Body=resized_img_bytes,
            ContentType="image/png",
        )
    except Exception as e:
        raise HTTPException(500, f"Failed to upload new image: {e}")

    # 6. Return same S3 URL
    return {"image_url": old_generated_url, "zip_url": updated_zip_url}


class rankRequest(BaseModel):
    asin: str
    search_query: str
    marketplace: str


@app.post("/get_rank_product")
async def get_rank_product(rankRequest: rankRequest):
    asins = rankRequest.asin
    search_query = rankRequest.search_query

    asin_list = [asin.strip() for asin in asins.split(",") if asin.strip()]

    search_query_list = [search.strip() for search in search_query.split(",") if search.strip()]
    marketplace = rankRequest.marketplace
    if marketplace.lower() == "amazon":
        final_results = []
        for asin in asin_list:
            results = []
            try:
                for search in search_query_list:
                    result = await get_product_rank(asin, search)
                    result["search_query"] = search
                    results.append(result)
                    #time.sleep(50)
                final_results.append({"asin": asin, "results": results})
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    elif marketplace.lower() == "flipkart":
        final_results = []
        for asin in asin_list:
            results = []
            try:
                for search in search_query_list:
                    result = await get_flipkart_rank(asin, search)
                    result["search_query"] = search
                    results.append(result)
                    #time.sleep(50)
                final_results.append({"itmid": asin, "results": results})
            except Exception as e:
                raise HTTPException(status_code=500, detail=str(e))
    else:
        raise HTTPException(status_code=400, detail="Marketplace not supported")

    return final_results
        

class CompetitorRequest(BaseModel):
    asins: str  # Comma-separated ASINs
    marketplace: str


@app.post("/competitor_analysis")
async def competitor_analysis(competitor_request: CompetitorRequest):
    try:
        # Convert comma-separated ASINs into a list
        asin_list = [asin.strip() for asin in competitor_request.asins.split(",") if asin.strip()]
        
        if not asin_list:
            raise ValueError("No valid ASINs provided.")
        
        # Optionally process marketplace
        marketplace = competitor_request.marketplace.lower()
        response = {}
        if(marketplace.lower() == "amazon"):
            results = await scrape_all_product_details(asin_list)
            local_file = "amazon_products.xlsx"

            generate_excel_from_products(results, local_file)

            key = "excel_files/amazon_products.xlsx"

            s3.upload_file(
                Filename=local_file,
                Bucket=S3_BUCKET,
                Key=key,
                ExtraArgs={'ContentType': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'}
            )
            response["result"] = results
        elif(marketplace.lower() == "flipkart"):
            results = await scrape_flipkart_items(asin_list)
            local_file = "flipkart_products.xlsx"

            generate_excel_from_products_flipkart(results, local_file)

            key = "excel_files/flipkart_products.xlsx"

            s3.upload_file(
                Filename=local_file,
                Bucket=S3_BUCKET,
                Key=key,
                ExtraArgs={'ContentType': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'}
            )
            response["result"] = results
        elif(marketplace.lower() == "myntra"):
            results = await scrape_myntra_items(asin_list)
            local_file = "myntra_products.xlsx"

            generate_excel_from_products_myntra(results, local_file)

            key = "excel_files/myntra_products.xlsx"

            s3.upload_file(
                Filename=local_file,
                Bucket=S3_BUCKET,
                Key=key,
                ExtraArgs={'ContentType': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'}
            )
            response["result"] = results

        # Placeholder response (you can replace with actual analysis logic)
        response["url"] = f"https://alyaimg.s3.amazonaws.com/excel_files/{local_file}"
        return response
    
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    
@app.post("/get_number_of_offers")
async def get_number_of_offers(competitor_request: CompetitorRequest):
    marketplace = competitor_request.marketplace
    asins = competitor_request.asins
    response = {}
    if marketplace.lower() == "amazon":
        try:
            asin_list = [asin.strip() for asin in asins.split(",") if asin.strip()]
            result = await get_offer_counts(asin_list)
            response["result"] = result

            local_file = "amazon_offers.xlsx"

            write_offers_to_excel(result, local_file)

            key = "excel_files/amazon_offers.xlsx"

            s3.upload_file(
                Filename=local_file,
                Bucket=S3_BUCKET,
                Key=key,
                ExtraArgs={'ContentType': 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'}
            )
        except Exception as e:
            raise HTTPException(status_code=500, detail=str(e))
        
        response["url"] = f"https://alyaimg.s3.amazonaws.com/excel_files/{local_file}"
        return response
    else:
        raise HTTPException(status_code=400, detail="Marketplace not supported")
    


if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
