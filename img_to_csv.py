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
from typing import List,Dict,Any
import requests
import logging
import uuid
from pydantic import BaseModel
import chromadb
from chromadb.config import Settings
from datetime import date
import uvicorn
from PIL import Image
from io import BytesIO
from pymilvus import Collection
from typing import List
from utils import *
from prompts import *
from excel_fields import *
import httpx


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


API_KEY = os.getenv("API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(model_name="gemini-2.0-flash")

S3_BUCKET = os.getenv("S3_BUCKET", "alyaimg")

from image_search_engine import *

# Load model and DB once
clipmodel, processor, device = init_clip()
collection_db = init_milvus(
    host=os.getenv("MILVUS_HOST", "localhost"),
    port=os.getenv("MILVUS_PORT", "19530"),
    collection_name=os.getenv("COLLECTION_NAME", "image_embeddings")
)


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
    prompt = "Return the table containing product bills in a list of JSONs where each product details will be combined in one JSON. The columns will be the keys of the JSON. The rows should contain the details of products only. Don't repeat the rows if it is not repeated in the image. \n Keep the other details like transaction details, buyer, payer, total costs, etc. in the last JSON of the list. This JSON will have key-value pairs, each key and value must be valid entities (no empty strings, whitespace string or null value). There should be strictly one last JSON for the other details! Last json will be empty if no extra information \n Strictly Combine all the JSONs in a single list! Strictly Return a single list. Don't remove prefix or postfix from values. example :- 00001, $100, etc. should be displayed as it is. Enclose every value in double quotes.\nNo preambles or postambles i.e. the response should start with '[' and end with ']'\n"
    # Save the uploaded image to a file
    image_content = await image.read()
    image_name = f"{uuid.uuid4()}_{image.filename}"
    save_path = f"./{image_name}"
    with open(save_path, "wb") as f:
        f.write(image_content)
    # Send the image and prompt to the Gemini API
    sample_file = genai.upload_file(path=save_path,
                            display_name=image_name)
    

    # Prompt the model with text and the previously uploaded image.
    response = model.generate_content([sample_file, prompt],safety_settings={
        HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
        HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,

    })

    print(response.text)
    # Get the response idata from Gemini
    gemini_response = response.text
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
async def generate_caption(file: UploadFile = File(...),type: str = Form(...)):
        
        if not file:
            return {"error": "No files received"}
        
        paths = get_image_paths_from_s3(S3_BUCKET)
        index_images_from_s3(collection_db, paths, clipmodel, processor, device)

        image_bytes = await file.read()
        
        image_s = Image.open(BytesIO(image_bytes)).convert("RGB")

        query_emb = embed_image(image_s, clipmodel, processor, device)

    # Step 3: Search similar

        results = search_similar(collection_db, query_emb, 1)

    # Step 4: Load ID-to-path map
        int_hash_map_file = os.getenv("INT_HASH_MAP_FILE", "int_hash_to_path.json")
        try:
            with open(int_hash_map_file, 'r') as f:
                id_to_path = json.load(f)
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Failed to load hash map: {str(e)}"})

    # Step 5: Build response
        matches = []
        for _id, dist in results:
            if 100 - dist >= 90:
                path = id_to_path.get(str(_id), "unknown")
                matches.append({"id": _id, "distance": dist, "path": path})

        duplicate = False

        if matches:
            duplicate = True

        s3_url = ""
        if duplicate:
            s3_url = matches[0]["path"]
        

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
                s3.upload_file(
                    Filename=save_path,
                    Bucket=S3_BUCKET,
                    Key=image_name,
                    ExtraArgs={"ContentType": "image/jpeg"}
                )
                s3_url = f"https://{S3_BUCKET}.s3.amazonaws.com/{image_name}"
            except Exception as e:
                return JSONResponse(status_code=500, content={"error": f"Failed to upload to S3: {str(e)}"})


        # Now you can use `save_path` to upload the file to Gemini or process further
        sample_file = genai.upload_file(path=save_path, display_name=image_name)
        #ADD PREV PROMPT STYLES OF USER
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

        response0 = model.generate_content([sample_file, pp], safety_settings={
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }).text

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
        os.remove(save_path)

        
        if duplicate:
            return {
                "display_name": image_name,
                "generated_name": js["name"],
                "quality": js["quality"],
                "description": js["final_caption"],
                "attributes": js["attributes"],
                "prompt": js["prompt"],
                "color" : js["color"],
                "s3_url": js["s3_url"],
                "duplicate": "duplicate found"
            }
        else:
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
async def image_searh(file: UploadFile = File(...), top_k: int = 1):
    # Step 1: Re-index all images (only new or deleted ones will be changed)
    paths = get_image_paths_from_s3(S3_BUCKET)
    index_images_from_s3(collection_db, paths, clipmodel, processor, device)

    # Step 2: Read query image and embed it
    try:
        image = Image.open(BytesIO(await file.read())).convert("RGB")
    except Exception as e:
        return JSONResponse(status_code=400, content={"error": f"Invalid image: {str(e)}"})

    query_emb = embed_image(image, clipmodel, processor, device)

    # Step 3: Search similar

    results = search_similar(collection_db, query_emb, top_k)

    # Step 4: Load ID-to-path map
    int_hash_map_file = os.getenv("INT_HASH_MAP_FILE", "int_hash_to_path.json")
    try:
        with open(int_hash_map_file, 'r') as f:
            id_to_path = json.load(f)
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Failed to load hash map: {str(e)}"})

    # Step 5: Build response
    matches = []
    for _id, dist in results:
        if 100 - dist >= 90:
            path = id_to_path.get(str(_id), "unknown")
            matches.append({"id": _id, "distance": dist, "path": path})

    return {"results": matches}

class CatalogRequest(BaseModel):
    image_urls: str
    type: str
    marketplace: str
    skuids: str


@app.post("/catalog-ai")
async def catalog_ai(req: CatalogRequest):

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
                "filename": url,
                "description": description,
                "skuid": skuid
            }
            
            final_response = {**response_json, **fixed_values , **dict}

            filename_map = {
                "fli_ear": "earrings_flipkart.xlsx",
                "ama_ear": "earrings_amz.xlsx",
                "mee_ear": "earrings_meesho.xlsx",
                "fli_nec": "necklace_flipkart.xlsx",
                "ama_nec": "necklace_amz.xlsx",
                "mee_nec": "necklace_meesho.xlsx",
                "fli_bra": "bracelet_flipkart.xlsx",
                "mee_bra": "bracelet_meesho.xlsx"
            }

            static_file_name = filename_map.get(format)
            static_file_path = os.path.join("static", static_file_name)
            static_url = f"http://15.206.26.88:8000/static/{static_file_name}"  

            excel_results.append((skuid,response_json,description))

            results.append({
                "attributes": final_response,
            })
    

    if format == "fli_ear":
        write_to_excel_flipkart(excel_results, filename=static_file_path, target_fields=target_fields_earrings, fixed_values=fixed_values_earrings)
    elif format == "ama_ear":
        write_to_excel_amz_xl(excel_results, filename=static_file_path, target_fields=target_fields_earrings_amz, fixed_values=fixed_values_earrings_amz)
    elif format == "mee_ear":
        write_to_excel_meesho(excel_results, filename=static_file_path, target_fields=target_fields_earrings_meesho, fixed_values=fixed_values_earrings_meesho)
    elif format == "fli_nec":
        write_to_excel_flipkart(excel_results, filename=static_file_path, target_fields=target_fields_necklace_flipkart, fixed_values=fixed_values_necklace_flipkart)
    elif format == "ama_nec":
        write_to_excel_amz_xl(excel_results, filename=static_file_path, target_fields=target_fields_necklace_amz, fixed_values=fixed_values_necklace_amz)
    elif format == "mee_nec":
        write_to_excel_meesho(excel_results, filename=static_file_path, target_fields=target_fields_necklace_meesho, fixed_values=fixed_values_necklace_meesho)
    elif format == "fli_bra":
        write_to_excel_flipkart(excel_results, filename=static_file_path, target_fields=target_fields_bracelet_flipkart, fixed_values=fixed_values_bracelet_flipkart)
    elif format == "mee_bra":   
        write_to_excel_meesho(excel_results, filename=static_file_path, target_fields=target_fields_bracelet_meesho, fixed_values=fixed_values_bracelet_meesho)


    return {"results": results , "excel_file": static_url}


@app.post("/clear-excel/")
def clear_excel_file(filename: str = Form(...)):
    static_dir = "static"
    file_path = file_path = os.path.join("api_sever_1", "static", filename)

    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File not found")

    # Determine starting row based on filename
    filename_lower = filename.lower()
    sheet_number = 0
    if "amz" in filename_lower or "ama" in filename_lower:
        start_row = 4
        sheet_number = 0
    elif "fli" in filename_lower:
        start_row = 5
        sheet_number = 2
    elif "mee" in filename_lower:
        start_row = 5
        sheet_number = 1
    else:
        raise HTTPException(status_code=400, detail="Unsupported file type")

    try:
        wb = openpyxl.load_workbook(file_path)
        ws = wb.worksheets[sheet_number]

        max_row = ws.max_row
        if max_row >= start_row:
            ws.delete_rows(start_row, max_row - start_row + 1)

        wb.save(file_path)
        wb.close()

        return {"message": f"Cleared rows from row {start_row} in {filename}"}

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to clear Excel file: {str(e)}")



if __name__ == "__main__":
    uvicorn.run(app, host="localhost", port=8000)
