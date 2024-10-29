from fastapi import FastAPI, File, UploadFile, Form
from fastapi.responses import FileResponse, JSONResponse,StreamingResponse
import csv
import io
# import requests
# import base64
import os
from dotenv import load_dotenv
from fastapi.middleware.cors import CORSMiddleware
# from pymongo.mongo_client import MongoClient
# from gridfs import GridFS
import json
# from io import BytesIO
import google.generativeai as genai
# from IPython.display import Markdown
from typing import List

app = FastAPI()
load_dotenv()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace with your Streamlit app's URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# uri = "mongodb+srv://aryan22102:8UEMkbvn1X6sWiOs@cluster0.xf4bl.mongodb.net/?retryWrites=true&w=majority&appName=Cluster0"


# # MongoDB Setup (replace with your MongoDB URI)
# client = MongoClient(uri)
# try:
#     client.admin.command('ping')
#     print("Pinged your deployment. You successfully connected to MongoDB!")
# except Exception as e:
#     print(e)
# db = client["image_database"]
# fs = GridFS(db)




# Gemini API endpoint and headers (replace with actual API endpoint and key)
# GEMINI_API_URL = "https://actual.gemini.api.endpoint/v1/process"
API_KEY = os.getenv("API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")






@app.post('/further_req')
async def further_req(prompt: str=Form(...)):

    response = model.generate_content([ prompt])
    print(response)
    # Get the response data from Gemini
    gemini_response = response.text
    print(gemini_response)
    my_dict = json.loads(gemini_response)
    # Create a CSV in-memory file
    csv_file = io.StringIO()
    writer = csv.writer(csv_file)
    # Dynamically get field names from the first item of the response
    if my_dict:
        header = my_dict[0].keys()  # Get field names from the first item
        writer.writerow(header)
        # Write each row based on fields dynamically
        for item in my_dict:
            writer.writerow([item[field] for field in header])

    # Save the CSV content
    csv_file.seek(0)  # Move back to the start of the file for reading

    # Save CSV to a temporary file on disk for downloading
    temp_csv_file = f"gemini_output_bill.csv"
    with open(temp_csv_file, "w", newline='') as f:
        f.write(csv_file.getvalue())    
    
    ret=FileResponse(temp_csv_file, media_type="text/csv", filename=temp_csv_file)
    # os.remove(temp_csv_file)
    # Return the CSV file as a downloadable response
    return ret

@app.post("/process-image-and-prompt/")
async def process_image_and_prompt(image: UploadFile = File(...), prompt: str = Form(...)):
    
    
    # Save the uploaded image to a file
    image_content = await image.read()
    with open("save.jpg", "wb") as f:
        f.write(image_content)
    # Send the image and prompt to the Gemini API
    sample_file = genai.upload_file(path="save.jpg",
                            display_name="Vendor Bill")
    

    # Prompt the model with text and the previously uploaded image.
    response = model.generate_content([sample_file, prompt])


    # if response.status_code != 200:
    #     return JSONResponse(status_code=response.status_code, content={"error": "Failed to get response from Gemini API"})
    print(response)
    # Get the response data from Gemini
    gemini_response = response.text
    print(gemini_response)
    my_dict = json.loads(gemini_response)

    # Extract relevant data from the response (adjust based on your needs)
    

    # Create a CSV in-memory file
    csv_file = io.StringIO()
    writer = csv.writer(csv_file)

    # Write headers
    # Dynamically get field names from the first item of the response
    if my_dict:
        header = my_dict[0].keys()  # Get field names from the first item
        writer.writerow(header)

        # Write each row based on fields dynamically
        for item in my_dict:
            writer.writerow([item[field] for field in header])

    # Save the CSV content
    csv_file.seek(0)  # Move back to the start of the file for reading

    # Save CSV to a temporary file on disk for downloading
    temp_csv_file = f"gemini_output_bill.csv"
    with open(temp_csv_file, "w", newline='') as f:
        f.write(csv_file.getvalue())
    
    ret=FileResponse(temp_csv_file, media_type="text/csv", filename=temp_csv_file)
    # os.remove("save.jpg")
    # os.remove(temp_csv_file)
    # os.remove(temp_csv_file)
    # Return the CSV file as a downloadable response
    return ret


    




# @app.post("/upload_image")
# async def upload_image(user_id: str = Form(...), file: UploadFile = File(...)):
#     try:
#         # Save image to GridFS (MongoDB)
#         existing_image = db.fs.files.find_one({"metadata.user_id": user_id, "filename": file.filename})
        
#         if not existing_image:
            
#             image_id = fs.put(file.file, filename=file.filename, metadata={"user_id": user_id})
#             return {"message": f"Image {file.filename} saved successfully"}
#         else:
#             return JSONResponse(status_code=500, content={"message": f"Image {file.filename} already exists."})
#     except :
#         return JSONResponse(status_code=500, content={"message": f"Image {file.filename} already exists."})



# Route to generate captions
@app.post("/generate_caption")
async def generate_caption(files: List[UploadFile] = File(...)):
    try:
        # Retrieve all images for the user
        # images = db.fs.files.find({"metadata.user_id": user_id})
        # image_data = []
        final=[]
        # print(images.count())
        # Send each image to Gemini API for caption generatio
        nam=[]
        for image in files:
            image_name = image.filename
            nam.append(image_name)
            # Create the path to save the file using the filename
            save_path = f"./{image_name}"
            
            # Save the uploaded image directly to the disk
            with open(save_path, "wb") as f:
                f.write(await image.read())  # Write the image data to the file

            # Now you can use `save_path` to upload the file to Gemini or process further
            sample_file = genai.upload_file(path=save_path, display_name=image_name)
            final.append(sample_file)
            os.remove(save_path)
        
        final.append(f"please generate a fine description for each image of given jwellery. Give output formatted as json list where the elements will be display_names and corresponding generated caption\nNo preambles or postambles i.e. the response should start with '[' and end with ']'\n")
        print(len(final),"  \n ",final)
        response = model.generate_content(final)
        print(response)
        # Get the response data from Gemini
        gemini_response = response.text
        print(gemini_response)
        print(type(gemini_response),type(gemini_response[0]))
        print(nam)
        
        my_dict = json.loads(gemini_response)
        for i in range(len(nam)):
            my_dict[i]["display_name"]=nam[i]
        # Create a CSV in-memory file
        csv_file = io.StringIO()
        writer = csv.writer(csv_file)
        # Dynamically get field names from the first item of the response
        if my_dict:
            header = my_dict[0].keys()
            writer.writerow(header)

        # Write each row based on fields dynamically
            for item in my_dict:
                writer.writerow([item[field] for field in header])
        # Save the CSV content
        csv_file.seek(0)  # Move back to the start of the file for reading

        # Save CSV to a temporary file on disk for downloading
        temp_csv_file = f"jwelellry.csv"
        with open(temp_csv_file, "w", newline='') as f:
            f.write(csv_file.getvalue())
        
        ret=FileResponse(temp_csv_file, media_type="text/csv", filename=temp_csv_file)
        # images = db.fs.files.find({"metadata.user_id": user_id})
        
        # # Delete each file and its associated chunks
        # for image in images:
        #     fs.delete(image["_id"]) 
        # os.remove(temp_csv_file)
        # Return the CSV file as a downloadable response
        # os.remove(temp_csv_file)
        
        return ret
        
        
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"Error generating captions: {str(e)}"})



# @app.get("/get_image")
# async def generate_caption(user_id: str,filename:str):
#     try:
#         # Retrieve all images for the user
#         images = db.fs.files.find({"metadata.user_id": user_id, "filename": filename})
#         if not images:
#             return JSONResponse(status_code=404, content={"message": "Image not found."})
        
#         # Retrieve image data from GridFS
#         file_data = fs.get(images["_id"]).read()

#         # Create a BytesIO stream from the file data
#         image_stream = BytesIO(file_data)

#         # Return image as a StreamingResponse
#         return StreamingResponse(image_stream, media_type="image/jpeg")
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"message": f"Error getting image: {str(e)}"})
    

# @app.delete("/delete_images")
# async def delete_images(user_id: str):
#     try:
#         # Find all files associated with the user_id
#         images = db.fs.files.find({"metadata.user_id": user_id})
        
#         # Delete each file and its associated chunks
#         for image in images:
#             fs.delete(image["_id"])  # Deletes the file and associated chunks
        
#         return {"message": f"All images for user_id: {user_id} have been deleted successfully."}
    
#     except Exception as e:
#         return JSONResponse(status_code=500, content={"message": f"Error deleting images: {str(e)}"})