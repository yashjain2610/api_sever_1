import streamlit as st
import os
import base64
import openpyxl

from openpyxl import Workbook

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



def write_to_excel_meesho(results,filename,target_fields,fixed_values):
    if os.path.exists(filename):
        #print("started loading")
        #print()
        wb = openpyxl.load_workbook(filename)
        #print("finished")
        #print()
        ws = wb.worksheets[1]

        # Read headers dynamically from row 1
        headers = {cell.value: cell.column for cell in ws[3] if cell.value}
        headers = {key.strip().split('\n\n')[0]: value for key, value in headers.items()}

        #print(headers)
        #print()
    else:
        wb = Workbook()
        ws = wb.active  

        # Define default headers
        default_headers = [
            "Image Name", "Type", "Color", "Gemstone", "Pearl Type",
            "Collection", "Occasion", "Piercing Required", "Number of Gemstones",
            "Earring Back Type", "Finish", "Design", "Metal Color", "Diamond Type",
            "Pearl Shape", "Pearl Color", "Search Keywords", "Key Features", "Trend",
            "Closure Type", "Sub Type", "Shape", "Ear Chain", "Earring Set Type",
            "Ornamentation Type", "Trend"
        ]
        
        headers = {name: idx + 1 for idx, name in enumerate(default_headers)}
        ws.append(default_headers)  # Write headers in row 1 if creating a new file

    #print()
    # Combine target fields and fixed value fields
    all_fields = set(target_fields).union(fixed_values.keys())

    # Get column indices for all relevant fields
    target_columns = {field: headers[field] for field in all_fields if field in headers}
    # print(target_columns)

    # **Write output from row 6 onwards**
    row_idx = 1
    while ws.cell(row=row_idx, column=1).value:  # Assuming column 1 is "Image Name" or always filled
        row_idx += 1

    for image_name, response , description in results:
        #print(response)
        # answers = response.split("\n")  
        # answers = [ans.strip() for ans in answers]
        # answers.insert(0, image_name)

        # print()
        # print(answers)  
        # print()

        # Create a dictionary mapping field names to values
        # print(target_fields)
        # field_values = {field: answers[i] if i < len(answers) else "" for i, field in enumerate(target_fields)}
        # print()
        # print(field_values)
        field_values = response

        # Write values only in the specified fields
        for key, val in fixed_values.items():
            field_values[key] = val

        field_values["Product Description"] = description
        field_values["Fields + Description:"] = os.path.splitext(image_name)[0]

        for field, col_idx in target_columns.items():
            ws.cell(row=row_idx, column=col_idx, value=field_values.get(field, ""))

        row_idx += 1  # Move to the next row

    wb.save(filename)


def write_to_excel_flipkart(results,filename,target_fields,fixed_values):
    if os.path.exists(filename):
        #print("started loading")
        #print()
        wb = openpyxl.load_workbook(filename)
        #print("finished")
        #print()
        ws = wb.worksheets[2]

        # Read headers dynamically from row 1
        headers = {cell.value: cell.column for cell in ws[1] if cell.value}
        #print(headers)
        #print()
    else:
        wb = Workbook()
        ws = wb.active  

        # Define default headers
        default_headers = [
            "Image Name", "Type", "Color", "Gemstone", "Pearl Type",
            "Collection", "Occasion", "Piercing Required", "Number of Gemstones",
            "Earring Back Type", "Finish", "Design", "Metal Color", "Diamond Type",
            "Pearl Shape", "Pearl Color", "Search Keywords", "Key Features", "Trend",
            "Closure Type", "Sub Type", "Shape", "Ear Chain", "Earring Set Type",
            "Ornamentation Type", "Trend"
        ]
        
        headers = {name: idx + 1 for idx, name in enumerate(default_headers)}
        ws.append(default_headers)  # Write headers in row 1 if creating a new file

    #print()
    # Combine target fields and fixed value fields
    all_fields = set(target_fields).union(fixed_values.keys())

    # Get column indices for all relevant fields
    target_columns = {field: headers[field] for field in all_fields if field in headers}
    # print(target_columns)

    # **Write output from row 6 onwards**
    row_idx = 1
    while ws.cell(row=row_idx, column=7).value:  # Assuming column 1 is "Image Name" or always filled
        row_idx += 1
    #print(row_idx)  
    for image_name, response , description in results:
        field_values = response

        # Write values only in the specified fields
        for key, val in fixed_values.items():
            field_values[key] = val

        field_values["Description"] = description
        field_values["Seller SKU ID"] = os.path.splitext(image_name)[0]

        for field, col_idx in target_columns.items():
            ws.cell(row=row_idx, column=col_idx, value=field_values.get(field, ""))

        row_idx += 1  # Move to the next row

    wb.save(filename)


def write_to_excel_amz_xl(results,filename,target_fields,fixed_values):
    if os.path.exists(filename):
        #print("started loading")
        #print()
        wb = openpyxl.load_workbook(filename)
        #print("finished")
        #print()
        ws = wb.worksheets[0]

        # Read headers dynamically from row 1
        headers = {cell.value: cell.column for cell in ws[3] if cell.value}
        #print(headers)
        #print()
    else:
        wb = Workbook()
        ws = wb.active  

        # Define default headers
        default_headers = [
            "Image Name", "Type", "Color", "Gemstone", "Pearl Type",
            "Collection", "Occasion", "Piercing Required", "Number of Gemstones",
            "Earring Back Type", "Finish", "Design", "Metal Color", "Diamond Type",
            "Pearl Shape", "Pearl Color", "Search Keywords", "Key Features", "Trend",
            "Closure Type", "Sub Type", "Shape", "Ear Chain", "Earring Set Type",
            "Ornamentation Type", "Trend"
        ]
        
        headers = {name: idx + 1 for idx, name in enumerate(default_headers)}
        ws.append(default_headers)  # Write headers in row 1 if creating a new file

    #print()
    # Combine target fields and fixed value fields
    all_fields = set(target_fields).union(fixed_values.keys())

    # Get column indices for all relevant fields
    target_columns = {field: headers[field] for field in all_fields if field in headers}
    #print(target_columns)

    # **Write output from row 6 onwards**
    row_idx = 1
    while ws.cell(row=row_idx, column=2).value:  # Assuming column 1 is "Image Name" or always filled
        row_idx += 1
    #print(row_idx)  
    for image_name, response , description in results:
        # print(response)
        # answers = response.split("\n")  
        # answers = [ans.strip() for ans in answers]
        # answers.insert(0, os.path.splitext(image_name)[0])

        # print()
        # print(answers)  
        # print()

        # Create a dictionary mapping field names to values
        # print(target_fields)
        # field_values = {field: answers[i] if i < len(answers) else "" for i, field in enumerate(target_fields)}
        # print()
        # print(field_values)
        field_values = response

        # Write values only in the specified fields
        for key, val in fixed_values.items():
            field_values[key] = val

        field_values["product_description"] = description
        field_values["item_sku"] = os.path.splitext(image_name)[0]

        for field, col_idx in target_columns.items():
            ws.cell(row=row_idx, column=col_idx, value=field_values.get(field, ""))

        row_idx += 1  # Move to the next row

    wb.save(filename)