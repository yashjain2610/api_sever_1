import streamlit as st
import requests
import os
import pandas as pd

def further_req(text, prompt):
    text += "\n"
    text += prompt
    text += "Give output formatted as json list where the columns will be the elements of the JSON. \nNo preambles or postambles i.e. the response should start with '[' and end with ']'\n"
    response = requests.post("http://127.0.0.1:8000/further_req/", data={'prompt': text})
    return response.content

# Streamlit UI
st.title("CREATE CSV FROM BILL")

# File uploader for the image
uploaded_image = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

if uploaded_image is not None:
    st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

# Initialize session state variables
if "response" not in st.session_state:
    st.session_state.response = None
if "df" not in st.session_state:
    st.session_state.df = None

submitted = st.button("Submit")

# Submit button
if submitted:
    if uploaded_image is None:
        st.error("Please upload an image.")
    else:
        # Construct the prompt
        prompt = "Return the table containing product bills in a list of JSONs where each product details will be combined in one JSON. The columns will be the keys of the JSON. \n Keep the other details like transaction details, buyer, payer, total costs, etc. in the last JSON of the list. This JSON will have key-value pairs, each key and value must be valid entities (no empty strings, whitespace string or null value). There should be strictly one last JSON for the other details! \n Strictly Combine all the JSONs in a single list! Strictly Return a single list. Don't remove prefix or postfix. example :- 00001, $100, etc. Enclose every value in double quotes.\nNo preambles or postambles i.e. the response should start with '[' and end with ']'\n"
        
        # Prepare form data for FastAPI request
        files = {'image': uploaded_image}
        data = {'prompt': prompt}
        
        # Send a POST request to FastAPI
        response = requests.post("http://127.0.0.1:8000/process-image-and-prompt/", files=files, data=data)
        r2 = requests.get("http://127.0.0.1:8000/download/")
        if response.status_code == 200:
            # Save the CSV file locally
            
            output_file = "output.csv"
            output_extra="output_extra.csv"
            with open(output_file, "wb") as f:
                f.write(response.content)
            with open(output_extra, "wb") as f:
                f.write(r2.content)
            # Store response and DataFrame in session state
            st.session_state.response = response
            st.session_state.df = pd.read_csv(output_file)
            st.session_state.df_extra = pd.read_csv(output_extra)
            st.header("BILL")
            st.table(st.session_state.df)
            with open(output_file, "rb") as f:
                st.download_button("Download Bill CSV", f, file_name="gemini_output.csv", mime="text/csv")
            st.header("EXTRA DETAILS")
            st.table(st.session_state.df_extra)

            # Provide a download button for the user to download the CSV
            
            with open(output_extra, "rb") as f:
                st.download_button("Download Extra Details CSV", f, file_name="output_extra.csv", mime="text/csv")
            # Optionally clean up the file after download
            os.remove(output_file)
            # os.remove(output_extra)

# Check if the response exists to show modification options
if st.session_state.df is not None:
    options = st.session_state.df.columns

    # Store the selected options in a list
    selected_options = st.multiselect('Select Columns from the MAIN BILL:', options)
    column_names = ",".join(selected_options)
    print(column_names)
    print(type(column_names))
    row_filter = st.text_input("Enter row filter (optional)")
    # add_info = st.text_input("Enter additional information (optional)")
    sec_req = st.button("Modify")

    if sec_req:
        new_prompt = "Modify the previous response based on the following criteria:\n"
        if row_filter:
            new_prompt += f"Select only those entries which follow: {row_filter}\n"
        if column_names:
            new_prompt += f"Only select the following columns: {column_names}\n"
        # if add_info:
        #     new_prompt += f"Additional Information: {add_info}\n"
        
        # Call the further_req function with the new prompt
        # print(type(st.session_state.response.content.decode()))
        new_res = further_req(st.session_state.response.content.decode(), new_prompt)  # Decode response.content
        output_file = "final.csv"
        with open(output_file, "wb") as f:
            f.write(new_res)
        
        # Load the new response into a DataFrame
        df2 = pd.read_csv(output_file)
        st.header("MODIFIED BILL")
        st.table(df2)
        with open(output_file, "rb") as f:
            st.download_button("Download Modified CSV", f, file_name="gemini_final_output.csv", mime="text/csv")
        st.header("EXTRA DETAILS")
        st.table(st.session_state.df_extra)

        # Provide a download button for the user to download the CSV
        
        with open("output_extra.csv", "rb") as f:
            st.download_button("Download Extra Details CSV", f, file_name="output_extra.csv", mime="text/csv")
        # Optionally clean up the file after download
        os.remove(output_file)
        os.remove("output_extra.csv")