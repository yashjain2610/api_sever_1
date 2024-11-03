import streamlit as st
import requests
import os
import pandas as pd

def further_req(text, prompt):
    text += "\n"
    text += prompt
    text += "Give output formatted as json list where the columns will be the elements of the JSON\nNo preambles or postambles i.e. the response should start with '[' and end with ']'\n"
    response = requests.post("https://python-intern.onrender.com/further_req/", data={'prompt': text})
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
        prompt = "Return the table in JSON list format where the columns will be the elements of the JSON\nNo preambles or postambles i.e. the response should start with '[' and end with ']'\n"
        
        # Prepare form data for FastAPI request
        files = {'image': uploaded_image}
        data = {'prompt': prompt}
        
        # Send a POST request to FastAPI
        response = requests.post("https://python-intern.onrender.com/process-image-and-prompt/", files=files, data=data)
        
        if response.status_code == 200:
            # Save the CSV file locally
            output_file = "output.csv"
            with open(output_file, "wb") as f:
                f.write(response.content)

            # Store response and DataFrame in session state
            st.session_state.response = response
            st.session_state.df = pd.read_csv(output_file)
            st.table(st.session_state.df)

            # Provide a download button for the user to download the CSV
            with open(output_file, "rb") as f:
                st.download_button("Download CSV", f, file_name="gemini_output.csv", mime="text/csv")
            
            # Optionally clean up the file after download
            os.remove(output_file)

# Check if the response exists to show modification options
if st.session_state.df is not None:
    options = st.session_state.df.columns

    # Store the selected options in a list
    selected_options = st.multiselect('Select options:', options)
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
        st.table(df2)

        # Provide a download button for the user to download the CSV
        with open(output_file, "rb") as f:
            st.download_button("Download Modified CSV", f, file_name="gemini_final_output.csv", mime="text/csv")
        
        # Optionally clean up the file after download
        os.remove(output_file)
