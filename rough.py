import google.generativeai as genai
import os
from dotenv import load_dotenv
load_dotenv()
API_KEY = os.getenv("API_KEY")
genai.configure(api_key=API_KEY)
model = genai.GenerativeModel(model_name="gemini-1.5-flash")
sample_file = genai.upload_file(path="save.jpg",
                            display_name="Vendor Bill")
print(sample_file)