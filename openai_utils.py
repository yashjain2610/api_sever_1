from openai import OpenAI
import base64
import json
import os

API_KEY = os.getenv("API_KEY_GPT")
proj_id = os.getenv("PROJ_ID")
client = OpenAI(
    api_key=API_KEY,
    project=proj_id
    )
def ask_gpt_with_image(prompt: str, image_path: str = None, image_bytes: bytes = None):
    """
    Send a structured prompt + image to GPT-4.1 Vision and return ONLY the JSON response.
    """

    if not image_path and not image_bytes:
        raise ValueError("Provide either image_path or image_bytes.")

    # Convert image to base64
    if image_path:
        with open(image_path, "rb") as f:
            img_b64 = base64.b64encode(f.read()).decode("utf-8")
    else:
        img_b64 = base64.b64encode(image_bytes).decode("utf-8")

    # Make API request
    completion = client.chat.completions.create(
        model="gpt-4.1",
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{img_b64}"
                        }
                    }
                ],
            }
        ],
        response_format={"type": "json_object"}  # force valid JSON
    )

    # Extract ONLY JSON content
    raw_json = completion.choices[0].message.content
    print(raw_json)

    # Convert to dict and return
    try:
        return json.loads(raw_json)
    except json.JSONDecodeError:
        # If model returned imperfect json, still return raw text
        return {"error": "Invalid JSON", "raw": raw_json}