from openai import OpenAI
import os
import base64
import requests

client = OpenAI()

# OpenAI API Key
api_key = os.getenv("OPENAI_API_KEY")

# Function to encode the image
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

# function to generate embedding from a given text
def generate_embedding(text, model="text-embedding-3-small"):
    response = client.embeddings.create(input=text, model="text-embedding-3-small")
    return response.data[0].embedding

# function to generate description from a given image
def generate_description(image_path, system_prompt, model="gpt-4o-mini"):
    # Getting the base64 string
    base64_image = encode_image(image_path)

    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {api_key}"
    }

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": system_prompt
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{base64_image}"
                        }
                    }
                ]
            }
        ],
        "max_tokens": 1000
    }

    try:
        response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
        response.raise_for_status()  # This will raise an HTTPError if the HTTP request returned an unsuccessful status code
        return response.json()['choices'][0]['message']['content']

    except requests.exceptions.RequestException as e:
        print(f"Error generating description: {e}")
        return None