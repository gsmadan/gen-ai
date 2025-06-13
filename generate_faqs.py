import pandas as pd
import google.generativeai as genai
from tqdm import tqdm
from dotenv import load_dotenv
import os
import time

# Load API key
load_dotenv()
genai.configure(api_key=os.getenv("GEMINI_API_KEY"))

# Load dataset
df = pd.read_excel("data/amazon_reviews.xlsx")
df = df[['name', 'reviews.text']].dropna()
df = df.groupby('name')['reviews.text'].apply(lambda x: ' '.join(x[:50])).reset_index()

# Create Gemini model instance
model = genai.GenerativeModel('gemini-2.0-flash')

faq_data = []

# FAQ prompt template
def make_prompt(product, reviews):
    return f"""
You are a helpful customer support assistant.

Based on these real customer reviews of **{product}**, generate 3 to 5 frequently asked questions and their answers.

Reviews:
{reviews}

Format:
Q1: ...
A1: ...
Q2: ...
A2: ...
"""

for _, row in tqdm(df.iterrows(), total=len(df)):
    product = row['name']
    reviews = row['reviews.text']
    prompt = make_prompt(product, reviews)

    try:
        response = model.generate_content(prompt)
        faqs = response.text
        faq_data.append({'product_name': product, 'faqs': faqs})
        time.sleep(1)  # be nice to the API

    except Exception as e:
        print(f"Error with {product}: {e}")
        continue

# Save output
pd.DataFrame(faq_data).to_csv("data/generated_faqs_gemini.csv", index=False)
print("✅ FAQ generation complete and saved.")
