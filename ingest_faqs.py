import os
from dotenv import load_dotenv
import pandas as pd
from langchain_community.vectorstores import FAISS
# from langchain.document_loaders import CSVLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain.docstore.document import Document

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Set up Gemini embedding model
embedding = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=api_key)

# Load the CSV
csv_path = "data/generated_faqs_gemini.csv"
df = pd.read_csv(csv_path)

# Extract documents from the 'faqs' column
docs = [Document(page_content=row['faqs'], metadata={"product": row['product_name']}) for _, row in df.iterrows()]

# Optional: Split long FAQ blocks into smaller chunks
splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
split_docs = splitter.split_documents(docs)

# Create FAISS index
vectorstore = FAISS.from_documents(split_docs, embedding)

# Save index to local folder
faiss_path = "faiss_index"
vectorstore.save_local(faiss_path)

print(f"✅ FAISS index created and saved to `{faiss_path}`")
