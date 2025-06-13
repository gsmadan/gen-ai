import os
from dotenv import load_dotenv
from langchain_community.vectorstores import FAISS
from langchain_google_genai.embeddings import GoogleGenerativeAIEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQA

# Load .env for API key
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")

# Set up the same embedding model used during indexing
embedding = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",  # ✅ matches FAISS index
    google_api_key=api_key
)

# Load the saved FAISS vector index
vectorstore = FAISS.load_local("faiss_index", embeddings=embedding,
    allow_dangerous_deserialization=True  # ✅ explicitly allow pickle loading
)


# Set up Gemini 2.0 Flash chat model
llm = ChatGoogleGenerativeAI(
    model="gemini-2.0-flash",
    google_api_key=api_key,
    temperature=0.4
)

# Create RetrievalQA chain (RAG pipeline)
qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    retriever=vectorstore.as_retriever(),
    return_source_documents=True
)

# Interactive CLI chatbot
print("\n🤖 Gemini Support Chatbot (type 'exit' to quit)")
while True:
    user_input = input("\nYou: ")
    if user_input.lower() in ["exit", "quit"]:
        print("Goodbye 👋")
        break

    try:
        response = qa_chain(user_input)
        answer = response["result"]
        print(f"\nGemini: {answer}")
    except Exception as e:
        print(f"\n[Error] {e}")
