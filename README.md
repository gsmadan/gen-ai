# 🤖 Gemini-Powered RAG FAQ Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers product-related queries using FAQs generated from Amazon review data. Powered by **Gemini 2.0 Flash**, FAISS vector search, and **LangChain's ConversationalRetrievalChain**, this chatbot supports multi-turn Q&A and source-grounded answers.
---

## 🔍 Features

- ✨ **Gemini 2.0 Flash** for conversational, grounded answers
- 🔎 **FAISS vector search** to find the most relevant FAQ chunks
- 💬 Multi-turn chat with LangChain's `ConversationalRetrievalChain`
- 📚 **CSV-based ingestion** from auto-generated FAQs
- 💡 **Google Generative AI embeddings** (`models/text-embedding-004`)
- 🔐 `.env` for secure API key management
- 🧠 Built with **LangChain**

---

## 🧱 Project Structure

faq-chatbot-rag/
│
├── data/
│ └── generated_faqs_gemini.csv # CSV of FAQ blocks
│
├── ingest_faqs.py # Converts CSV to FAISS index
├── chatbot_rag.py # Loads index + runs chatbot, with conversational memory
├── .env # Gemini API key
├── requirements.txt
└── README.md

📌 Technologies Used
LangChain
Gemini 2.0 Flash
Google Embeddings (models/text-embedding-004)
FAISS
Google Generative AI SDK
Python 3.9+
