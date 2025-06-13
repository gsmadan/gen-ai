# 🤖 Gemini-Powered RAG FAQ Chatbot

A Retrieval-Augmented Generation (RAG) chatbot that answers natural language queries using product FAQs generated from Amazon customer reviews. It uses Google Gemini for response generation and FAISS for fast similarity-based document retrieval.

---

## 🔍 Features

- ✨ **Gemini 1.5 Flash** for conversational, grounded answers
- 🔎 **FAISS vector search** to find the most relevant FAQ chunks
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
├── chatbot_rag.py # Loads index + runs chatbot
├── .env # Gemini API key
├── requirements.txt
└── README.md

📌 Technologies Used
LangChain
Gemini 1.5 Flash
FAISS
Google Generative AI SDK
Python 3.9+
