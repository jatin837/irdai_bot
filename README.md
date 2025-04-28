# IRDAI Bot 🚀

A **Retriever-Augmented Generation (RAG)** based bot for **IRDAI** (Insurance Regulatory and Development Authority of India) regulations using **FAISS** for fast document retrieval and **Ollama Mistral** for generating accurate and context-aware responses. 🤖

## Features 🌟
- **FAISS-based Embedding Search**: Efficient retrieval of relevant IRDAI-related documents using FAISS for fast and accurate searches.
- **Chunking & Preprocessing**: Documents are chunked, cleaned (removing redundant formatting and `NaN` values), and indexed for efficient retrieval.
- **RAG Architecture**: Combines retrieval of document chunks and generative text processing for smart, human-like answers.
- **Modular Design**: Easy to extend with new data sources, enhanced chunking logic, and more efficient retrieval mechanisms in future updates.

## Technologies Used 🛠️
- **FAISS**: For vector-based document retrieval and similarity search.
- **Ollama Mistral**: For natural language understanding and generation.
- **Python**: Backend development and scripting.
- **NumPy / Pandas**: For data processing and embedding handling.
- **Anaconda/Virtual Environments**: Managing dependencies in isolated environments.