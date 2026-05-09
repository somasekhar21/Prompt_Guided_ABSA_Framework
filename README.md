# Prompt Guided ABSA Framework

A powerful, modular Aspect-Based Sentiment Analysis (ABSA) framework built with **LangGraph**, **Ollama**, and **ChromaDB**. This project implements a sophisticated pipeline to extract specific aspects from text and determine their corresponding sentiments using Large Language Models (LLMs).

## 🚀 Overview

This framework moves beyond general sentiment analysis by identifying specific features (aspects) of a product or service mentioned in a sentence and assigning a sentiment to each. It uses a state-of-the-art graph-based workflow to ensure high accuracy and structured output.

### Key Features
- **Graph-Based Pipeline**: Powered by LangGraph for predictable, multi-step processing.
- **Pre-processing**: Automatic emoji normalization and POS-tagging-based keyword extraction.
- **LLM Guided**: Uses Ollama (Llama/Mistral) for intelligent keyword extraction and aspect mapping.
- **Vector Search Integration**: Handles "unknown" or ambiguous aspects by storing them in ChromaDB for later clustering or retrieval.
- **Structured Output**: Uses Pydantic for strict data validation and parsing.

## 🛠️ Tech Stack

- **Core**: Python 3.13+
- **Orchestration**: [LangGraph](https://github.com/langchain-ai/langgraph)
- **LLM Integration**: [LangChain](https://github.com/langchain-ai/langchain) & [Ollama](https://ollama.ai/)
- **Vector Store**: [ChromaDB](https://www.trychroma.com/)
- **NLP**: SpaCy, NLTK
- **Embeddings**: Sentence-Transformers
- **API**: FastAPI & Uvicorn

## 📋 Prerequisites

1. **Python**: Version 3.13 or higher.
2. **Ollama**: Install [Ollama](https://ollama.ai/) and pull the required model:
   ```bash
   ollama pull llama3 # or your preferred model
   ```
3. **SpaCy Model**:
   ```bash
   python -m spacy download en_core_web_sm
   ```

## ⚙️ Setup & Installation

1. **Clone the repository**:
   ```bash
   git clone https://github.com/somasekhar21/Prompt_Guided_ABSA_Framework.git
   cd Prompt_Guided_ABSA_Framework
   ```

2. **Create a virtual environment** (using `uv` or `venv`):
   ```bash
   # Using uv (recommended)
   uv venv
   source .venv/bin/activate  # On Windows: .venv\Scripts\activate
   uv sync
   ```

3. **Environment Variables**:
   Create a `.env` file in the root directory:
   ```env
   OLLAMA_MODEL=llama3
   EMBEDDING_MODEL=all-MiniLM-L6-v2
   ```

## 🖥️ Usage

### Running the Graph Locally
You can test the pipeline with a single sentence using `graph.py`:
```bash
python graph.py
```

### Starting the FastAPI Server
To run the framework as a web service:
```bash
python main.py
```
The API will be available at `http://localhost:8000`. You can access the Swagger UI at `http://localhost:8000/docs`.

### Using the Notebook
Check out `demo.ipynb` for a step-by-step walkthrough of the internal logic and data structures.

## 📈 Use Cases

1. **E-commerce Reviews**: Automatically identify what customers like (e.g., "battery life") or dislike (e.g., "delivery speed") about a product.
2. **Customer Feedback**: Process survey results to find recurring themes and sentiments across different service categories.
3. **Brand Monitoring**: Track social media mentions to understand which specific aspects of a brand are trending positively or negatively.
4. **Market Research**: Analyze competitor reviews to find gaps in features or service quality.

## 🗺️ Project Structure

- `graph.py`: The core LangGraph workflow definition.
- `main.py`: FastAPI server implementation.
- `modelsetup.py`: Configuration for LLM and Embedding models.
- `vectorstoresetup.py`: ChromaDB initialization and management.
- `prompts.py`: Optimized LLM prompts for each stage of the pipeline.
- `utilityfunctions.py`: NLP utilities (preprocessing, POS tagging).

## 📄 License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.
