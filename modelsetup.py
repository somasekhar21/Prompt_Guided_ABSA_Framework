import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from langchain_ollama import ChatOllama
from langchain_community.embeddings import HuggingFaceEmbeddings


load_dotenv() # This will load your HF_TOKEN from the .env file for faster download of model from huggingface

#to change model edit here
model_name = "BAAI/bge-large-en-v1.5"
local_model_path = "./models/BGE_large_EN_v1.5"  


def download_model():
    try:
        model = SentenceTransformer(model_name)
        model.save(local_model_path)
    except Exception as e:
        return f"exception: {e}"


def load_embedding_model():
    try:
        if os.path.exists(local_model_path):
            print(f"Loading model from local path: {local_model_path}")
            model = HuggingFaceEmbeddings(model_name=local_model_path,encode_kwargs={"normalize_embeddings":True})
            return model
        
        else:
            print(f"Downloading model '{model_name}' from HuggingFace...")
            download_model()
            return load_embedding_model()

    except Exception as e:
        return f"exception: {e}"


def load_llm_model():
    try:
        model = ChatOllama(
            model="gemma4:31b-cloud",
            base_url="https://ollama.com",
            num_ctx=8192,
            temperature=0,
            top_k=10,
            top_p=0.9
        )
        return model

    except Exception as e:
        return f"exception: {e}"

if __name__ == "__main__":
    embeddings=load_embedding_model()
    print(embeddings.embed_query("i love the food"),"\n",embeddings)
    print("\n")
    llm=load_llm_model()
    print(llm.invoke("hello").content)
    
