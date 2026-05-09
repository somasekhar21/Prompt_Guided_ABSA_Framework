from langchain_community.vectorstores import Chroma
from modelsetup import load_embedding_model
import chromadb


def load_vectorstore_with_model():
    embedding_model = load_embedding_model()
    vectorstore=Chroma(embedding_function=embedding_model,collection_name="unknown_aspects_collection",persist_directory="./vectorstore")
    return vectorstore

def load_vectorstore():
    return Chroma(collection_name="unknown_aspects_collection",persist_directory="./vectorstore")

def load_documents():
    vectorstore = chromadb.PersistentClient(path="./vectorstore")
    collection = vectorstore.get_collection("unknown_aspects_collection")
    documents=collection.get(
        include=["embeddings","metadatas"]
    )
    return documents

if __name__ == "__main__":
    documents=load_documents()
    print(documents)