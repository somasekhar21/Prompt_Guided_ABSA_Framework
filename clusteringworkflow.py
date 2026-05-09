
import os
import json
import random
import pandas as pd
import numpy as np
import hdbscan
from vectorstoresetup import load_documents,load_vectorstore
from modelsetup import load_llm_model
from langchain_core.prompts import ChatPromptTemplate

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)


def load_data_vs():
    x = load_documents()
    em = x.get('embeddings', [])
    mt = x.get('metadatas', [])

    return em, mt

def format_data(mt):
    datastr=[]
    for i in mt:
        datastr.append(f"Aspect: {i['aspect_keywords']} | Sentence: {i['sentence']}")
    return datastr

def normalize_embeddings(embeddings):
    mean_vec_contextual = np.mean(embeddings, axis=0)
    embeddings_centered = embeddings - mean_vec_contextual
    # Re-normalize embeddings after mean-centering, avoiding division by zero
    norms = np.linalg.norm(embeddings_centered, axis=1, keepdims=True)
    norms[norms == 0] = 1
    embeddings_normalized = embeddings_centered / norms
    return embeddings_normalized

def display_cluster_result(datastr,labels_contextual):
# Display clustering results
    contextual_clusters = {}
    for data_str, label in zip(datastr, labels_contextual):
        contextual_clusters.setdefault(label, []).append(data_str)

    print("\n--- Clustering Results with Contextual Embeddings ---")
    for cid, items in contextual_clusters.items():
        print(f"\n🔹 Cluster {cid}:")
        for item_str in items:
            print(f" - {item_str}")

def clustering():

    em,md=load_data_vs()
    
    if len(em) < 2:
        print("Not enough data to cluster. At least 2 unknown aspects are required.")
        return

    embedding_normalized=normalize_embeddings(em)

# HDBSCAN clustering
    clusterer_contextual = hdbscan.HDBSCAN(
        min_cluster_size=2,  # Minimum number of samples in a cluster
        min_samples=1,       # The number of samples in a neighborhood for a point to be considered as a core point
        metric='euclidean'   # Distance metric
    )

    labels_contextual = clusterer_contextual.fit_predict(embedding_normalized)
    datastr=format_data(md)
    display_cluster_result(datastr,labels_contextual)
    
    # Save clustering results to CSV
    cluster_results = []
    for metadata, label in zip(md, labels_contextual):
        # Parse sentiments from list of JSON strings to list of dicts
        sentiments_raw = metadata.get("sentiments", [])
        if isinstance(sentiments_raw, list):
            sentiments_parsed = [json.loads(s) if isinstance(s, str) else s for s in sentiments_raw]
        else:
            sentiments_parsed = json.loads(sentiments_raw) if isinstance(sentiments_raw, str) else sentiments_raw
            
        cluster_results.append({
            "cluster no": label,
            "id": metadata.get("id", ""),
            "sentence": metadata.get("sentence", ""),
            "sentiments": sentiments_parsed
        })
    df_clusters = pd.DataFrame(cluster_results)
    output_path = os.path.join(OUTPUT_DIR, "clustering_results.csv")
    df_clusters.to_csv(output_path, index=False)
    print(f"\nClustering results saved to {output_path}")



def label_clusters():
    """Reads clustering results and uses an LLM to name each cluster, handling implicit targets."""
    try:
        path=os.path.join(OUTPUT_DIR, "clustering_results.csv")
        if not os.path.exists(path):
            print("Error: clustering_results.csv not found. Run clustering() first.")
            return
        df = pd.read_csv(path)
    except FileNotFoundError:
        print("Error: clustering_results.csv not found. Run clustering() first.")
        return

    llm = load_llm_model()
    cluster_groups = []
    for cid, group in df.groupby("cluster no"):
        if cid == -1: continue
        context_items = [f"Sentence: {row['sentence']}\nSentiments: {row['sentiments']}" for _, row in group.iterrows()]
        cluster_groups.append((cid, context_items))
    
    cluster_labels = {-1: "Uncategorized / Noise"}
    print("\n--- Generating Semantic Labels (Handling Implicit Targets) ---")
    
    for cluster_id, items in cluster_groups:
        prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a highly precise semantic classifier for product aspects. 
            You will be given sentences and extracted sentiments.
            
            IMPORTANT: Some 'target' keywords may be null or 'implicit'. In these cases, 
            infer the specific topic from the context of the sentence and the sentiment polarity.
            
            Your goal: Identify the single common specific product feature or concept.
            Guidelines:
            - Focus on identifying the 'unknown' domain (e.g., Mobile Devices, Apps, etc.)
            - Provide a concise category name (2-4 words).
            - Example output: 'Display & Visual Feedback' or 'Network Signal Stability'."""),
            ("user", "Analyze this cluster data:\n\n{data}\n\nCommon Category Name:")
        ])
        
        chain = prompt | llm
        
        # Select up to 10 random samples to represent the cluster
        if len(items) > 10:
            sampled_items = random.sample(items, 10)
        else:
            sampled_items = items
            
        data_context = "\n\n".join(sampled_items)
        
        try:
            response = chain.invoke({"data": data_context})
            label = response.content.strip().replace('"', '').split('\n')[0]
            cluster_labels[cluster_id] = label
            print(f"Cluster {cluster_id}: {label}")
        except Exception as e:
            print(f"Error labeling cluster {cluster_id}: {e}")
            cluster_labels[cluster_id] = f"Cluster {cluster_id}"

    df["cluster_label"] = df["cluster no"].map(cluster_labels)
    output_path = os.path.join(OUTPUT_DIR, "labeled_clustering_results.csv")
    df.to_csv(output_path, index=False)
    print(f"\nLabeled results saved to {output_path}")
    
    return cluster_labels

if __name__ == "__main__":
    clustering()
    labels=label_clusters()
    print(labels)