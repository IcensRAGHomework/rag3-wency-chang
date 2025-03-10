import datetime
import chromadb
import traceback
import pandas as pd

from chromadb.utils import embedding_functions

from model_configurations import get_model_configuration

gpt_emb_version = 'text-embedding-ada-002'
gpt_emb_config = get_model_configuration(gpt_emb_version)

dbpath = "./"
column_words = "HostWords"

def generate_hw01():
    return get_travel_collection()
    
def generate_hw02(question, city, store_type, start_date, end_date):
    collection = get_travel_collection()

    results = collection.query(
        query_texts=question,
        n_results=10,
        where={
        "$and": [
            {"city": {"$in": city}},
            {"type": {"$in": store_type}},
            {"$and": [
                {"date": {"$gte": start_date.timestamp()}}, 
                {"date": {"$lte": end_date.timestamp()}}
                ]
            }
            ]
        }
    )
    
    filtered_results = [
        results["metadatas"][0][i]["name"]
        for i, distance in enumerate(results["distances"][0])
        if distance <= 0.20 
    ]
    return filtered_results
    
def generate_hw03(question, store_name, new_store_name, city, store_type):
    collection = get_travel_collection()
    docs = collection.get(where={"name": store_name})

    if docs["ids"]:
        doc_id = docs["ids"][0]
        metadata = docs["metadatas"][0]
        metadata["new_store_name"] = new_store_name
        collection.update(ids=[doc_id], metadatas=[metadata])
    
    results = collection.query(
        query_texts=question,
        n_results=10,
        where={
        "$and": [
            {"city": {"$in": city}},
            {"type": {"$in": store_type}}
        ]}
    )

    filtered_results = [
        results["metadatas"][0][i].get("new_store_name", results["metadatas"][0][i]["name"])
        for i, distance in enumerate(results["distances"][0])
        if distance <= 0.20 
    ]

    return filtered_results
    
def get_travel_collection():
    chroma_client = chromadb.PersistentClient(path=dbpath)
    openai_ef = embedding_functions.OpenAIEmbeddingFunction(
        api_key = gpt_emb_config['api_key'],
        api_base = gpt_emb_config['api_base'],
        api_type = gpt_emb_config['openai_type'],
        api_version = gpt_emb_config['api_version'],
        deployment_id = gpt_emb_config['deployment_name']
    )

    collection = chroma_client.get_or_create_collection(
        name="TRAVEL",
        metadata={"hnsw:space": "cosine"},
        embedding_function=openai_ef
    )

    if collection.count() == 0:
        load_data_into_chromadb(collection)

    return collection

def load_data_into_chromadb(collection):
    df = pd.read_csv("COA_OpenData.csv")

    documents = df[column_words].tolist()
    metadata_list = [
        {
            "file_name": "COA_OpenData.csv",
            "name": row["Name"],
            "type": row["Type"],
            "address": row["Address"],
            "tel": row["Tel"],
            "city": row["City"],
            "town": row["Town"],
            "date": int(pd.to_datetime(row["CreateDate"]).timestamp())
        }
        for _, row in df.iterrows()
    ]
    
    collection.add(
        ids=[str(i) for i in range(len(documents))],
        documents=documents,
        metadatas=metadata_list
    )
    print(f"已成功加入 {len(documents)} 筆資料")

