#!/usr/bin/env python
import argparse
import json
import openai

def update_empty_embedding(embedding_json, openai_api_key, model="text-embedding-3-large"):
    openai.api_key = openai_api_key
    with open(embedding_json, "r") as f:
        embedding_map = json.load(f)
    if "" in embedding_map:
        print("Empty string embedding already exists.")
    else:
        client = openai.OpenAI(api_key=openai_api_key)
        try:
            response = client.embeddings.create(input="", model=model)
            embedding = response.data[0].embedding
            embedding_map[""] = embedding
            with open(embedding_json, "w") as f:
                json.dump(embedding_map, f, indent=2)
            print("Added empty string embedding to the embedding map.")
        except Exception as e:
            print(f"Failed to create embedding for empty string: {e}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--embedding_json", type=str, required=True, help="Path to embedding JSON file")
    parser.add_argument("--openai_api_key", type=str, required=True, help="OpenAI API key")
    parser.add_argument("--model", type=str, default="text-embedding-3-large", help="OpenAI embedding model to use")
    args = parser.parse_args()
    update_empty_embedding(args.embedding_json, args.openai_api_key, args.model)
