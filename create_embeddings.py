import json

def create_jsonl_for_embedding(texts, output_file="embeddings.jsonl"):
    """
    Create a JSONL file formatted for OpenAI's embedding API.
    Each line will be a JSON object with 'text' field.
    
    Args:
        texts (list): List of strings to be embedded
        output_file (str): Path to output JSONL file
    """
    with open(output_file, 'w', encoding='utf-8') as f:
        for text in texts:
            json_line = {"text": text.strip()}
            f.write(json.dumps(json_line, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    # Example usage
    sample_texts = [
        "This is the first text to embed.",
        "Here's another text for embedding.",
        "And a third one for good measure."
    ]
    
    create_jsonl_for_embedding(sample_texts)
