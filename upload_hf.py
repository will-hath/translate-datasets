import json
import os
from huggingface_hub import HfApi, create_repo, upload_file

def split_jsonl(content):
    """
    Splits a jsonl along "}]\n" boundaries
    """
    json_lines = []
    json_line = ""
    lines = content.split("}]\n")
    # Replace all "\n" with "\\n" to avoid splitting on newlines within the JSON
    lines = [line.replace("\n", "\\n") + "}]\n" for line in lines]
    return lines

def process_jsonl(input_file_path, output_file_path):
    # Create a dictionary to store data temporarily
    temp_data = {}

    # Read the content of the input JSONL file
    with open(input_file_path, 'r') as input_file:
        for line in input_file:
            data = json.loads(line)
            cluster_id = data[-1]["cluster_id"]
            
            if cluster_id not in temp_data:
                temp_data[cluster_id] = {"sentences": [], "labels": []}
            
            # Extract the sentence from the 'content' field
            sentence = data[1]["choices"][0]["message"]["content"]
            
            # Extract the label from the 'labels' field
            label = data[-1]["labels"]
            
            temp_data[cluster_id]["sentences"].append(sentence)
            temp_data[cluster_id]["labels"].append(label)

    # Write each cluster's data as a separate line in the output JSONL file
    with open(output_file_path, 'w') as output_file:
        for cluster_data in temp_data.values():
            output_file.write(json.dumps(cluster_data) + '\n')

def upload_to_huggingface(dataset_name, hf_username, hf_token, jsonl_file_path):

    # Create a new dataset repository on Hugging Face
    new_dataset_name = f"{hf_username}/{dataset_name}"

    create_repo(new_dataset_name, token=hf_token, repo_type="dataset")
    
    # Upload the JSONL file to Hugging Face
    upload_file(
        path_or_fileobj=jsonl_file_path,
        path_in_repo="test.jsonl",
        repo_id=new_dataset_name,
        repo_type="dataset",
        token=hf_token,
    )



if __name__ == "__main__":
    input_file_path = '/home/ec2-user/translate/responses/french-reddit-clustering/response.jsonl'
    dataset_name = 'french-reddit-clustering'
    hf_username = 'willhath'
    hf_token = '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'
    output_file_path = input_file_path.replace(".jsonl", "-processed.jsonl")
    process_jsonl(input_file_path, output_file_path)
    upload_to_huggingface(dataset_name, hf_username, hf_token, output_file_path)