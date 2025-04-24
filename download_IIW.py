import os
import requests
from datasets import load_dataset
from tqdm import tqdm
import json

# Directory to save everything
save_dir = "" #input the save dir here!!
os.makedirs(save_dir, exist_ok=True)

# Directory for images
image_dir = os.path.join(save_dir, "images")
os.makedirs(image_dir, exist_ok=True)

# Directory for annotations/captions
data_dir = os.path.join(save_dir, "data")
os.makedirs(data_dir, exist_ok=True)

print(f"Loading IIW-400 dataset...")
dataset = load_dataset('google/imageinwords', token=None, name="IIW-400", trust_remote_code=True)
print(f"Dataset loaded: {dataset}")

# Get the test split (which is the only split in this dataset)
data = dataset['test']

# Create a simple JSON file for the dataset metadata
metadata = {
    "info": {
        "description": "ImageInWords (IIW-400) dataset",
        "url": "https://google.github.io/imageinwords/",
        "version": "1.0.0"
    },
    "images": [],
    "annotations": []
}

# List to store information about successfully downloaded images
downloaded_samples = []

# Download all images
print(f"Downloading {len(data)} images...")
for idx, sample in enumerate(tqdm(data)):
    image_key = sample['image/key']
    image_url = sample['image/url']
    image_path = os.path.join(image_dir, f"{image_key}.jpg")
    
    # Create annotation file with safe access to potentially None fields
    annotation = {
        "image_id": idx,
        "image_key": image_key,
        "image_file": f"{image_key}.jpg",
        "IIW": sample.get('IIW', ''),
        "IIW-P5B": sample.get('IIW-P5B', '')
    }
    
    # Safely add metrics if they exist
    metrics = {}
    if sample.get('iiw-human-sxs-gpt4v') is not None:
        metrics = {
            "comprehensiveness": sample['iiw-human-sxs-gpt4v'].get('metrics/Comprehensiveness', 'N/A'),
            "specificity": sample['iiw-human-sxs-gpt4v'].get('metrics/Specificity', 'N/A'),
            "hallucination": sample['iiw-human-sxs-gpt4v'].get('metrics/Hallucination', 'N/A'),
            "tldr": sample['iiw-human-sxs-gpt4v'].get('metrics/First few line(s) as tldr', 'N/A'),
            "human_like": sample['iiw-human-sxs-gpt4v'].get('metrics/Human Like', 'N/A')
        }
    
    annotation["metrics"] = metrics
    
    # Download image if it doesn't exist
    if not os.path.exists(image_path):
        try:
            response = requests.get(image_url, timeout=15)
            if response.status_code == 200:
                with open(image_path, 'wb') as f:
                    f.write(response.content)
                
                # Add to metadata only if download successful
                metadata["images"].append({
                    "id": idx,
                    "file_name": f"{image_key}.jpg",
                    "image_key": image_key,
                    "url": image_url
                })
                metadata["annotations"].append(annotation)
                downloaded_samples.append(annotation)
            else:
                print(f"\nFailed to download {image_key}, status code: {response.status_code}")
        except Exception as e:
            print(f"\nError downloading {image_key}: {e}")
    else:
        # Image already exists
        metadata["images"].append({
            "id": idx,
            "file_name": f"{image_key}.jpg",
            "image_key": image_key,
            "url": image_url
        })
        metadata["annotations"].append(annotation)
        downloaded_samples.append(annotation)

# Save the JSON metadata
metadata_file = os.path.join(save_dir, "metadata.json")
with open(metadata_file, 'w') as f:
    json.dump(metadata, f, indent=2)

# Save individual JSON files for each image-caption pair
for sample in downloaded_samples:
    json_file = os.path.join(data_dir, f"{sample['image_key']}.json")
    with open(json_file, 'w') as f:
        json.dump(sample, f, indent=2)

# Create a simple text file with all the descriptions
txt_file = os.path.join(save_dir, "all_descriptions.txt")
with open(txt_file, 'w') as f:
    for sample in downloaded_samples:
        f.write(f"Image: {sample['image_key']}.jpg\n")
        f.write(f"Description (IIW):\n{sample.get('IIW', 'N/A')}\n\n")
        f.write(f"Description (IIW-P5B):\n{sample.get('IIW-P5B', 'N/A')}\n\n")
        f.write("-" * 80 + "\n\n")

print(f"\nDownloaded {len(downloaded_samples)} images out of {len(data)} total")
print(f"Dataset saved to: {save_dir}")
print(f"Images saved to: {image_dir}")
print(f"Metadata saved to: {metadata_file}")
print(f"Individual annotations saved to: {data_dir}")

print("\nYou can now access the dataset with both images and descriptions at the specified location.") 
