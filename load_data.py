import json
import os

import open_clip
import torch
from torchvision import datasets
import numpy as np
from tqdm import tqdm


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def dump_json(obj, path, indent=None):
    with open(path, "w") as f:
        json.dump(obj, f, indent=indent)


def load_data(limit=-1):
    """
    Load a subset of the MNIST dataset and encode images using the OpenCLIP model.
    Returns the embeddings and labels of the dataset.
    """
    # Load the MNIST dataset
    trainset = datasets.MNIST('data', download=True, train=True)

    # Load OpenCLIP model and preprocessing tools
    model, _, preprocess = open_clip.create_model_and_transforms('convnext_base_w',
                                                                 pretrained='laion2b_s13b_b82k_augreg')

    # Initialize lists to store embeddings and labels
    embeddings = []
    labels = []

    cache_dir = "cache/train"
    os.makedirs(cache_dir, exist_ok=True)

    # Process a small subset of the dataset, for example, the first 1000 images
    for i, (image, label) in enumerate(tqdm(trainset)):
        if limit != -1:
            if i >= limit:  # Limit the number of images processed
                break

        path = f"{cache_dir}/{i}.json"
        try:
            el = load_json(path)
            image_embedding = np.array(el["e"])
            label = el["l"]
            # print(f"cache {i}")

        except:
            preprocessed_image = preprocess(image).unsqueeze(0)
            with torch.no_grad():
                image_embedding = model.encode_image(preprocessed_image)
                image_embedding /= image_embedding.norm(dim=-1, keepdim=True)
            image_embedding = image_embedding.squeeze().numpy()

            # print(type(image_embedding))
            # print(type(label))

            dump_json(dict(
                e=image_embedding.tolist(),
                l=label,
            ), path)

            # quit()

        embeddings.append(image_embedding)  # Remove batch dimension and convert to numpy
        labels.append(label)

    # Convert to numpy arrays
    embeddings = np.array(embeddings)
    return embeddings, labels


def main():
    embeddings, labels = load_data(10)
    print(embeddings[0])
    print(labels[0])


if __name__ == '__main__':
    main()
