import open_clip
import torch
from torchvision import datasets
import numpy as np
from tqdm import tqdm


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

    # Process a small subset of the dataset, for example, the first 1000 images
    for i, (image, label) in enumerate(tqdm(trainset)):
        if limit != -1:
            if i >= limit:  # Limit the number of images processed
                break
        preprocessed_image = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            image_embedding = model.encode_image(preprocessed_image)
            image_embedding /= image_embedding.norm(dim=-1, keepdim=True)

        embeddings.append(image_embedding.squeeze().numpy())  # Remove batch dimension and convert to numpy
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
