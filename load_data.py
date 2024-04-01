import open_clip
import torch
from torchvision import datasets
import numpy as np
from tqdm import tqdm


def load_data(limit=-1):
    """
    Load a subset of the MNIST dataset and encode images using the OpenCLIP model.
    Returns the features and labels of the dataset.
    """
    # Load the MNIST dataset
    trainset = datasets.MNIST('data', download=True, train=True)

    # Load OpenCLIP model and preprocessing tools
    model, _, preprocess = open_clip.create_model_and_transforms('convnext_base_w',
                                                                 pretrained='laion2b_s13b_b82k_augreg')

    # Initialize lists to store features and labels
    features = []
    labels = []

    # Process a small subset of the dataset, for example, the first 1000 images
    for i, (image, label) in enumerate(tqdm(trainset)):
        if limit != -1:
            if i >= limit:  # Limit the number of images processed
                break
        preprocessed_image = preprocess(image).unsqueeze(0)

        with torch.no_grad():
            image_feature = model.encode_image(preprocessed_image)
            image_feature /= image_feature.norm(dim=-1, keepdim=True)

        features.append(image_feature.squeeze().numpy())  # Remove batch dimension and convert to numpy
        labels.append(label)

    # Convert to numpy arrays
    features = np.array(features)
    return features, labels


def main():
    features, labels = load_data(10)
    print(features[0])
    print(labels[0])


if __name__ == '__main__':
    main()
