# External Imports
import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np
import os

# Local Import
from utils.utils import AttrDict
from utils.dataloader import IrisLoader
from utils.dataset import DatasetTranslator
from model import Model

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if __name__ == "__main__":
    # Get the config data
    with open('config.yaml') as f:
        # Load data to dictionary
        config_dict = yaml.safe_load(f)

        # Convert the dictionary to object
        config = AttrDict(config_dict)

    # Get the dataloader
    test_dataloader = IrisLoader("dataset/Iris.csv")

    # Get the data decoder
    translator = DatasetTranslator(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])

    # Load the model
    model = Model(config).to(device)
    model.load_state_dict(torch.load(config.model_dir + "best_weight.pth"))

    # Set the model to eval mode
    model.eval()

    # True and False Counter
    true_counter = 0
    false_counter = 0

    # Start the testing for every data in the dataset
    for input_vectors, output_vectors in test_dataloader:
        # Get the prediction in list
        pred = model(input_vectors)

        # Decode the prediction and actual output
        pred_text, confidence = translator.decode(pred.tolist())
        actual_text, confidence = translator.decode(output_vectors.tolist())
        
        # Get the true and false counter counting
        if pred_text == actual_text:
            true_counter += 1
        else:
            false_counter += 1

    print(f"True : {true_counter}\t False : {false_counter}")
    print(f"Accuracy : {true_counter/(true_counter + false_counter)*100:.2f}%")
