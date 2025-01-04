# External Imports
import torch
import yaml

# Local Import
from utils.utils import AttrDict
from utils.dataloader import IrisLoader
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
    
    # Get model
    model = Model(config).to(device)

    # Set to train mode
    model.train()

    # Test tensor input (TODO: Remove this)
    input_tensor = torch.Tensor([0.1, 0.2, 0.3]).to(device)
    print(model.forward(input_tensor))
    loader = IrisLoader("dataset_trainable/train.csv")
    print(loader.__getitem__(10))
