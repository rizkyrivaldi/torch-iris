"""
Dataloader for torch batch training, for faster training
"""

# Import generic library
import pandas as pd

# Import torch dataset library
import torch
from torch.utils.data import Dataset

# Import local functions
from utils.dataset import DatasetTranslator

# Set device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class IrisLoader(Dataset):
    # Initialization
    def __init__(self, dataset_path):
        # Initialize the Dataset object from torch
        super().__init__()

        # Get the dataset to pandas
        self.train_df = pd.read_csv(dataset_path, header = 0)

        # Get the dataset translator
        self.translator = DatasetTranslator(["Iris-setosa", "Iris-versicolor", "Iris-virginica"])

    # Length checker
    def __len__(self):
        return len(self.train_df)

    # Item getter
    def __getitem__(self, idx):
        # Get the data row
        data_row = self.train_df.iloc[idx].to_numpy()

        # Tensorize the input vector
        raw_input_vector = data_row[:-2]
        normalized_input_vector = list(raw_input_vector / 10.0)
        input_vector = torch.Tensor(normalized_input_vector).to(device)

        # Tensorize the output vector
        encoded_output = self.translator.encode(data_row[-1])
        output_vector = torch.Tensor(encoded_output).to(device)

        return input_vector, output_vector