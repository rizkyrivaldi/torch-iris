"""
Split the dataset into train and validation, this will be useful to get more accurate evaluation
"""

# Import generic library
import numpy as np
import pandas as pd
import shutil
import os

class SplitTrainVal:
    def __init__(self, input_path, output_path, input_header = None, split_ratio = 0.8):
        self.input_path = input_path
        self.output_path = output_path
        self.split_ratio = split_ratio

        # Get the raw dataset
        self.raw_df = pd.read_csv(self.input_path, header = input_header)
    
    def splitTrainVal(self):
        # Shuffle the data for a better randomization
        df = self.raw_df.sample(frac=1).reset_index(drop=True)

        # Split the dataframe to train and validation
        df_train = df.iloc[:int(len(df)*self.split_ratio)]
        df_valid = df.iloc[int(len(df)*self.split_ratio):]
        
        # For debug purpose, check max value for each column, for normalization
        # print(df.max())

        # Delete the output path directory if exist
        if os.path.exists(self.output_path):
            shutil.rmtree(self.output_path)
        
        # Create a new directory
        os.makedirs(self.output_path)

        # Save the train and valid dataset in output directory
        df_train.to_csv(self.output_path + "train.csv", index = False)
        df_valid.to_csv(self.output_path + "valid.csv", index = False)

if __name__ == "__main__":
    splitter = SplitTrainVal(
        input_path = "dataset/Iris.csv",
        output_path = "dataset_trainable/",
        input_header = 0,
        split_ratio = 0.8
    )

    splitter.splitTrainVal()
