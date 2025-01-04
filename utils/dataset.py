"""
This part will encode/decode the neural network output to human readable
"""
# Generic imports
import numpy as np

class DatasetTranslator:
    def __init__(self, output_features: list):
        """
        Tell the object how many features the output layer has
        so that the class is able to encode/decode the output layer
        correctly.
        Note: This is like a decryption key or whatever you think it is in comparison
                Keep the key the same everytime you want to use the neural network
        i.e
        output_features = ["Flower A", "Flower B", "Flower C"]
        """
        # Get the output_features
        self.output_features = output_features

        # Check if the output features are unique
        if len(self.output_features) != len(set(self.output_features)):
            raise ValueError("The output features are not unique")

        # Create an enumerate dict based on the features
        self.feature_key = dict()
        for i in range(len(self.output_features)):
            feature = self.output_features[i]
            self.feature_key.update({i:feature})

        # Create the inverse enumerate dict
        self.feature_key_inv = {value : key for key, value in self.feature_key.items()}

        # Get the feature key length
        self.feature_key_len = len(self.feature_key)
        
    def encode(self, input_value):
        """
        Do a one hot encoding based on the dataset provided
        i.e:
        input_value = "Flower A"
        it will return [1, 0, 0], based on feature key enumerate

        another example:
        input_value = "Flower B"
        it will return [0, 1, 0], based on feature key enumerate
        """
        # Check if the input is not valid
        if input_value not in self.feature_key_inv.keys():
            raise ValueError(f"The encode input is invalid, value not found in the feature {list(self.feature_key.values())}")

        encoded_list = [0.0] * self.feature_key_len
        encoded_list[self.feature_key_inv[input_value]] = 1.0

        return encoded_list

    def decode(self, output_vector):
        """
        Decode the one hot encoding based on the dataset provided
        i.e:
        output_vector = [0.2, 0.8, 0.1]
        it will return "Flower B" and 0.8, based on the max value

        returns Decoded feature, confidence
        """
        # Check if the object is not iterable
        if not hasattr(output_vector, '__iter__'):
            raise TypeError(f"The decode input is invalid, object is not iterable: {output_vector}")

        # Check if the object length is the same as feature length
        if len(output_vector) != self.feature_key_len:
            raise ValueError(f"The decode input is invalid, object length don't have the same number as the keys, expected {self.feature_key_len}, got {len(output_vector)}")

        # Get the max index
        max_index = np.argmax(output_vector)

        # Return the decoded feature and the confidence rate
        return self.feature_key[max_index], output_vector[max_index]

if __name__ == "__main__":
    # test = DatasetTranslator(["Flower A", "Flower B", "Flower C"])
    # print(test.feature_key)
    # print(test.feature_key_inv)
    # print(test.encode("Flower C"))
    # print(test.decode([0.1, 0.8, 0.4]))
    pass
    
   