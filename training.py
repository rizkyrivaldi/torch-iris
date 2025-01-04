# External Imports
import torch
from torch.utils.data import DataLoader
import yaml
import numpy as np
import os

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

    # Set the loss function
    loss_function = torch.nn.MSELoss()

    # Set the optimizer, Stochastic Gradient Descent
    optimizer = torch.optim.SGD(model.parameters(), lr = config.learning_rate, momentum = config.learning_momentum)

    # Set LR Scheduler
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = config.lr_decay_epoch, gamma = config.lr_decay_gamma)
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'max')

    # Set training and validation dataloader
    train_loader = DataLoader(IrisLoader("dataset_trainable/train.csv"), batch_size = config.batch_size, shuffle = True)
    valid_loader = DataLoader(IrisLoader("dataset_trainable/valid.csv"), batch_size = config.batch_size, shuffle = True)

    # Set the model to train mode
    model.train()

    # Do the training
    for epoch in range(config.max_epoch):
        # Train the model on every batch
        ## Reset the temp loss
        temp_loss = []
        for input_vectors, output_vectors in train_loader:
            # print(input_vectors)
            # Feed Forward
            pred = model.forward(input_vectors)

            # Reset optimizer gradient
            optimizer.zero_grad()

            # Calculate the loss function and backpropagate
            loss = loss_function(pred, output_vectors)
            loss.backward()
            
            # Sum the loss
            temp_loss.append(float(loss))
 
            # Adjust learning weights
            optimizer.step()
        
        # Step the scheduler
        scheduler.step()

        # Calculate the epoch loss
        epoch_loss = np.mean(temp_loss)

        # Step the scheduler
        # scheduler.step(epoch_loss)

        print(f"Epoch : {epoch}\t Loss : {epoch_loss}")

    # Save the model at the end of training
    ## Check if the directory is available, create one if not avaiable
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)

    ## Save the model
    torch.save(model.state_dict(), f'{config.model_dir}saved_model.pth')
    print("Model Has Been Saved")

# # create one test tensor from the testset
# X_test, y_test = default_collate(testset)
# model.eval()
# y_pred = model(X_test)
# acc = (y_pred.round() == y_test).float().mean()
# acc = float(acc)
# print("Model accuracy: %.2f%%" % (acc*100))