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
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size = config.lr_decay_epoch, gamma = config.lr_decay_gamma)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min')

    # Set training and validation dataloader
    train_loader = DataLoader(IrisLoader("dataset_trainable/train.csv"), batch_size = config.batch_size, shuffle = True)
    valid_loader = DataLoader(IrisLoader("dataset_trainable/valid.csv"), batch_size = config.batch_size, shuffle = True)

    # Set the model to train mode
    model.train()

    # Set the max validation loss to infinite
    max_validation_loss = np.inf

    # Do the training
    for epoch in range(config.max_epoch):
        # Train the model on every batch
        ## Reset the temp loss
        temp_loss = []
        for input_vectors, output_vectors in train_loader:
            # print(input_vectors)
            # Feed Forward
            pred = model.forward(input_vectors)

            # Calculate the loss function and backpropagate
            loss = loss_function(pred, output_vectors)
            loss.backward()
            
            # Sum the loss
            temp_loss.append(float(loss))
 
            # Adjust learning weights
            optimizer.step()

            # Reset optimizer gradient
            optimizer.zero_grad()

        # Calculate the training loss
        training_loss = np.mean(temp_loss)
        
        # Validate the model every specified epoch
        if (epoch + 1) % config.validation_epoch == 0:
            # Set the model to evaluation mode
            model.eval()

            ## Reset the temp loss
            temp_loss = []
            for input_vectors, output_vectors in valid_loader:
                # Do a Feed Forward
                pred = model.forward(input_vectors)
                loss = loss_function(pred, output_vectors)

                # Sum the loss
                temp_loss.append(float(loss))

            # Calculate the validation loss
            validation_loss = np.mean(temp_loss)

            # Step the plateau scheduler
            scheduler.step(validation_loss)

            print(f"Validation Loss: {validation_loss}")

            # Save the model that has the nearest valid loss with the train loss
            if validation_loss < max_validation_loss:
                ## Set the max_validation_loss to the new value
                max_validation_loss = validation_loss

                ## Check if the directory is available, create one if not avaiable
                if not os.path.exists(config.model_dir):
                    os.makedirs(config.model_dir)

                ## Save the best model
                torch.save(model.state_dict(), f'{config.model_dir}best_weight.pth')
                print("Best model has been saved")

            # Set the model back to training mode
            model.train()

        # Step the scheduler
        # scheduler.step()

        print(f"Epoch : {epoch}\t Loss : {training_loss}")

    # Save the model at the end of training
    ## Check if the directory is available, create one if not avaiable
    if not os.path.exists(config.model_dir):
        os.makedirs(config.model_dir)

    ## Save the model
    torch.save(model.state_dict(), f'{config.model_dir}final_weight.pth')
    print("Final Weight Has Been Saved")

# # create one test tensor from the testset
# X_test, y_test = default_collate(testset)
# model.eval()
# y_pred = model(X_test)
# acc = (y_pred.round() == y_test).float().mean()
# acc = float(acc)
# print("Model accuracy: %.2f%%" % (acc*100))