import torch

class Model(torch.nn.Module):

    def __init__(self, config):
        # Initialize super
        super(Model, self).__init__()

        self.activation_function = torch.nn.Tanh
        # self.activation_function = torch.nn.ReLU
        # self.activation_function = torch.nn.Sigmoid

        # The linear layer
        self.linear_layer = torch.nn.Sequential(
            torch.nn.Linear(config.input_neuron, config.hidden_neuron),
            self.activation_function(),
            torch.nn.Linear(config.hidden_neuron, config.hidden_neuron),
            self.activation_function(),
            torch.nn.Linear(config.hidden_neuron, config.hidden_neuron),
            self.activation_function(),
            torch.nn.Linear(config.hidden_neuron, config.hidden_neuron),
            self.activation_function(),
            torch.nn.Linear(config.hidden_neuron, config.output_neuron),
            self.activation_function()
        )

    def forward(self, x):
        x = self.linear_layer(x)
        return x