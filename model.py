import torch

class Model(torch.nn.Module):

    def __init__(self, config):
        # Initialize super
        super(Model, self).__init__()

        # The linear layer
        self.linear_layer = torch.nn.Sequential(
            torch.nn.Linear(config.input_neuron, config.hidden_neuron),
            torch.nn.Tanh(),
            torch.nn.Linear(config.hidden_neuron, config.hidden_neuron),
            torch.nn.Tanh(),
            torch.nn.Linear(config.hidden_neuron, config.hidden_neuron),
            torch.nn.Tanh(),
            torch.nn.Linear(config.hidden_neuron, config.hidden_neuron),
            torch.nn.Tanh(),
            torch.nn.Linear(config.hidden_neuron, config.output_neuron),
            torch.nn.Tanh()
        )

    def forward(self, x):
        x = self.linear_layer(x)
        return x