import torch
import torch.nn as nn

class Create(torch.nn.Module):
    def __init__(self, input_shape, outputs_count, hidden_count = 64):
        super(Create, self).__init__()

        self.device = "cuda"

        features_count = input_shape[0]
        self.rnn_layer = nn.GRU(input_size = features_count, hidden_size = hidden_count, batch_first = True)
        self.output_layer = nn.Linear(hidden_count,outputs_count[0])

        torch.nn.init.xavier_uniform_(self.output_layer.weight)

        self.rnn_layer.to(self.device)
        self.output_layer.to(self.device)

        print("model")
        print(self.rnn_layer)
        print(self.output_layer)
        print("\n\n")


    def forward(self, x):
        xt = x.transpose(1,2)
        _,(hidden_state) = self.rnn_layer(xt)
        return self.output_layer(hidden_state[0])


    def save(self, path):
        torch.save(self.rnn_layer.state_dict(), path + "./model_rnn.pt")
        torch.save(self.output_layer.state_dict(), path + "./model_output.pt")

    def load(self, path):
        self.rnn_layer.load_state_dict(torch.load(path + "./model_rnn.pt", map_location = self.device))
        self.rnn_layer.eval()
        self.output_layer.load_state_dict(torch.load(path + "./model_output.pt", map_location = self.device))
        self.output_layer.eval()


if __name__ == "__main__":
    input_shape = (3, 1024)
    batch_size  = 16

    model = Create(input_shape, 3)

    x = torch.randn((batch_size, ) + input_shape)

    y = model.forward(x)

    print("y shape = ", y.shape)
    print(y)
