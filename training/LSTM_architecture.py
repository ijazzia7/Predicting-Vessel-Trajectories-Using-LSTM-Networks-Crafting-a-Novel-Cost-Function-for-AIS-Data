import torch
import torch.nn as nn
import torch.optim as optim

class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1):
        super(LSTMModel, self).__init__()
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc1 = nn.Linear(hidden_size, 16)
        self.fc2 = nn.Linear(16, output_size)
        
        self.act = nn.ReLU()
        
        
    def forward(self, x):
            h0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)
            c0 = torch.zeros(self.num_layers, x.size(1), self.hidden_size).to(x.device)
            out, _ = self.lstm(x[0], (h0, c0))
            out = self.fc2(self.act(self.fc1(out[:, -1, :])))
            return out
        
        


model = LSTMModel(15, 32, 2, 1).to(device)
