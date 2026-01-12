import torch
import torch.nn as nn

class SimpleLSTM(nn.Module):
    def __init__(self, input_size=3, hidden_size=8, num_layers=2, output_size=2):
        super(SimpleLSTM, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out

# Create model with random weights
model = SimpleLSTM(input_size=3, hidden_size=8, num_layers=2, output_size=2)

# Save the model state_dict
torch.save(model.state_dict(), 'lstm_model.pth')
print("Model saved to lstm_model.pth")
print(f"Model architecture: LSTM with input_size=3, hidden_size=8, num_layers=2, output_size=2")
