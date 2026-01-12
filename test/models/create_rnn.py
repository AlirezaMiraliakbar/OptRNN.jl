import torch
import torch.nn as nn

class SimpleRNN(nn.Module):
    def __init__(self, input_size=3, hidden_size=8, num_layers=2, output_size=2):
        super(SimpleRNN, self).__init__()
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        out, _ = self.rnn(x)
        out = self.fc(out[:, -1, :])
        return out

# Create model with random weights
model = SimpleRNN(input_size=3, hidden_size=8, num_layers=2, output_size=2)

# Save the model state_dict
torch.save(model, 'rnn_model.pth')  # Full model
print("Model saved to rnn_model.pth")
print(f"Model architecture: RNN with input_size=3, hidden_size=8, num_layers=2, output_size=2")
