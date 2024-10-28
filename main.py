import torch
import torch.nn as nn
import torch.optim as optim


class DelayedMLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(DelayedMLP, self).__init__()
        # MLP layers
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

        # Delay mechanism
        self.delay_gate = nn.Linear(input_size, 1)  # Outputs a single delay value per input
        self.sigmoid = nn.Sigmoid()

        # Buffer to hold delayed inputs
        self.buffer = []

    def forward(self, x):
        batch_size, seq_len, feature_dim = x.shape
        outputs = []

        for t in range(seq_len):
            # Process each timestep
            current_input = x[:, t, :]

            # Compute delay for the current input
            delay_prob = self.sigmoid(self.delay_gate(current_input)).squeeze()

            # Create a mask for immediate processing
            immediate_mask = (delay_prob >= 0.5).float()
            delayed_mask = 1.0 - immediate_mask  # Delayed inputs go to the buffer

            # Immediate inputs are processed directly by the MLP
            if immediate_mask.sum() > 0:
                immediate_input = current_input * immediate_mask.unsqueeze(1)
                hidden = torch.relu(self.fc1(immediate_input))
                output = self.fc2(hidden)
                outputs.append(output)

            # Delayed inputs are stored in the buffer
            delayed_input = current_input * delayed_mask.unsqueeze(1)
            self.buffer.append(delayed_input)

            # Process delayed inputs in the buffer if they’re ready (every timestep)
            for i, buf in enumerate(self.buffer):
                # Apply MLP to buffered inputs
                hidden = torch.relu(self.fc1(buf))
                output = self.fc2(hidden)
                outputs.append(output)

            # Clear the buffer after processing
            self.buffer = []

        return torch.stack(outputs, dim=1)


# Hyperparameters
input_size = 1  # Simple scalar inputs (e.g., sequence of numbers)
hidden_size = 8
output_size = 1
seq_len = 5

# Dummy input data (batch_size=1, seq_len=5, feature_dim=1)
x = torch.tensor([[[1.0], [2.0], [3.0], [4.0], [5.0]]])

# Initialize and run the model
model = DelayedMLP(input_size, hidden_size, output_size)
output = model(x)
print("Model Output:\n", output)
