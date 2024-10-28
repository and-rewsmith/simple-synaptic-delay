import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb

# Constants
NUM_SEQUENCES = 1
SEQUENCE_LENGTH = 200
NUM_MODES = 1
FREQ_RANGE = (1.5, 10.5)
AMP_RANGE = (0.5, 1.5)
PHASE_RANGE = (0, 2 * np.pi)
NUM_BINS = 25
HIDDEN_SIZE = 64
OUTPUT_SIZE = NUM_BINS
BATCH_SIZE = 32
LEARNING_RATE = 1e-3
NUM_EPOCHS = 50

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def initialize_wandb():
    wandb.init(project="delayed-sequence-prediction", config={
        "num_sequences": NUM_SEQUENCES,
        "sequence_length": SEQUENCE_LENGTH,
        "num_modes": NUM_MODES,
        "freq_range": FREQ_RANGE,
        "amp_range": AMP_RANGE,
        "phase_range": PHASE_RANGE,
        "num_bins": NUM_BINS,
        "hidden_size": HIDDEN_SIZE,
        "output_size": OUTPUT_SIZE,
        "batch_size": BATCH_SIZE,
        "learning_rate": LEARNING_RATE,
        "num_epochs": NUM_EPOCHS,
    })


def generate_waveforms(num_sequences: int, sequence_length: int, num_modes: int,
                       freq_range: tuple, amp_range: tuple, phase_range: tuple) -> np.ndarray:
    waveforms = np.zeros((num_sequences, sequence_length))
    t = np.linspace(0, 2 * np.pi, sequence_length, endpoint=False)
    for i in range(num_sequences):
        for _ in range(num_modes):
            frequency = np.random.uniform(*freq_range)
            amplitude = np.random.uniform(*amp_range)
            phase = np.random.uniform(*phase_range)
            waveforms[i] += amplitude * np.sin(frequency * t + phase)
    return waveforms


def discretize_waveforms(waveforms: np.ndarray, num_bins: int) -> np.ndarray:
    min_val, max_val = waveforms.min(), waveforms.max()
    scaled_waveforms = (waveforms - min_val) / (max_val - min_val) * (num_bins - 1)
    discretized_waveforms = np.clip(np.round(scaled_waveforms), 0, num_bins - 1).astype(int)
    one_hot_waveforms = np.eye(num_bins)[discretized_waveforms]
    return one_hot_waveforms


class DelayedMLP(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, output_size: int):
        super(DelayedMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)
        self.delay_gate = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()
        self.buffer = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, input_size = x.size()
        self.buffer = torch.zeros((batch_size, input_size), device=x.device)
        outputs = []

        for t in range(seq_len):
            current_input = x[:, t, :]
            decay_weights = self.sigmoid(self.delay_gate(current_input))

            immediate_contribution = current_input * decay_weights
            delayed_contribution = (1 - decay_weights) * current_input
            self.buffer = self.buffer * decay_weights + delayed_contribution

            combined_input = immediate_contribution + self.buffer
            hidden = torch.relu(self.fc1(combined_input))
            output = self.fc2(hidden)
            outputs.append(output)

        return torch.stack(outputs, dim=1)


def train_model(model: nn.Module, data_loader: torch.utils.data.DataLoader,
                optimizer: optim.Optimizer, criterion: nn.CrossEntropyLoss, num_epochs: int) -> None:
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0.0
        for inputs, targets in data_loader:
            inputs, targets = inputs.to(DEVICE), targets.to(DEVICE)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs.view(-1, NUM_BINS), targets.view(-1))
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

        avg_loss = total_loss / len(data_loader)
        wandb.log({"epoch": epoch + 1, "loss": avg_loss})
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {avg_loss:.4f}")


def main():
    initialize_wandb()

    continuous_waveforms = generate_waveforms(NUM_SEQUENCES, SEQUENCE_LENGTH, NUM_MODES,
                                              FREQ_RANGE, AMP_RANGE, PHASE_RANGE)
    one_hot_waveforms = discretize_waveforms(continuous_waveforms, NUM_BINS)

    inputs = torch.tensor(one_hot_waveforms, dtype=torch.float32).to(DEVICE)
    targets = torch.tensor(np.argmax(one_hot_waveforms, axis=-1), dtype=torch.long).to(DEVICE)

    dataset = torch.utils.data.TensorDataset(inputs, targets)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DelayedMLP(input_size=NUM_BINS, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    train_model(model, data_loader, optimizer, criterion, NUM_EPOCHS)

    wandb.finish()


if __name__ == "__main__":
    main()
