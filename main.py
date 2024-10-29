import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import wandb
from tqdm import tqdm
import matplotlib.pyplot as plt


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
LEARNING_RATE = 1e-2
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
        self.delay_gate = nn.Linear(input_size, input_size)
        self.sigmoid = nn.Sigmoid()
        self.buffer = None

        self.mlp = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Linear(hidden_size, output_size)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, seq_len, input_size = x.size()
        assert len(x.shape) == 3, f"Expected input x to have 3 dimensions, got {x.shape}"
        self.buffer = torch.zeros((batch_size, input_size), device=x.device)
        outputs = []

        for t in range(seq_len):
            current_input = x[:, t, :]
            assert current_input.shape == (
                batch_size, input_size), f"Expected current_input shape to be {(batch_size, input_size)}, got {current_input.shape}"

            decay_weights = self.sigmoid(self.delay_gate(current_input))
            assert decay_weights.shape == (
                batch_size, input_size), f"Expected decay_weights shape to be {(batch_size, input_size)}, got {decay_weights.shape}"

            immediate_contribution = current_input * decay_weights
            assert immediate_contribution.shape == (
                batch_size, input_size), f"Expected immediate_contribution shape to be {(batch_size, input_size)}, got {immediate_contribution.shape}"

            delayed_contribution = (1 - decay_weights) * current_input
            assert delayed_contribution.shape == (
                batch_size, input_size), f"Expected delayed_contribution shape to be {(batch_size, input_size)}, got {delayed_contribution.shape}"
            self.buffer += delayed_contribution

            buffer_decay_weights = self.sigmoid(self.delay_gate(self.buffer))
            assert buffer_decay_weights.shape == (
                batch_size, input_size), f"Expected buffer_decay_weights shape to be {(batch_size, input_size)}, got {buffer_decay_weights.shape}"

            buffer_release = self.buffer * buffer_decay_weights
            assert buffer_release.shape == (
                batch_size, input_size), f"Expected buffer_release shape to be {(batch_size, input_size)}, got {buffer_release.shape}"

            self.buffer = self.buffer * (1 - buffer_decay_weights)
            assert self.buffer.shape == (
                batch_size, input_size), f"Expected buffer shape to be {(batch_size, input_size)}, got {self.buffer.shape}"

            combined_input = immediate_contribution + buffer_release
            assert combined_input.shape == (
                batch_size, input_size), f"Expected combined_input shape to be {(batch_size, input_size)}, got {combined_input.shape}"

            output = self.mlp(combined_input)
            assert output.shape == (batch_size, output.size(-1)
                                    ), f"Expected output shape to be {(batch_size, output.size(-1))}, got {output.shape}"
            outputs.append(output)

            wandb.log({
                "timestep": t,
                "buffer_l2_norm": torch.norm(self.buffer, p=2).item(),
                "decay_weight_mean": decay_weights.mean().item(),
            })

        final_output = torch.stack(outputs, dim=1)
        assert final_output.shape == (batch_size, seq_len, output.size(
            -1)), f"Expected final output shape to be {(batch_size, seq_len, output.size(-1))}, got {final_output.shape}"
        return final_output


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


def inference_and_plot(model, inputs):
    model.eval()
    with torch.no_grad():
        predictions = []
        teacher_forcing = True

        for t in range(inputs.size(1)):
            if t == inputs.size(1) // 2:
                teacher_forcing = False

            if teacher_forcing:
                current_input = inputs[:, t, :]
            else:
                current_input = predictions[-1]

            output = model(current_input.unsqueeze(1))
            predictions.append(output.squeeze(1))

        predictions = torch.stack(predictions, dim=1)
        original_data = inputs[0].cpu().numpy()
        predicted_data = predictions[0].argmax(dim=-1).cpu().numpy()

    plt.figure(figsize=(12, 6))
    plt.plot(range(1, len(original_data)), original_data.argmax(axis=-1)[1:], label="Original Data", color="blue")

    # Shift predictions back by one timestep on the x-axis
    plt.plot(range(len(predicted_data) - 1), predicted_data[:-1], 'o', label="Predicted Output", color="orange")

    plt.xlabel("Timestep")
    plt.ylabel("Bin Index")
    plt.title("Inference: Teacher-Forced vs. Autoregressive Prediction")
    plt.legend()
    plt.show()


def main():
    initialize_wandb()

    continuous_waveforms = generate_waveforms(NUM_SEQUENCES, SEQUENCE_LENGTH, NUM_MODES,
                                              FREQ_RANGE, AMP_RANGE, PHASE_RANGE)
    one_hot_waveforms = discretize_waveforms(continuous_waveforms, NUM_BINS)

    inputs = torch.tensor(one_hot_waveforms, dtype=torch.float32).to(DEVICE)

    # Shift targets by one timestep for next-token prediction
    targets = torch.tensor(np.argmax(one_hot_waveforms, axis=-1), dtype=torch.long).to(DEVICE)
    targets = targets[:, 1:]  # Shift to the next timestep for prediction
    inputs = inputs[:, :-1]   # Remove last input timestep for alignment

    dataset = torch.utils.data.TensorDataset(inputs, targets)
    data_loader = torch.utils.data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

    model = DelayedMLP(input_size=NUM_BINS, hidden_size=HIDDEN_SIZE, output_size=OUTPUT_SIZE).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    criterion = nn.CrossEntropyLoss()

    train_model(model, data_loader, optimizer, criterion, NUM_EPOCHS)

    for inputs, _ in data_loader:
        inputs = inputs.to(DEVICE)
        inference_and_plot(model, inputs)
        break

    wandb.finish()


if __name__ == "__main__":
    main()
