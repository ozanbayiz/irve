import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb

# --- WandB Init ---
wandb.init(project="sparse-autoencoder", config={
    "input_size": 28 * 28,
    "hidden_size": (28 * 28) // 16,
    "batch_size": 64,
    "num_epochs": 20,
    "learning_rate": 1e-3,
    "l1_lambda": 1e-5
})

config = wandb.config

# --- Hyperparameters ---
input_size = config.input_size
hidden_size = config.hidden_size
batch_size = config.batch_size
num_epochs = config.num_epochs
learning_rate = config.learning_rate
l1_lambda = config.l1_lambda

# --- Load MNIST (flattened images) ---
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Lambda(lambda x: x.view(-1))  # Flatten to (784,)
])

mnist_train = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
mnist_val = datasets.MNIST(root='./data', train=False, download=True, transform=transform)

train_loader = DataLoader(mnist_train, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(mnist_val, batch_size=batch_size, shuffle=False)

# --- Sparse Autoencoder ---
class SparseAutoencoder(nn.Module):
    def __init__(self):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Identity()
        )

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z

# --- Initialize Model ---
model = SparseAutoencoder()
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# --- Training Loop ---
for epoch in range(num_epochs):
    model.train()
    total_train_loss = 0.0
    for batch, _ in train_loader:
        optimizer.zero_grad()
        recon, encoded = model(batch)
        recon_loss = criterion(recon, batch)
        l1_loss = l1_lambda * torch.norm(encoded, 1)
        loss = recon_loss + l1_loss
        loss.backward()
        optimizer.step()
        total_train_loss += loss.item()

    avg_train_loss = total_train_loss / len(train_loader)

    # --- Validation ---
    model.eval()
    total_val_loss = 0.0
    with torch.no_grad():
        for batch, _ in val_loader:
            recon, encoded = model(batch)
            recon_loss = criterion(recon, batch)
            l1_loss = l1_lambda * torch.norm(encoded, 1)
            loss = recon_loss + l1_loss
            total_val_loss += loss.item()

    avg_val_loss = total_val_loss / len(val_loader)

    # --- Log both training & validation losses
    wandb.log({
        "epoch": epoch + 1,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss
    })

    print(f"Epoch [{epoch+1}/{num_epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")
