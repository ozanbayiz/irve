import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import wandb

# --- Sparse Autoencoder ---
class SparseAutoencoder(nn.Module):
    def __init__(self, input_size, hidden_size, batch_size = 64, num_epochs = 20, lr = 1e-3, l1 = 1e-5):
        super(SparseAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, input_size),
            nn.Identity()
        )

        self.input_size = input_size
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.epochs = num_epochs
        self.lr = lr
        self.l1 = l1
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.parameters(), lr=lr)

    def forward(self, x):
        z = self.encoder(x)
        out = self.decoder(z)
        return out, z
    
    def encode(self, x):
        return self.encoder(x)
    
    def decode(self, x):
        return self.decoder(x)
    
    def train_with_val(self, train_data, val_data):

        # --- WandB Init ---
        wandb.init(project="sparse-autoencoder", config={
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "batch_size": self.batch_size,
            "num_epochs": self.epochs,
            "learning_rate": self.lr,
            "l1_lambda": self.l1
        })

        train_loader = DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=self.batch_size, shuffle=False)

        # --- Training Loop ---
        for epoch in range(self.epochs):
            self.train()
            total_train_loss = 0.0
            for batch, _ in train_loader:
                self.optimizer.zero_grad()
                recon, encoded = self(batch)
                recon_loss = self.criterion(recon, batch)
                l1_loss = self.l1 * torch.norm(encoded, 1)
                loss = recon_loss + l1_loss
                loss.backward()
                self.optimizer.step()
                total_train_loss += loss.item()

            avg_train_loss = total_train_loss / len(train_loader)

            # --- Validation ---
            self.eval()
            total_val_loss = 0.0
            with torch.no_grad():
                for batch, _ in val_loader:
                    recon, encoded = self(batch)
                    recon_loss = self.criterion(recon, batch)
                    l1_loss = self.l1 * torch.norm(encoded, 1)
                    loss = recon_loss + l1_loss
                    total_val_loss += loss.item()

            avg_val_loss = total_val_loss / len(val_loader)

            # --- Log both training & validation losses
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": avg_train_loss,
                "val_loss": avg_val_loss
            })

            print(f"Epoch [{epoch+1}/{self.epochs}], Train Loss: {avg_train_loss:.4f}, Val Loss: {avg_val_loss:.4f}")

    def save_parameters(self, save_path):
        torch.save(self.state_dict(), save_path)
        print(f"Model parameters saved to {save_path}")