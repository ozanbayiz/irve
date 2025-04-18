import h5py
import torch
from torch.utils.data import Dataset, DataLoader
import torch.nn as nn
import torch.optim as optim

'''
PARAMETERS TO CHANGE
INPUT DIMENSION --> SIZE OF VECTOR INPUTS
AGE, RACE, GENDER CLASSES


Also change the torch.save parameters as well
'''

class FairFaceDataset(Dataset):
    def __init__(self, hdf5_path, mode='training'):
        self.hdf5_path = hdf5_path
        self.mode = mode
        
        # Might be useful to have dimensions
        with h5py.File(hdf5_path, 'r') as f:
            self.length = f[mode]['encoded'].shape[0]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with h5py.File(self.hdf5_path, 'r') as f:
            group = f[self.mode]
    
            features = torch.tensor(group['encoded'][idx], dtype=torch.float32)

            age = torch.tensor(group['labels']['age'][idx], dtype=torch.long)
            gender = torch.tensor(group['labels']['gender'][idx], dtype=torch.long)
            race = torch.tensor(group['labels']['race'][idx], dtype=torch.long)
            
            return features, (age, gender, race)

class SimpleTaskClassifiers(nn.Module):
    def __init__(self, input_dim, age_classes, gender_classes, race_classes):
        super(SimpleTaskClassifiers, self).__init__()
        
        self.age_classifier = nn.Linear(input_dim, age_classes)
        self.gender_classifier = nn.Linear(input_dim, gender_classes)
        self.race_classifier = nn.Linear(input_dim, race_classes)
    
    def forward(self, x):
        age_out = self.age_classifier(x)
        gender_out = self.gender_classifier(x)
        race_out = self.race_classifier(x)
        
        return age_out, gender_out, race_out

def train_simple_classifiers(train_hdf5, val_hdf5, num_epochs=10, batch_size=32, learning_rate=0.001, save_path='model.pth'):
    # Create datasets
    train_dataset = FairFaceDataset(train_hdf5, mode='training')
    val_dataset = FairFaceDataset(val_hdf5, mode='validation')
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    
    # Determine input dimensions automatically from data
    sample_features, _ = train_dataset[0]
    input_dim = sample_features.size(0)
    
    model = SimpleTaskClassifiers(
        input_dim=input_dim,
        age_classes=9,    # Adjust as needed
        gender_classes=2, # Adjust as needed
        race_classes=7    # Adjust as needed
    )
    
    criterion = nn.CrossEntropyLoss()
    
    # Create separate optimizers for each task
    age_optimizer = optim.Adam(model.age_classifier.parameters(), lr=learning_rate)
    gender_optimizer = optim.Adam(model.gender_classifier.parameters(), lr=learning_rate)
    race_optimizer = optim.Adam(model.race_classifier.parameters(), lr=learning_rate)
    
    for epoch in range(num_epochs):
        model.train()
        total_age_loss = 0.0
        total_gender_loss = 0.0
        total_race_loss = 0.0
        
        # Training loop with shuffled batches
        for features, (age_labels, gender_labels, race_labels) in train_loader:
            # Forward pass - get all predictions at once
            age_pred, gender_pred, race_pred = model(features)
            
            # Age task
            age_optimizer.zero_grad()
            age_loss = criterion(age_pred, age_labels)
            age_loss.backward(retain_graph=True)  # Retain graph for multiple backward passes
            age_optimizer.step()
            total_age_loss += age_loss.item()
            
            # Gender task
            gender_optimizer.zero_grad()
            gender_loss = criterion(gender_pred, gender_labels)
            gender_loss.backward(retain_graph=True)
            gender_optimizer.step()
            total_gender_loss += gender_loss.item()
            
            # Race task
            race_optimizer.zero_grad()
            race_loss = criterion(race_pred, race_labels)
            race_loss.backward()  # No need to retain graph for the last backward
            race_optimizer.step()
            total_race_loss += race_loss.item()
        
        # Validation
        model.eval()
        val_age_loss, val_gender_loss, val_race_loss = 0.0, 0.0, 0.0
        correct_age, correct_gender, correct_race = 0, 0, 0
        total = 0
        
        with torch.no_grad():
            for features, (age_labels, gender_labels, race_labels) in val_loader:
                age_pred, gender_pred, race_pred = model(features)
                
                # Calculate individual validation losses
                val_age_loss += criterion(age_pred, age_labels).item()
                val_gender_loss += criterion(gender_pred, gender_labels).item()
                val_race_loss += criterion(race_pred, race_labels).item()
                
                # Calculate accuracy
                _, age_pred = torch.max(age_pred, 1)
                _, gender_pred = torch.max(gender_pred, 1)
                _, race_pred = torch.max(race_pred, 1)
                
                correct_age += (age_pred == age_labels).sum().item()
                correct_gender += (gender_pred == gender_labels).sum().item()
                correct_race += (race_pred == race_labels).sum().item()
                total += age_labels.size(0)
        
        # Print detailed statistics
        print(f'Epoch {epoch+1}/{num_epochs}')
        print(f'Train Losses - Age: {total_age_loss/len(train_loader):.4f}')
        print(f'             - Gender: {total_gender_loss/len(train_loader):.4f}')
        print(f'             - Race: {total_race_loss/len(train_loader):.4f}')
        print(f'Val Losses   - Age: {val_age_loss/len(val_loader):.4f}')
        print(f'             - Gender: {val_gender_loss/len(val_loader):.4f}')
        print(f'             - Race: {val_race_loss/len(val_loader):.4f}')
        print(f'Accuracies   - Age: {100 * correct_age/total:.2f}%')
        print(f'             - Gender: {100 * correct_gender/total:.2f}%') 
        print(f'             - Race: {100 * correct_race/total:.2f}%')
        print('-' * 50)
    
    # Save model with all necessary information
    torch.save({
        'model_state_dict': model.state_dict(),
        'epoch': num_epochs,
        'input_dim': input_dim,
        'age_classes': 9,
        'gender_classes': 2,
        'race_classes': 7
    }, save_path)
    
    print(f"Model saved to {save_path}")
    
    return model