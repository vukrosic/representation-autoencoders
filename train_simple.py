import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from src.stage1.rae import RAE
from tqdm import tqdm

def train():
    # 1. Setup Device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # 2. Hyperparameters
    image_size = 224
    batch_size = 16
    lr = 1e-4
    epochs = 10
    data_path = "./data" # User should specify their data path

    # 3. Model Initialization (Simple DINOv2-based RAE)
    # Note: These paths and configs should match what's available or desired.
    # Default parameters for RAE use facebook/dinov2-base and vit_mae-base.
    model = RAE(
        encoder_cls='Dinov2withNorm',
        encoder_config_path='facebook/dinov2-base',
        decoder_config_path='facebook/vit-mae-base',
        noise_tau=0.0 # No noise for simple reconstruction training
    ).to(device)
    
    # Freeze encoder, train decoder
    model.encoder.eval()
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.decoder.train()

    # 4. Data Loading
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    if not os.path.exists(data_path):
        print(f"Directory {data_path} not found. Please create it and add 'train' folder with class subdirectories.")
        return

    dataset = datasets.ImageFolder(root=data_path, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    # 5. Optimizer & Loss
    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=lr)
    criterion = nn.MSELoss()

    # 6. Training Loop
    for epoch in range(epochs):
        model.decoder.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0
        
        for images, _ in pbar:
            images = images.to(device)
            
            # Forward
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
            
        print(f"Epoch {epoch+1} Average Loss: {epoch_loss / len(dataloader)}")
        
        # Save simple checkpoint
        torch.save(model.state_dict(), f"rae_simple_epoch_{epoch+1}.pt")

if __name__ == "__main__":
    train()
