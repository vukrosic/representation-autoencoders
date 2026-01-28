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
    batch_size = 32 # Increased batch size for real data
    lr = 1e-4
    epochs = 10

    # 3. Model Initialization (Simple DINOv2-based RAE)
    # Using decoder_patch_size=14 to reconstruct 224x224 from 16x16=256 patches.
    # DINOv2-base (224) produces 16x16 patches. 16 * 14 = 224.
    model = RAE(
        encoder_cls='Dinov2withNorm',
        encoder_config_path='facebook/dinov2-base',
        encoder_params={'dinov2_path': 'facebook/dinov2-base'},
        decoder_config_path='facebook/vit-mae-base',
        decoder_patch_size=14,
        noise_tau=0.0 
    ).to(device)
    
    # Freeze encoder, train decoder
    model.encoder.eval()
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.decoder.train()

    # 4. Data Loading - Using CIFAR-10 as real data
    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    
    print("Loading real dataset (CIFAR-10)...")
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
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
