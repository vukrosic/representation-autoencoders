import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
from src.stage1.rae import RAE
from tqdm import tqdm
import os
import json

def run_ablation(name, encoder_cls, encoder_path, encoder_params, noise_tau, epochs=1):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n--- Starting Ablation: {name} ---")
    
    image_size = 224
    batch_size = 32
    lr = 1e-4

    model = RAE(
        encoder_cls=encoder_cls,
        encoder_config_path=encoder_path,
        encoder_params=encoder_params,
        decoder_config_path='facebook/vit-mae-base',
        decoder_patch_size=14 if 'dinov2' in encoder_path else 16, # Adjust for patch size differences if any
        noise_tau=noise_tau
    ).to(device)
    
    model.encoder.eval()
    for param in model.encoder.parameters():
        param.requires_grad = False
    model.decoder.train()

    transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
    ])
    dataset = datasets.CIFAR10(root="./data", train=True, download=True, transform=transform)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=4)

    optimizer = torch.optim.AdamW(model.decoder.parameters(), lr=lr)
    criterion = nn.MSELoss()

    total_loss = 0
    for epoch in range(epochs):
        model.decoder.train()
        pbar = tqdm(dataloader, desc=f"Epoch {epoch+1}/{epochs}")
        epoch_loss = 0
        for images, _ in pbar:
            images = images.to(device)
            reconstructed = model(images)
            loss = criterion(reconstructed, images)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            pbar.set_postfix({"loss": loss.item()})
        avg_loss = epoch_loss / len(dataloader)
        print(f"{name} Epoch {epoch+1} Average Loss: {avg_loss}")
        total_loss = avg_loss

    return total_loss

if __name__ == "__main__":
    results = {}
    
    # 1. SigLIP Ablation
    siglip_loss = run_ablation(
        "SigLIP", 
        "SigLIP2wNorm", 
        "google/siglip-base-patch16-224", 
        {"model_name": "google/siglip-base-patch16-224"}, 
        noise_tau=0.0
    )
    results["SigLIP"] = siglip_loss

    # 2. DINOv2 with Noise Ablation
    noisy_loss = run_ablation(
        "DINOv2_Noisy", 
        "Dinov2withNorm", 
        "facebook/dinov2-base", 
        {"dinov2_path": "facebook/dinov2-base"}, 
        noise_tau=0.5
    )
    results["DINOv2_Noisy"] = noisy_loss

    with open("ablation_results.json", "w") as f:
        json.dump(results, f)
    print("\nAblations complete. Results saved to ablation_results.json")
