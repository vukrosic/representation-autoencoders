import torch
import torchvision.transforms as T
from torchvision.datasets import CIFAR10
from src.stage1.rae import RAE
from PIL import Image
import os

def test_and_visualize():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # 1. Load Model
    # We use the checkpoint from epoch 1
    checkpoint_path = "rae_simple_epoch_1.pt"
    if not os.path.exists(checkpoint_path):
        print(f"Checkpoint {checkpoint_path} not found. Please wait for at least one epoch to finish.")
        return

    model = RAE(
        encoder_cls='Dinov2withNorm',
        encoder_config_path='facebook/dinov2-base',
        encoder_params={'dinov2_path': 'facebook/dinov2-base'},
        decoder_config_path='facebook/vit-mae-base',
        decoder_patch_size=14,
        noise_tau=0.0 
    ).to(device)
    
    state_dict = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state_dict)
    model.eval()
    print(f"Loaded model from {checkpoint_path}")

    # 2. Prepare Data
    transform = T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
    ])
    dataset = CIFAR10(root="./data", train=False, download=True, transform=transform)
    
    # 3. Inference
    indices = [0, 10, 20, 30, 40] # Test on a few samples
    results = []
    
    with torch.no_grad():
        for idx in indices:
            img, label = dataset[idx]
            img = img.unsqueeze(0).to(device)
            
            reconstructed = model(img)
            
            # Convert to PIL
            orig_img = T.ToPILImage()(img.squeeze().cpu())
            rec_img = T.ToPILImage()(reconstructed.squeeze().cpu().clamp(0, 1))
            
            results.append((orig_img, rec_img))

    # 4. Save a comparison grid
    grid_w = 224 * 2
    grid_h = 224 * len(indices)
    grid = Image.new('RGB', (grid_w, grid_h))
    
    for i, (orig, rec) in enumerate(results):
        grid.paste(orig, (0, i * 224))
        grid.paste(rec, (224, i * 224))
    
    output_path = "reconstruction_results.png"
    grid.save(output_path)
    print(f"Saved comparison grid to {output_path}")

if __name__ == "__main__":
    test_and_visualize()
