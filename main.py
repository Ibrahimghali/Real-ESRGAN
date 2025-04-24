import os 
import torch
from PIL import Image
import numpy as np
from RealESRGAN import RealESRGAN
import glob

def main() -> int:
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = RealESRGAN(device, scale=4)
    
    # Check if weights file exists using a glob pattern to match regardless of snapshot hash
    weights_pattern = 'weights/models--sberbank-ai--Real-ESRGAN/snapshots/*/RealESRGAN_x4.pth'
    existing_weights = glob.glob(weights_pattern)
    
    if existing_weights:
        print(f"Loading existing weights from {existing_weights[0]}")
        model.load_weights(existing_weights[0], download=False)
    else:
        print(f"Weights not found locally, downloading...")
        # Create directory if it doesn't exist
        os.makedirs('weights', exist_ok=True)
        model.load_weights('weights/RealESRGAN_x4.pth', download=True)
    
    for i, image in enumerate(os.listdir("inputs")):
        image = Image.open(f"inputs/{image}").convert('RGB')
        sr_image = model.predict(image)
        sr_image.save(f'results/{i}.png')

if __name__ == '__main__':
    main()