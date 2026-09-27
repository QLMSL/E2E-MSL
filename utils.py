import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt



def load_image_as_tensor(img_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  
        transforms.ToTensor(),  
    ])
    img = Image.open(img_path)
    img_tensor = transform(img).unsqueeze(0) 
    img_tensor.requires_grad_(True) 
    return img_tensor

def visualize_masks(masks, num_masks=3):
    for i in range(num_masks):
        mask = masks[0, i].detach().cpu().numpy() 
        plt.subplot(1, num_masks, i + 1)  
        plt.imshow(mask, cmap='gray')  
        plt.title(f"Mask {i + 1}")
        plt.axis('off')
    plt.show()
