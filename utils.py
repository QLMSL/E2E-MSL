import torch
from torchvision import transforms
from PIL import Image
import matplotlib.pyplot as plt



def load_image_as_tensor(img_path):
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),  # 确保单通道
        transforms.ToTensor(),  # 自动转换为[0,1]范围且格式为(C,H,W)
    ])
    img = Image.open(img_path)
    img_tensor = transform(img).unsqueeze(0)  # 添加batch维度 -> (1,1,H,W)
    img_tensor.requires_grad_(True)  # 启用梯度
    return img_tensor

def visualize_masks(masks, num_masks=3):
    for i in range(num_masks):
        mask = masks[0, i].detach().cpu().numpy()  # 获取第i个掩码并转换为numpy数组
        plt.subplot(1, num_masks, i + 1)  # 创建子图
        plt.imshow(mask, cmap='gray')  # 使用热力图显示掩码
        plt.title(f"Mask {i + 1}")
        plt.axis('off')  # 不显示坐标轴
    plt.show()