from model import CAEC
import torch
from PIL import Image
import torchvision
import matplotlib.pyplot as plt
from torchsummary import summary

loader = torchvision.transforms.Compose([torchvision.transforms.ToTensor()])

def image_loader(image_name):
    """load image, returns cuda tensor"""
    image = Image.open(image_name)
    image = image.convert('RGB')
    image = loader(image).float()
    image = image.unsqueeze(0) 
    return image

image = image_loader('autoencoder/1_21.png') 



encoder_arhitecture_layers_size = [
        64,
        64,
        "max",
        128,
        128,
        "max",
        256,
        256,
        256,
        "max",
        512,
        512,
        512,
        "max",
        512,
        512,
        512,
    ]
decoder_arhitecture_layers_size = [
    784,
    16,
    32,
    64,
    128,
    64,
    3
]
model = CAEC(in_channels=3, encoder_arhitecture_layers_size=encoder_arhitecture_layers_size,decoder_arhitecture_layers_size=decoder_arhitecture_layers_size)


model.load_state_dict(torch.load('autoencoder/model_weights_4.pth',map_location=torch.device('cpu')))
x_return = model(image).detach()
plt.imshow(  x_return.squeeze(0).permute(1, 2, 0)  )
plt.show()
print(summary(model,(3, 224, 224)))