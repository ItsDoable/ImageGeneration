import os
from tqdm import tqdm

import torch
import torch.nn as nn
import torchvision
from torchsummary import summary
import math

class EvolAE(nn.Module):
    def __init__(self, input_size):
        super().__init__()
        conv_block = (lambda iteration: [nn.Conv2d(3 ** iteration, 3 ** (iteration + 1), (3,3), (1,1), padding=1),
                                     nn.MaxPool2d((2,2), (2,2)),
                                     nn.ReLU()])
        deconv_block = (lambda iteration: [nn.ConvTranspose2d(3 ** (iteration + 1), 3 ** iteration, (2,2), (2,2), padding=0)])
        iterations = int(math.log2(input_size))
        print(iterations)
        conv_blocks = []
        for it in range(iterations):
            conv_blocks.extend(conv_block(it + 1))
        print(conv_blocks)
        deconv_blocks = []
        for it in range(iterations - 1): # Letzter Block hat keine ReLU
            deconv_blocks.extend(deconv_block(iterations - it))
            deconv_blocks.append(nn.ReLU())
        deconv_blocks.extend(deconv_block(1))
        print(deconv_blocks)
        self.encoder = nn.Sequential(*conv_blocks) # Stern ist Unpack-Operator: aus [a, ..., z] wird (a, ..., z)
        self.decoder = nn.Sequential(*deconv_blocks)

    def forward(self, x):
        code = self.encoder(x)
        output = self.decoder(code)
        return output

path = "C:/Users/Tryerand Retryer/Datasets/thumbnails128x128/"
img = torchvision.io.read_image(path + "00000.png").float()
model = EvolAE(128)
print(img.shape)
print(summary(model, (3, 128, 128)))
print(model(img).shape)

optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)
loss = nn.MSELoss()
for epoch in range(10):

    teilmenge = os.listdir(path)[10 * epoch : 10 * (epoch + 1)]

    for file in tqdm(teilmenge):
        img = torchvision.io.read_image(path + file).float()
        optimizer.zero_grad()
        output = model(img)
        curr_loss = loss(output, img)
        curr_loss.backward()
        optimizer.step()
        print(curr_loss.item())
    gen_img = model(img) # Müsste letztes Bild sein
    torchvision.utils.save_image(gen_img, f"gen_img_{epoch}.png")
    torch.save(model.state_dict(), "model.pth")