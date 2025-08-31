import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob
from matplotlib import pyplot as plt

# Parameter
image_size = 32  # Bildgröße
batch_size = 64  # Batch-Größe
latent_dim = 100  # Dimension des latenten Vektors
epochs = 1000  # Anzahl der Epochen
save_interval = 10  # Speichern alle 10 Epochen
folder_path = "C:\\Users\\Tryerand Retryer\\Datasets\\faces_32_32"
#folder_path = "./rects"
save_path = "gan_model_128.pth"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# Dataset-Klasse
class FaceDataset(Dataset):
    def __init__(self, folder_path, transform=None):
        self.image_paths = glob.glob(f"{folder_path}/*.jpg") + glob.glob(f"{folder_path}/*.png")
        self.transform = transform

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img = Image.open(self.image_paths[idx]).convert("RGB")
        if self.transform:
            img = self.transform(img)
        return img


# Transformationen
transform = transforms.Compose([
    transforms.Resize((image_size, image_size)),
    transforms.ToTensor(),
    transforms.Normalize([0.5], [0.5])
])

dataset = FaceDataset(folder_path, transform=transform)
dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 256, 4, 1, 0, bias=False),  # 1x1 → 4x4
            nn.BatchNorm2d(256),
            nn.ReLU(),

            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),  # 4x4 → 8x8
            nn.BatchNorm2d(128),
            nn.ReLU(),

            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),  # 8x8 → 16x16
            nn.BatchNorm2d(64),
            nn.ReLU(),

            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),  # 16x16 → 32x32
            nn.Tanh()
        )

    def forward(self, z):
        z = z.view(z.size(0), z.size(1), 1, 1)
        return self.model(z)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),  # 32x32 → 16x16
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(64, 128, 4, 2, 1, bias=False),  # 16x16 → 8x8
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(128, 256, 4, 2, 1, bias=False),  # 8x8 → 4x4
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(256, 512, 4, 2, 1, bias=False),  # 4x4 → 2x2
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(512, 1, 2, 1, 0, bias=False),  # 2x2 → 1x1
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img).view(img.size(0), -1)  # Ausgabe (batch_size, 1)


def load_models():
    if os.path.exists(save_path):
        checkpoint = torch.load(save_path)
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])

# Modelle initialisieren
generator = Generator().to(device)
discriminator = Discriminator().to(device)

load_models()

criterion = nn.BCELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training
for epoch in range(epochs):
    for i, imgs in enumerate(dataloader):
        imgs = imgs.to(device)
        real_labels = torch.full((imgs.size(0), 1), 0.9, device=device)  # Statt 1.0 für echte Labels
        fake_labels = torch.full((imgs.size(0), 1), 0.1, device=device)  # Statt 0.0 für Fake-Labels

        # Diskriminator trainieren
        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(imgs), real_labels)
        z = torch.randn(imgs.size(0), latent_dim, 1, 1, device=device)
        fake_imgs = generator(z).detach()
        fake_loss = criterion(discriminator(fake_imgs), fake_labels)
        loss_D = real_loss + fake_loss
        loss_D.backward()
        optimizer_D.step()

        for i in range(3):
            # Generator trainieren
            optimizer_G.zero_grad()
            z = torch.randn(imgs.size(0), latent_dim, 1, 1, device=device)
            fake_imgs = generator(z)
            gen_loss = criterion(discriminator(fake_imgs), real_labels)
            gen_loss.backward()
            optimizer_G.step()

        sample_z = torch.randn(16, latent_dim, device=device)
        samples = generator(sample_z)

        # grid = vutils.make_grid(torch.cat((samples, reconstructed_samples)), normalize=True, scale_each=True,
        grid = vutils.make_grid(samples, normalize=True, scale_each=True, nrow=8)
        # Plotte das Bild mit Matplotlib
        plt.figure(figsize=(8, 8))
        plt.imshow(grid.permute(1, 2, 0).cpu().numpy())  # Tensor in NumPy-Format umwandeln
        plt.axis("off")  # Achsen ausblenden
        plt.show()

    print(f"Epoch {epoch + 1}/{epochs} - Loss D: {loss_D.item():.4f}, Loss G: {gen_loss.item():.4f}")

    # Speichern alle 20 Epochen
    if (epoch + 1) % save_interval == 0:
        torch.save({
            'generator': generator.state_dict(),
            'discriminator': discriminator.state_dict()
        }, save_path)
        with torch.no_grad():
            sample_z = torch.randn(16, latent_dim, 1, 1, device=device)
            samples = generator(sample_z)
            vutils.save_image(samples, f"samples_epoch_{epoch + 1}.png", normalize=True)
        print(f"Model gespeichert und Beispielbilder generiert (samples_epoch_{epoch + 1}.png)")

print("Training abgeschlossen.")