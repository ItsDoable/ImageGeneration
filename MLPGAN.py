import os
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.utils.data import DataLoader, Dataset
from PIL import Image
import glob

# Parameter
image_size = 32  # Bildgröße
batch_size = 64  # Batch-Größe
latent_dim = 100  # Dimension des latenten Vektors
epochs = 1000  # Anzahl der Epochen
save_interval = 10  # Speichern alle 10 Epochen
#folder_path = "C:\\Users\\Tryerand Retryer\\Datasets\\archive\\faces_dataset_small"  # Bildordner
folder_path = "./rects"
save_path = "simple_gan_128.pth"

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


# Einfacher Generator
class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(latent_dim, 512),
            nn.ReLU(),
            nn.Linear(512, 1024),
            nn.ReLU(),
            nn.Linear(1024, image_size * image_size * 3),
            nn.Tanh()
        )

    def forward(self, z):
        return self.model(z).view(-1, 3, image_size, image_size)


# Einfacher Diskriminator
class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size * image_size * 3, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1),
            nn.Sigmoid()
        )

    def forward(self, img):
        return self.model(img.view(img.size(0), -1))

# Einfacher Enkoder
class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(image_size * image_size * 3, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, latent_dim)
        )

    def forward(self, img):
        return self.model(img.view(img.size(0), -1))


# Modelle initialisieren
generator = Generator().to(device)
discriminator = Discriminator().to(device)
#encoder = Encoder().to(device)

if os.path.exists(save_path):
    try:
        checkpoint = torch.load(save_path)
        generator.load_state_dict(checkpoint['generator'])
        discriminator.load_state_dict(checkpoint['discriminator'])
        #encoder.load_state_dict(checkpoint['encoder'])
    except:
        pass

from torchview import draw_graph

model_graph = draw_graph(generator, input_size=(32, latent_dim), expand_nested=True)

outpath = model_graph.visual_graph.render(
    filename="generator_graph",  # ohne Endung
    format="png",  # z.B. "png", "pdf", "svg"
    view=True,  # öffnet den Viewer des OS
    cleanup=True
)

model_graph = draw_graph(discriminator, input_size=(1, 3, 32, 32), expand_nested=True)

outpath = model_graph.visual_graph.render(
    filename="discriminator_graph",  # ohne Endung
    format="png",  # z.B. "png", "pdf", "svg"
    view=True,  # öffnet den Viewer des OS
    cleanup=True
)

quit()

criterion = nn.BCELoss()
criterion_enc = nn.MSELoss()
optimizer_G = optim.Adam(generator.parameters(), lr=0.0005, betas=(0.5, 0.999))
optimizer_D = optim.Adam(discriminator.parameters(), lr=0.0002, betas=(0.5, 0.999))
#optimizer_E = optim.Adam(encoder.parameters(), lr=0.0002, betas=(0.5, 0.999))

# Training
for epoch in range(epochs):
    for i, imgs in enumerate(dataloader):
        imgs = imgs.to(device)
        real_labels = torch.full((imgs.size(0), 1), 0.9, device=device)  # Statt 1.0 für echte Labels
        fake_labels = torch.full((imgs.size(0), 1), 0.1, device=device)  # Statt 0.0 für Fake-Labels
        # Diskriminator trainieren
        optimizer_D.zero_grad()
        real_loss = criterion(discriminator(imgs), real_labels)
        z = torch.randn(imgs.size(0), latent_dim, device=device)
        fake_imgs = generator(z).detach()
        fake_loss = criterion(discriminator(fake_imgs), fake_labels)
        loss_D = real_loss + fake_loss
        loss_D.backward()
        optimizer_D.step()

        # Generator trainieren
        optimizer_G.zero_grad()
        z = torch.randn(imgs.size(0), latent_dim, device=device)
        fake_imgs = generator(z)
        gen_loss = criterion(discriminator(fake_imgs), real_labels)
        gen_loss.backward()
        optimizer_G.step()

        """# Enkoder trainieren
        optimizer_E.zero_grad()
        fake_imgs_E = fake_imgs.detach()
        enc_loss = criterion_enc(encoder(fake_imgs_E), z)
        enc_loss.backward()
        optimizer_E.step()"""

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
            #'encoder': encoder.state_dict()
        }, save_path)
        with torch.no_grad():
            sample_z = torch.randn(16, latent_dim, device=device)
            samples = generator(sample_z)
            # reconstructed_samples = generator(encoder(samples))
            # vutils.save_image(torch.cat((samples, reconstructed_samples)), f"simple_samples_epoch_{epoch + 1}.png", normalize=True)
            vutils.save_image(samples, f"simple_samples_epoch_{epoch + 1}.png", normalize=True)

        print(f"Modell gespeichert und Beispielbilder generiert (simple_samples_epoch_{epoch + 1}.png)")

print("Training abgeschlossen.")