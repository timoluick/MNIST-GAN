import torch
from torch import nn, optim
from torch.autograd.variable import Variable
from torchvision import transforms, datasets
from torch.utils.data.dataloader import DataLoader
import matplotlib.pyplot as plt
from IPython import display
import numpy as np

transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize([0], [1])
     ])
data = datasets.MNIST(root='./dataset', train=True, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=True)
num_batches = len(data_loader)


class DiscriminatorNet(nn.Module):

    def __init__(self):
        super(DiscriminatorNet, self).__init__()

        self.conv1 = nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2, stride=2)
        self.linear1 = nn.Linear(392, 20)
        self.linear2 = nn.Linear(20, 1)

    def forward(self, x):
        a = x.shape[0]
        x = torch.tanh(self.conv1(x))
        x = x.reshape(a, 1, -1)
        x = torch.sigmoid(self.linear1(x))
        x = torch.sigmoid(self.linear2(x))
        return x


discriminator = DiscriminatorNet()


def images_to_vectors(images):
    return images.view(images.size(0), 784)


def vectors_to_images(vectors):
    return vectors.view(vectors.size(0), 1, 28, 28)


class GeneratorNet(nn.Module):

    def __init__(self, in_size):
        super(GeneratorNet, self).__init__()

        self.linear1 = nn.Linear(in_size, 900)
        self.conv1 = torch.nn.Conv2d(in_channels=1, out_channels=2, kernel_size=2)
        self.conv2 = torch.nn.Conv2d(in_channels=2, out_channels=1, kernel_size=2)

    def forward(self, x):
        x = torch.sigmoid(self.linear1(x))
        x = x.reshape((-1, 1, 30, 30))
        x = torch.sigmoid(self.conv1(x))
        x = torch.sigmoid(self.conv2(x))
        return x


generator = GeneratorNet(in_size=50)


def noise(size):
    n = Variable(torch.randn(size, 50))
    return n


d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
g_optimizer = optim.Adam(generator.parameters(), lr=0.001)

loss = torch.nn.BCELoss()


def ones_target(size):
    data = Variable(torch.ones(size, 1, 1))
    return data


def zeros_target(size):
    data = Variable(torch.zeros(size, 1, 1))
    return data


def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)
    # Reset gradients
    optimizer.zero_grad()

    # 1.1 Train on Real Data
    prediction_real = discriminator(real_data)
    # Calculate error and backpropagate
    error_real = loss(prediction_real, ones_target(N))
    error_real.backward()

    # 1.2 Train on Fake Data
    prediction_fake = discriminator(fake_data)
    # Calculate error and backpropagate
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()

    # 1.3 Update weights with gradients
    optimizer.step()

    # Return error and predictions for real and fake inputs
    return error_real + error_fake, prediction_real, prediction_fake


def train_generator(optimizer, fake_data):
    N = fake_data.size(0)
    # Reset gradients
    optimizer.zero_grad()
    # Sample noise and generate fake data
    prediction = discriminator(fake_data)
    # Calculate error and backpropagate
    error = loss(prediction, ones_target(N))
    error.backward()
    # Update weights with gradients
    optimizer.step()
    # Return error
    return error


def plot_durations(x, y, y_net):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('x')
    plt.ylabel('y')

    lines = plt.plot(x, y, x, y_net)
    plt.setp(lines[0], linewidth=1, color='r')
    plt.setp(lines[1], linewidth=1, color='g')

    plt.pause(0.0001)
    display.clear_output(wait=True)
    display.display(plt.gcf())


num_test_samples = 16
test_noise = noise(num_test_samples)

# Create logger instance
# Total number of epochs to train

losses_disc = []
losses_gen = []
runs = []
run = 0

num_epochs = 10000
for epoch in range(num_epochs):
    disc_error = []
    gen_error = []
    runs.append(epoch)
    for n_batch, (image_batch, label_batch) in enumerate(data_loader):
        N = image_batch.size(0)
        real_data = image_batch
        fake_data = generator(noise(N)).detach()
        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)
        fake_data = generator(noise(N))
        g_error = train_generator(g_optimizer, fake_data)

        disc_error.append(d_error.detach().numpy())
        gen_error.append(g_error.detach().numpy())
    losses_disc.append(np.mean(disc_error))
    losses_gen.append(np.mean(gen_error))
    #plot_durations(runs, losses_disc, losses_gen)

    print('Epoch: ' + str(epoch))

    f, axarr = plt.subplots(3, 3)
    axarr[0, 0].imshow(generator(noise(1)).detach().numpy()[0, 0], cmap='gray')
    axarr[0, 1].imshow(generator(noise(1)).detach().numpy()[0, 0], cmap='gray')
    axarr[0, 2].imshow(generator(noise(1)).detach().numpy()[0, 0], cmap='gray')
    axarr[1, 0].imshow(generator(noise(1)).detach().numpy()[0, 0], cmap='gray')
    axarr[1, 1].imshow(generator(noise(1)).detach().numpy()[0, 0], cmap='gray')
    axarr[1, 2].imshow(generator(noise(1)).detach().numpy()[0, 0], cmap='gray')
    axarr[2, 0].imshow(generator(noise(1)).detach().numpy()[0, 0], cmap='gray')
    axarr[2, 1].imshow(generator(noise(1)).detach().numpy()[0, 0], cmap='gray')
    axarr[2, 2].imshow(generator(noise(1)).detach().numpy()[0, 0], cmap='gray')
    plt.savefig('a.png')
    f.close()
