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
data = datasets.MNIST(root='./dataset', train=False, transform=transform, download=True)
data_loader = torch.utils.data.DataLoader(data, batch_size=128, shuffle=True)
num_batches = len(data_loader)


class DiscriminatorNet(nn.Module):

    def __init__(self):
        super(DiscriminatorNet, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=1, kernel_size=2, stride=1),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        self.linear1 = nn.Sequential(
            nn.Linear(676, 20),
            nn.LeakyReLU(0.1),
            nn.Dropout(0.2)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(20, 1),
            nn.Sigmoid()
        )


    def forward(self, x):
        n = x.size(0)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.reshape(n, 1, -1)
        x = self.linear1(x)
        x = self.linear2(x)
        return x


discriminator = DiscriminatorNet()


class GeneratorNet(nn.Module):

    def __init__(self, in_size):
        super(GeneratorNet, self).__init__()

        self.linear1 = nn.Sequential(
            nn.Linear(in_size, 200),
            nn.LeakyReLU(0.3)
        )
        self.linear2 = nn.Sequential(
            nn.Linear(200, 300),
            nn.LeakyReLU(0.3)
        )
        self.linear3 = nn.Sequential(
            nn.Linear(300, 784),
            nn.LeakyReLU(0.3)
        )
        self.linear4 = nn.Sequential(
            nn.Linear(500, 784),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.linear1(x)
        x = self.linear2(x)
        x = self.linear3(x)
        #x = self.linear4(x)
        return x


generator = GeneratorNet(in_size=100)


def noise(size):
    n = Variable(torch.randn(size, 100))
    return n


d_optimizer = optim.Adam(discriminator.parameters(), lr=0.001)
g_optimizer = optim.Adam(generator.parameters(), lr=0.0002)

loss = torch.nn.BCELoss()


def ones_target(size):
    data = Variable(torch.ones(size, 1, 1))
    return data


def zeros_target(size):
    data = Variable(torch.zeros(size, 1, 1))
    return data


def train_discriminator(optimizer, real_data, fake_data):
    N = real_data.size(0)
    optimizer.zero_grad()
    prediction_real = discriminator(real_data)
    error_real = loss(prediction_real, ones_target(N))
    error_real.backward()
    prediction_fake = discriminator(fake_data)
    error_fake = loss(prediction_fake, zeros_target(N))
    error_fake.backward()
    optimizer.step()
    return error_real + error_fake, prediction_real, prediction_fake


def to_img(x):
    return x.reshape(x.size(0), 1, 28, 28)


def to_vec(x):
    return x.reshape(x.size(0), 784)


def train_generator(optimizer, fake_data):
    N = fake_data.size(0)
    optimizer.zero_grad()
    prediction = discriminator(fake_data)
    error = loss(prediction, ones_target(N))
    error.backward()
    optimizer.step()
    return error


def plot_durations(x, y, y_net):
    plt.figure(2)
    plt.clf()
    plt.title('Training...')
    plt.xlabel('x')
    plt.ylabel('y')

    lines = plt.plot(x, y, x, y_net)
    plt.setp(lines[0], linewidth=1, color='g', label='discriminator')
    plt.setp(lines[1], linewidth=1, color='r', label='generator')

    plt.pause(0.0001)
    display.clear_output(wait=True)
    display.display(plt.gcf())


losses_disc = []
losses_gen = []
runs = []
run = 0

test_noise = noise(9)

num_epochs = 10000
for epoch in range(num_epochs):
    disc_error = []
    gen_error = []
    runs.append(epoch)
    for n_batch, (image_batch, label_batch) in enumerate(data_loader):
        N = image_batch.size(0)
        real_data = image_batch

        fake_data = to_img(generator(noise(N)).detach())

        d_error, d_pred_real, d_pred_fake = train_discriminator(d_optimizer, real_data, fake_data)

        fake_data = to_img(generator(noise(N)))
        g_error = train_generator(g_optimizer, fake_data)

        disc_error.append(d_error.detach().numpy())
        gen_error.append(g_error.detach().numpy())
    losses_disc.append(np.mean(disc_error))
    losses_gen.append(np.mean(gen_error))

    plot_durations(runs, losses_disc, losses_gen)

    print('Epoch: ' + str(epoch))

    f, axarr = plt.subplots(3, 3)
    test_images = to_img(generator(test_noise)).detach().numpy()
    '''axarr[0, 0].imshow(to_img(generator(noise(1))).detach().numpy()[0, 0], cmap='gray')
axarr[0, 1].imshow(to_img(generator(noise(1))).detach().numpy()[0, 0], cmap='gray')
axarr[0, 2].imshow(to_img(generator(noise(1))).detach().numpy()[0, 0], cmap='gray')
axarr[1, 0].imshow(to_img(generator(noise(1))).detach().numpy()[0, 0], cmap='gray')
axarr[1, 1].imshow(to_img(generator(noise(1))).detach().numpy()[0, 0], cmap='gray')
axarr[1, 2].imshow(to_img(generator(noise(1))).detach().numpy()[0, 0], cmap='gray')
axarr[2, 0].imshow(to_img(generator(noise(1))).detach().numpy()[0, 0], cmap='gray')
axarr[2, 1].imshow(to_img(generator(noise(1))).detach().numpy()[0, 0], cmap='gray')
axarr[2, 2].imshow(to_img(generator(noise(1))).detach().numpy()[0, 0], cmap='gray')'''
    axarr[0, 0].imshow(test_images[0, 0], cmap='gray')
    axarr[0, 1].imshow(test_images[1, 0], cmap='gray')
    axarr[0, 2].imshow(test_images[2, 0], cmap='gray')
    axarr[1, 0].imshow(test_images[3, 0], cmap='gray')
    axarr[1, 1].imshow(test_images[4, 0], cmap='gray')
    axarr[1, 2].imshow(test_images[5, 0], cmap='gray')
    axarr[2, 0].imshow(test_images[6, 0], cmap='gray')
    axarr[2, 1].imshow(test_images[7, 0], cmap='gray')
    axarr[2, 2].imshow(test_images[8, 0], cmap='gray')
    plt.savefig(str(epoch) + '.png')
