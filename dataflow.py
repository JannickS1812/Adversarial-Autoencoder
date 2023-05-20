import torch
import torchvision
import torch.nn.functional as F


def load_MNIST():
    transform = torchvision.transforms.Compose(
        [
            torchvision.transforms.ToTensor()
        ]
    )

    transform_alt = torchvision.transforms.Compose(
        [
            torchvision.transforms.PILToTensor(),
            # torchvision.transforms.Lambda(lbd),
            torchvision.transforms.ToPILImage()
        ]
    )

    target_transform = torchvision.transforms.Compose(
        [
            lambda x: torch.LongTensor([x]),  # or just torch.tensor
            lambda x: F.one_hot(x, 10)
        ]
    )

    mnist_train = torchvision.datasets.MNIST(
        'MNIST', train=True,
        download=True, transform=transform, target_transform=target_transform)
    mnist_test = torchvision.datasets.MNIST(
        'MNIST',
        train=False, download=True, transform=transform, target_transform=target_transform)

    mnist_vis_test = torchvision.datasets.MNIST(
        'C:\\Git\\UNI\\39-M-Inf-VML Vertiefung Maschinelles Lernen\\Exercise Adversial Autoencoders\\MNIST',
        train=False, download=True, transform=transform_alt, target_transform=target_transform)
    mnist_vis_train = torchvision.datasets.MNIST(
        'C:\\Git\\UNI\\39-M-Inf-VML Vertiefung Maschinelles Lernen\\Exercise Adversial Autoencoders\\MNIST', train=True,
        download=True, transform=transform_alt, target_transform=target_transform)

    return mnist_train, mnist_test, mnist_vis_test, mnist_vis_train



