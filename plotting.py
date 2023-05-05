__all__ = ['vis_distributions', 'vis_encoding', 'vis_manifold', 'vis_style_manifold', 'vis_reconstruction', 'vis_results', 'visualizes']
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import norm

import torch
import torchvision
import torch.nn.functional as F

from prior import gaussian, gaussian_mixture, swiss_roll


labels_map = {
        0: "0",
        1: "1",
        2: "2",
        3: "3",
        4: "4",
        5: "5",
        6: "6",
        7: "7",
        8: "8",
        9: "9",
    }


def discrete_cmap(N, base_cmap=None):
    base = plt.cm.get_cmap(base_cmap)
    color_list = base(np.linspace(0, 1, N))
    cmap_name = base.name + str(N)
    return base.from_list(cmap_name, color_list, N)


def vis_distributions():
    batch_size = 10000
    n_dim = 2

    def sample(prior_type, use_label_info=True):
        # get samples from prior
        if prior_type == 'mixGaussian':
            z_id = np.random.randint(0, 10, size=[batch_size])
            if use_label_info:
                z = gaussian_mixture(batch_size, n_dim, label_indices=z_id)
            else:
                z = gaussian_mixture(batch_size, n_dim)
        elif prior_type == 'swiss_roll':
            z_id = np.random.randint(0, 10, size=[batch_size])
            if use_label_info:
                z = swiss_roll(batch_size, n_dim, label_indices=z_id)
            else:
                z = swiss_roll(batch_size, n_dim)
        elif prior_type == 'normal':
            z, z_id = gaussian(batch_size, n_dim, use_label_info=use_label_info)
        else:
            raise Exception("[!] There is no option for " + prior_type)

        return z, z_id

    # plot
    plt.rcParams['figure.figsize'] = [16, 9]
    plt.figure()
    fig, axes = plt.subplots(2, 3)
    for i, prior_type in enumerate(['normal', 'mixGaussian', 'swiss_roll']):
        for j, tf in enumerate([False, True]):
            z, z_id = sample(prior_type, use_label_info=tf)

            ax = axes[j, i]
            s = ax.scatter(z[:, 0], z[:, 1], s=25, c=z_id, marker='o', edgecolor='none', cmap=discrete_cmap(10, 'jet'))
            ax.grid(True)
            #ax.set_xlim([-4.5, 4.5])
            #ax.set_ylim([-4.5, 4.5])

            if j == 1:
                ax.set_xlabel(prior_type)
    fig.colorbar(s, ax=axes.ravel().tolist())
    plt.show()


def vis_encoding(model, data, dev, parent=None):
    if parent is None:
        parent = plt.figure()

    x = []
    y = []
    labels = []
    axs = parent.add_subplot(1,1,1)
    axs.axis('equal')
    for dat, label in data:
        dat = dat.to(dev)
        label_idc = np.argmax(label.cpu().detach().numpy(),axis=2)
        enc = model.encode(dat)
        if type(enc) == tuple:
            enc = model.reparam(*enc)
        enc = enc.cpu().detach().numpy()
        x.append([enc[:, 0]])
        y.append([enc[:, 1]])
        labels.append(label_idc)

    s = axs.scatter(y, x, c=labels, s=2, cmap=discrete_cmap(10, 'jet'))
    plt.legend(s.legend_elements()[0],np.unique(labels))


def vis_manifold(model, device, n_row_col=20):
    x = norm.ppf(np.linspace(0.05,0.95,n_row_col),scale=5.0)
    y = norm.ppf(np.linspace(0.05,0.95,n_row_col),scale=5.0)
    x = np.flip(x)

    fig = plt.figure(figsize=(8,8))
    to_imag = torchvision.transforms.ToPILImage()
    for i in range(len(x)):
        for j in range(len(y)):
            fig.add_subplot(n_row_col, n_row_col, i*n_row_col+j+1)

            sample = torch.tensor([[x[i], y[j]]])
            sample = sample.to(torch.float32)
            sample = sample.to(device)
            tens = model.decode(sample)
            plt.imshow(to_imag(tens.reshape(28,28)), cmap="gray")
            plt.axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)


def vis_style_manifold(model, dev, labels, scale=5.0):
    plt.rcParams['figure.figsize'] = [12, 7]
    plt.figure()

    n = len(labels)
    x = norm.ppf(np.linspace(0.05, 0.95, n), scale=scale)
    y = np.random.normal(scale=5.0)

    figure = plt.figure(figsize=(8, 8))
    to_imag = torchvision.transforms.ToPILImage()
    for i, label in enumerate(labels):
        for j in range(n):
            figure.add_subplot(n, n, i * n + j + 1)

            sample = torch.tensor([[x[j], y]])
            sample = sample.to(torch.float32)
            sample = sample.to(dev)
            label_one_hot = F.one_hot(torch.LongTensor([label]), n).float().to(dev)
            tens = model.decode(sample, label_one_hot)

            plt.imshow(to_imag(tens.reshape(28, 28)), cmap="gray")
            plt.axis("off")
    plt.subplots_adjust(wspace=0, hspace=0, left=0, right=1, bottom=0, top=1)
    plt.show()


def vis_reconstruction(model, data, device,
                       parent=None,
                       cols=10,
                       rows=3):

    if parent is None:
        parent = plt.figure()

    imgs = []
    for i in range(1, cols * rows + 1):
        axs = parent.add_subplot(rows + 1, cols, i)

        sample_idx = torch.randint(len(data), size=(1,)).item()
        img, label = data[sample_idx]
        lab_idx = np.argmax(label.cpu().detach().numpy(), axis=1)
        axs.set_title(labels_map[lab_idx[0]])
        axs.axis("off")
        axs.imshow(img, cmap="gray")
        if i > (rows-1)*cols:
            imgs.append(img)

    model.to(device)
    to_tens = torchvision.transforms.ToTensor()
    to_imag = torchvision.transforms.ToPILImage()
    for i in range(cols):
        axs = parent.add_subplot(rows + 1, cols, i + 1 + cols * rows)

        inp_tens = to_tens(imgs[i])
        inp_tens = inp_tens.to(device).to(torch.float32)
        outp_tens = model(inp_tens)
        outp_imag = to_imag(outp_tens.reshape(28, 28))
        axs.axis("off")
        axs.imshow(outp_imag, cmap="gray")


def vis_results(model, train, test, device):
    fig = plt.figure(constrained_layout=True, figsize=(12, 6))  # , width_ratios=[2, 1, 2])

    subfigs = fig.subfigures(1, 2, wspace=0.05)
    vis_reconstruction(model, test, device, subfigs[0], cols=6)  # subfigs[0][0])
    vis_encoding(model, train, device, subfigs[1])  # subfigs[0][1])
    plt.show()


def vis_training(**kwargs):
    plt.figure(constrained_layout=True, figsize=(12, 6))
    for key, losses in kwargs.items():
        plt.plot(losses, label=key)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()


def visualize(model, test, vis_test, device):
    vis_results(model, test, vis_test, device)
    vis_manifold(model, device)


def load_and_visualize(model,path,train,test,device):
    model.load_state_dict(torch.load(path))
    model.eval()
    visualize(model,train,test,device)
    
