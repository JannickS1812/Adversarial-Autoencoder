__all__ = ['vis_distributions', 'vis_encoding', 'vis_manifold', 'vis_style_manifold', 'vis_reconstruction', 'vis_results', 'visualizes']

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, Slider
from matplotlib import rcParams
from scipy.stats import norm
from scipy.special import comb

import torch
import torchvision
import torch.nn.functional as F

from .prior import gaussian, gaussian_mixture, swiss_roll

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


class Interpolater:
    def __init__(self, model, data, dev, label=None):

        self.model = model
        self.dev = dev
        self.label = label
        self.__interpolate = label is None

        embeddings = []
        labels = []
        for i, (dat, label) in enumerate(data):
            label_idc = np.argmax(label.cpu().detach().numpy(), axis=2)
            enc = model.encode(dat.to(dev))
            if type(enc) == tuple:
                enc = model.reparam(*enc)
            embeddings.extend(list(enc.cpu().detach().numpy()))
            labels.append(label_idc)
        embeddings = np.array(embeddings)
        assert embeddings.shape[1] == 2
        self.embeddings = embeddings

        self.rc_dot = None
        self.lc_dot = None
        self.references = []

        self.fig, self.axes = plt.subplots(1, 2, figsize=(9, 6))
        plt.subplots_adjust(bottom=0.25)
        self.axes[0].axis('equal')
        self.axes[0].scatter(embeddings[:, 0], embeddings[:, 1], c=labels, s=2, cmap=discrete_cmap(10, 'jet'))
        self.axes[1].axis('off')

        # clear button
        ax_btn = self.fig.add_axes([0.8, 0.095, 0.1, 0.04])
        b = Button(ax_btn, 'Clear')
        b.on_clicked(self.__on_press)

        # interpolation slider
        ax_slider = self.fig.add_axes([0.1, 0.1, 0.6, 0.03])
        self.slider = Slider(
            ax=ax_slider,
            label='T [0-1]',
            valmin=0,
            valmax=1,
            valinit=.5,
        )
        self.slider.on_changed(self.__on_slide)

        # add points for curve via click event
        self.fig.canvas.mpl_connect('button_press_event', self.__on_click)
        plt.show()

    def __on_click(self, event):
        if event.inaxes != self.axes[0]:
            return
        if event.button == 1 and self.__interpolate:
            self.lc_dot = np.array([event.xdata, event.ydata])
        elif event.button == 3:
            self.rc_dot = np.array([event.xdata, event.ydata])
        self.__update()

    def __on_press(self, _):
        self.rc_dot = None
        self.lc_dot = None
        self.__update()

    def __on_slide(self, _):
        self.__update()

    def __update(self):
        for r in self.references:
            r.remove()
        self.references = []

        if self.rc_dot is not None and (self.lc_dot is not None or not self.__interpolate):
            if self.__interpolate:
                self.references.append(self.axes[0].plot([self.rc_dot[0], self.lc_dot[0]],
                                                         [self.rc_dot[1], self.lc_dot[1]],
                                                         color='k',
                                                         linewidth=2)[0])

                curr_point = self.slider.val * self.rc_dot + (1 - self.slider.val) * self.lc_dot
                self.references.append(self.axes[0].scatter(curr_point[0],
                                                            curr_point[1],
                                                            color='r',
                                                            edgecolor='k',
                                                            s=2.5 * rcParams['lines.markersize'] ** 2,
                                                            zorder=10))
                img = self.model.decode(torch.Tensor(curr_point).to(self.dev)).reshape(28, 28)
            else:
                curr_point = self.rc_dot
                label = F.one_hot(self.label, 10).to(self.dev)
                img = self.model.decode(torch.Tensor(curr_point).to(self.dev), label).reshape(28, 28)

            to_imag = torchvision.transforms.ToPILImage()
            self.references.append(self.axes[1].imshow(to_imag(img), cmap="gray"))

        if self.rc_dot is not None:
            self.references.append(self.axes[0].scatter(self.rc_dot[0],
                                                        self.rc_dot[1],
                                                        color='k'))

        if self.lc_dot is not None:
            self.references.append(self.axes[0].scatter(self.lc_dot[0],
                                                        self.lc_dot[1],
                                                        color='k'))

        plt.draw()

    def __bezier(self, control_points, t=None):
        """Evaluates a Bezier curve defined by control_points at points t."""
        if t is None:
            t = np.linspace(0, 1, 200)
        return sum([np.outer(self.__bernstein_poly(i, control_points.shape[1], t), x) for i, x in enumerate(X)])

    def __bernstein_poly(self, i, N, t):
        return comb(N, i) * t ** i * (1. - t) ** (N - i)


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
            # ax.set_xlim([-4.5, 4.5])
            # ax.set_ylim([-4.5, 4.5])

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
    axs = parent.add_subplot(1, 1, 1)
    axs.axis('equal')
    for dat, label in data:
        dat = dat.to(dev)
        label_idc = np.argmax(label.cpu().detach().numpy(), axis=2)
        enc = model.encode(dat)
        if type(enc) == tuple:
            enc = model.reparam(*enc)
        enc = enc.cpu().detach().numpy()
        x.append([enc[:, 0]])
        y.append([enc[:, 1]])
        labels.append(label_idc)

    s = axs.scatter(y, x, c=labels, s=2, cmap=discrete_cmap(10, 'jet'))
    plt.legend(s.legend_elements()[0], np.unique(labels))


def vis_manifold(model, device, n_row_col=20):
    x = norm.ppf(np.linspace(0.05, 0.95, n_row_col), scale=5.0)
    y = norm.ppf(np.linspace(0.05, 0.95, n_row_col), scale=5.0)
    x = np.flip(x)

    fig = plt.figure(figsize=(8, 8))
    to_imag = torchvision.transforms.ToPILImage()
    for i in range(len(x)):
        for j in range(len(y)):
            fig.add_subplot(n_row_col, n_row_col, i * n_row_col + j + 1)

            sample = torch.tensor([[x[i], y[j]]])
            sample = sample.to(torch.float32)
            sample = sample.to(device)
            tens = model.decode(sample)
            plt.imshow(to_imag(tens.reshape(28, 28)), cmap="gray")
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
        if i > (rows - 1) * cols:
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
    plt.figure(constrained_layout=True, figsize=(8, 4))
    for key, losses in kwargs.items():
        plt.plot(losses, label=key)
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()


def visualize(model, test, vis_test, device):
    vis_results(model, test, vis_test, device)
    vis_manifold(model, device)


def load_and_visualize(model, path, train, test, device):
    model.load_state_dict(torch.load(path))
    model.eval()
    visualize(model, train, test, device)

