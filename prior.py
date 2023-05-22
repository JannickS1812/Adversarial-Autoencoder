import numpy as np
from math import sin, cos, sqrt

import torch
import torch.nn.functional as F

def sample(prior_type, n_samples, use_label_info=True):
    # get samples from prior
    if prior_type == 'mixture':
        z_id = np.random.randint(0, 10, size=[n_samples])
        if use_label_info:
            z = gaussian_mixture(n_samples, 2, label_indices=z_id)
        else:
            z = gaussian_mixture(n_samples, 2)
    elif prior_type == 'swiss_roll':
        z_id = np.random.randint(0, 10, size=[n_samples])
        if use_label_info:
            z = swiss_roll(n_samples, 2, label_indices=z_id)
        else:
            z = swiss_roll(n_samples, 2)
    elif prior_type == 'normal':
        z, z_id = gaussian(n_samples, 2, use_label_info=use_label_info)
    else:
        raise Exception("[!] There is no option for " + prior_type)

    return z, z_id


def uniform(batch_size, n_dim, n_labels=10, minv=-1, maxv=1, label_indices=None):
    if label_indices is not None:
        if n_dim != 2 or n_labels != 10:
            raise Exception("n_dim must be 2 and n_labels must be 10.")

        def sample(label, n_labels):
            num = int(np.ceil(np.sqrt(n_labels)))
            size = (maxv-minv)*1.0/num
            x, y = np.random.uniform(-size/2, size/2, (2,))
            i = label / num
            j = label % num
            x += j*size+minv+0.5*size
            y += i*size+minv+0.5*size
            return np.array([x, y]).reshape((2,))

        z = np.empty((batch_size, n_dim), dtype=np.float32)
        for batch in range(batch_size):
            for zi in range((int)(n_dim/2)):
                    z[batch, zi*2:zi*2+2] = sample(label_indices[batch], n_labels)
    else:
        z = np.random.uniform(minv, maxv, (batch_size, n_dim)).astype(np.float32)
    return z


def gaussian(batch_size, n_dim, mean=0, std=5, n_labels=10, use_label_info=False):
    if use_label_info:
        if n_dim != 2 or n_labels != 10:
            raise Exception("n_dim must be 2 and n_labels must be 10.")

        def sample(n_labels):
            x, y = np.random.normal(mean, std, (2,))
            angle = np.angle((x-mean) + 1j*(y-mean), deg=True)
            dist = np.sqrt((x-mean)**2+(y-mean)**2)

            # label 0
            if dist <1.0:
                label = 0
            else:
                label = ((int)((n_labels-1)*angle))//360

                if label<0:
                    label+=n_labels-1

                label += 1

            return np.array([x, y]).reshape((2,)), label

        z = np.empty((batch_size, n_dim), dtype=np.float32)
        z_id = np.empty((batch_size), dtype=np.int32)
        for batch in range(batch_size):
            for zi in range((int)(n_dim/2)):
                    a_sample, a_label = sample(n_labels)
                    z[batch, zi*2:zi*2+2] = a_sample
                    z_id[batch] = a_label
        return z, z_id
    else:
        z = np.random.normal(mean, std, (batch_size, n_dim)).astype(np.float32)
        return z, np.random.randint(0,10,size=[batch_size])


def gaussian_mixture(batch_size, n_dim=2, n_labels=10, x_var=3, y_var=0.5, label_indices=None):
    if n_dim != 2:
        raise Exception("n_dim must be 2.")

    def sample(x, y, label, n_labels):
        shift = 10.0
        r = 2.0 * np.pi / float(n_labels) * float(label)
        new_x = x * cos(r) - y * sin(r)
        new_y = x * sin(r) + y * cos(r)
        new_x += shift * cos(r)
        new_y += shift * sin(r)
        return np.array([new_x, new_y]).reshape((2,))

    x = np.random.normal(0, x_var, (batch_size, (int)(n_dim/2)))
    y = np.random.normal(0, y_var, (batch_size, (int)(n_dim/2)))
    z = np.empty((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range((int)(n_dim/2)):
            if label_indices is not None:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], label_indices[batch], n_labels)
            else:
                z[batch, zi*2:zi*2+2] = sample(x[batch, zi], y[batch, zi], np.random.randint(0, n_labels), n_labels)

    return z


def swiss_roll(batch_size, n_dim=2, n_labels=10, label_indices=None):
    if n_dim != 2:
        raise Exception("n_dim must be 2.")

    def sample(label, n_labels):
        uni = np.random.uniform(0.0, 1.0) / float(n_labels) + float(label) / float(n_labels)
        r = sqrt(uni) * 3.0 * 4 # Test
        rad = np.pi * 4.0 * sqrt(uni)
        x = r * cos(rad)
        y = r * sin(rad)
        return np.array([x, y]).reshape((2,))

    z = np.zeros((batch_size, n_dim), dtype=np.float32)
    for batch in range(batch_size):
        for zi in range((int)(n_dim/2)):
            if label_indices is not None:
                z[batch, zi*2:zi*2+2] = sample(label_indices[batch], n_labels)
            else:
                z[batch, zi*2:zi*2+2] = sample(np.random.randint(0, n_labels), n_labels)
    return z


def categorical(batch_size, n_labels):
    label_indices = torch.LongTensor(np.random.randint(0, n_labels, size=[batch_size]))
    return F.one_hot(label_indices)


