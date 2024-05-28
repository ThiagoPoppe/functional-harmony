import torch
import numpy as np
import torch.nn.functional as F

from scipy.ndimage import gaussian_filter1d
from librosa.segment import recurrence_matrix


def compute_chroma(pianoroll):
    """
    Computes a chromagram based on an input pianoroll of shape (batch_size, 12, seq_length).
    Returns a tensor of shape (batch_size, 12, seq_length) representing a chromagram normalized energy.
    """

    if type(pianoroll) == np.ndarray:
        pianoroll = torch.from_numpy(pianoroll)

    pianoroll[pianoroll == 0] = -torch.inf
    chroma = F.softmax(pianoroll, dim=1)

    return chroma


def compute_tonnetz(chroma, filtering=True, sigma=8):
    """
    Computes a tonnetz representation from an input chromagram of shape (batch_size, 12, seq_length).
    You can also apply a gaussian filter by setting filtering to True and defining a sigma value (default=8)

    Returns a tensor of shape (batch_size, 6, seq_length) representing the tonnetz representation feature.
    """
    # Defining "phi" transformation matrix
    r1, r2, r3 = 1, 1, 0.5
    phi_0 = r1 * torch.sin(torch.tensor(range(12)) * 7 * torch.pi / 6)
    phi_1 = r1 * torch.cos(torch.tensor(range(12)) * 7 * torch.pi / 6)
    phi_2 = r2 * torch.sin(torch.tensor(range(12)) * 3 * torch.pi / 2)
    phi_3 = r2 * torch.cos(torch.tensor(range(12)) * 3 * torch.pi / 2)
    phi_4 = r3 * torch.sin(torch.tensor(range(12)) * 2 * torch.pi / 3)
    phi_5 = r3 * torch.cos(torch.tensor(range(12)) * 2 * torch.pi / 3)
    phi = torch.cat([phi_0, phi_1, phi_2, phi_3, phi_4, phi_5]).reshape(-1, 6, 12)
    phi = phi.mT  # shape will be (batch_size, 12, 6)

    tonnetz = (chroma.mT @ phi).mT  # shape will be (batch_size, 6, seq_length)
    if filtering:
        tonnetz = gaussian_filter1d(tonnetz, sigma=sigma, axis=-1)
        tonnetz = torch.from_numpy(tonnetz)

    return tonnetz.float()


def compute_harmony_similarity(tonnetz):
    """
    Computes a harmony similarity matrix, i.e. a square matrix, where each pair (i, j) represents the harmony similarity between two frames.
    As input, you must provide the tonnetz feature representation of the audio/pianoroll of shape (batch_size, 6, seq_length).

    Returns a tensor of shape (batch_size, seq_length, seq_length) representing the harmony similarity of the input for each batch.
    """
    similarity_matrices = []
    batch_size = tonnetz.size(0)

    for i in range(batch_size):
        ssim = recurrence_matrix(tonnetz[i], mode='affinity')
        ssim = torch.from_numpy(ssim).float()
        similarity_matrices.append(ssim)

    similarity_matrices = torch.stack(similarity_matrices)
    return similarity_matrices
