import sys

import torch
import torchvision
from tqdm import tqdm
from sklearn.decomposition import PCA
import util
from histmatch import *
from vgg import Decoder, Encoder

torch.set_grad_enabled(False)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def random_rotation(N):
    """
    Draws random N-dimensional rotation matrix (det = 1, inverse = transpose)

    From https://github.com/scipy/scipy/blob/5ab7426247900db9de856e790b8bea1bd71aec49/scipy/stats/_multivariate.py#L3309
    """
    H = torch.eye(N, device=device)
    D = torch.empty((N,), device=device)
    for n in range(N - 1):
        x = torch.randn(N - n, device=device)
        norm2 = torch.dot(x, x)
        x0 = x[0].item()
        D[n] = torch.sign(x[0]) if x[0] != 0 else 1
        x[0] += D[n] * torch.sqrt(norm2)
        x /= torch.sqrt((norm2 - x0 ** 2 + x[0] ** 2) / 2.0)
        H[:, n:] -= torch.outer(H[:, n:] @ x, x)
    D[-1] = (-1) ** (N - 1) * D[:-1].prod()
    H = (D * H.T).T
    return H


if __name__ == "__main__":
    style = util.load_image("style/marble-small.jpg")
    output = torch.rand(style.shape, device=device)
    PCADimension = [30, 100, 200, 400, 250]
    U = [None]
    S = [None]
    V = [None]
    style_layers = [None]  # add one to index so it works better in next loop
    for layer in range(1, 6):
        with Encoder(layer).to(device) as encoder:
            enc_s = encoder(style).squeeze().permute(1, 2, 0)  # remove batch channel and move channels to last axis
            enc_s = enc_s.reshape(-1, enc_s.shape[2]) # [pixels, channels]
            print(enc_s.shape)
            # [pixels, channels PCA reduces]
            U_s, S_s, V_s = torch.svd(enc_s)
            U.append(U_s)
            S.append(S_s)
            V.append(V_s)
            style_layers.append(torch.matmul(enc_s, V_s[:,:PCADimension[layer-1]])) # Data projected on the principal components

    # multiple resolutions (e.g. each pass can be done for a new resolution ?)
    num_passes = 5
    pbar = tqdm(total=64 + 128 + 256 + 512 + 512, smoothing=1)
    for _ in range(num_passes):
        # PCA goes here
        for layer in range(5, 0, -1):
            with Encoder(layer).to(device) as encoder:
                output_layer = encoder(output).squeeze().permute(1, 2, 0)
                h, w, c = output_layer.shape
                output_layer = output_layer.reshape(-1, c)  # [pixels, channels]
                dataFirstPrincipalComponents = torch.matmul(output_layer, V[layer]) # output image projected on first principal components
                c_pca = PCADimension[layer-1]

            for it in range(int(c_pca / num_passes)):
                rotation = random_rotation(c_pca)
                proj_s = style_layers[layer] @ rotation
                proj_o = dataFirstPrincipalComponents @ rotation

                match_o = hist_match(proj_o, proj_s)

                output_layer = match_o @ rotation.T

                pbar.update(1)

            with Decoder(layer).to(device) as decoder:
                output = torch.matmul(output_layer, V[layer].T)
                output = decoder(output.T.reshape(1, c, h, w))

    torchvision.utils.save_image(torch.cat((style, output)), "output/texture.png")
