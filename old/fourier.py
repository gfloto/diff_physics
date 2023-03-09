import sys, os
import numpy as np
from einops import rearrange
import matplotlib.pyplot as plt

from utils import save_video 

if __name__ == '__main__':
    # load data
    x = np.load('smoke.npy')
    print(x.shape)

    # get fourier features at each time step
    phase = np.fft.rfft2(x, axes=(1,2))
    x_out = np.fft.irfftn(phase, axes=(1,2))
    #x_out = np.real(x_out)
    print(phase.shape)

    # dummy plotting
    for i in range(phase.shape[0]):
       pass 

    # print difference
    print(x[0,:5,:5])
    print(x_out[0,:5,:5])

    # show video
    img = np.stack((x, x_out))
    img = rearrange(img, 'b t h w -> t h (b w)')
    save_video(img, 'diff.gif')
