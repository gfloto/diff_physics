import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

# save set of matplotlib video of images
def save_video(x, filename):
    ims = []
    fig = plt.figure(figsize=(10, 10))
    for i in range(x.shape[0]):    
        #x_plt = np.flip(x[i].T, axis=(0,1))
        x_plt = x[i]
        im = plt.imshow(x_plt, animated=True, cmap='inferno', vmin=0, vmax=1)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save(filename, fps=20)
