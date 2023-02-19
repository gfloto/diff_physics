import os, sys
import numpy as np
from phi.flow import *
import matplotlib.pyplot as plt
from matplotlib import animation

# save set of matplotlib video of images
def save_video(x, filename):
    ims = []
    fig = plt.figure(figsize=(10, 10))
    for i in range(x.shape[0]):    
        im = plt.imshow(x[i].T, animated=True)
        ims.append([im])
    ani = animation.ArtistAnimation(fig, ims, interval=50, blit=True, repeat_delay=1000)
    ani.save(filename, fps=20)

if __name__ == '__main__':
    x = 100; y = 100
    smoke = CenteredGrid(0, extrapolation.BOUNDARY, x=x, y=y, bounds=Box(x=x, y=y))  # sampled at cell centers
    velocity = StaggeredGrid(0, extrapolation.ZERO, x=x, y=y, bounds=Box(x=x, y=y))  # sampled in staggered form at face centers
    inflow_loc = tensor([(30, 65)], batch('b'), channel(vector='x,y'))
    inflow = 0.6 * CenteredGrid(Sphere(center=inflow_loc, radius=3), extrapolation.BOUNDARY, x=x, y=y, bounds=Box(x=x, y=y))

    x = []
    for i in range(20):
        smoke = advect.mac_cormack(smoke, velocity, dt=1) + inflow
        buoyancy_force = smoke * (0, 0.5) @ velocity
        velocity = advect.semi_lagrangian(velocity, velocity, dt=1) + buoyancy_force
        velocity, _ = fluid.make_incompressible(velocity)

        x.append(smoke.values.numpy('b,x,y'))
    x = np.array(x)
    for i in range(x.shape[1]):
        save_video(x[:,i], f'out_{i}.gif')


