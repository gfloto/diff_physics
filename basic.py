import os, sys
import numpy as np
from phi.flow import *
from phi.physics.diffuse import explicit

from utils import save_video

if __name__ == '__main__':
    x = y = 256
    smoke = CenteredGrid(Noise(scale=x/10), extrapolation.BOUNDARY, x=x, y=y, bounds=Box(x=x, y=y))  # sampled at cell centers
    velocity = StaggeredGrid(0, extrapolation.ZERO, x=x, y=y, bounds=Box(x=x, y=y))  # sampled in staggered form at face centers
    #inflow_loc = tensor([(32, 32)], batch('b'), channel(vector='x,y'))
    #inflow = 0.6 * CenteredGrid(Sphere(center=inflow_loc, radius=10), extrapolation.BOUNDARY, x=x, y=y, bounds=Box(x=x, y=y))

    x = [smoke.values.numpy('b,x,y')]
    for i in range(200):
        smoke = advect.mac_cormack(smoke, velocity, dt=1)
        smoke = explicit(smoke, 0.1, dt=1, substeps=1)

        buoyancy_force = smoke * (0, 0.5) @ velocity
        velocity = advect.semi_lagrangian(velocity, velocity, dt=1) + buoyancy_force
        velocity, _ = fluid.make_incompressible(velocity)

        x.append(smoke.values.numpy('b,x,y'))
    x = np.array(x)
    for i in range(x.shape[1]):
        save_video(x[:,i], f'out_{i}.gif')


