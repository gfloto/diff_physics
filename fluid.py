import os, sys
import numpy as np
from einops import rearrange
from phi.flow import *

from utils import save_video

if __name__ == '__main__':
    x = y = 128
    press = CenteredGrid(0, extrapolation.ZERO, x=x, y=y, bounds=Box(x=x, y=y))  # sampled at cell centers
    velocity = StaggeredGrid(Noise(x/10), extrapolation.BOUNDARY, x=x, y=y, bounds=Box(x=x, y=y))  # sampled in staggered form at face centers
    #velocity = StaggeredGrid(0, extrapolation.ZERO, x=x, y=y, bounds=Box(x=x, y=y))  # sampled in staggered form at face centers

    inflow_loc = tensor([(x//2, y//4)], batch('b'), channel(vector='x,y'))
    inflow = 0.6 * CenteredGrid(Sphere(center=inflow_loc, radius=x//8), extrapolation.BOUNDARY, x=x, y=y, bounds=Box(x=x, y=y))

    values = []
    for i in range(200):
        #velocity = advect.mac_cormack(press, velocity, dt=1) + inflow
        #force = smoke * (0, 0.5) @ velocity
        #velocity += inflow
        velocity = advect.semi_lagrangian(velocity, velocity, dt=1)
        velocity, press = fluid.make_incompressible(velocity)
        if i > 0: 
            print(velocity)
            vx = velocity['x'].values.numpy('b,x,y')[0,1:,:]
            vy = velocity['y'].values.numpy('b,x,y')[0,:,1:]
            pr = press.values.numpy('b,x,y')[0]

            out = np.stack((pr, vx, vy))
            out = rearrange(out, 'b h w -> h (b w)')
            values.append(out)

    values = np.array(values)
    save_video(values, 'p_out.gif')
