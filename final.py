import sys, os
import time
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from einops import rearrange

class DataStream:
    def __init__(self, b=3, G=0.25, dt_min=1e-3):
        self.b = b
        self.G = G
        self.dt = dt_min
        self.dt_min = dt_min
        self.p = np.random.randn(self.b, 3)
        # normalize positions
        m = np.mean(self.p, axis=0)
        std = np.std(self.p, axis=0)
        self.p = (self.p - m)# / std 
        self.v = 0.1*np.random.randn(self.b, 3)

        self.chain = []
        self.last = self.p

    def stream(self, freq=20):
        while True:
            count = 0
            #while count < 1/freq:
            for _ in range(150):
                d = dist(self.p)
                norm_3 = np.linalg.norm(d, axis=2)**3 + np.eye(d.shape[0], d.shape[1])
                norm_3 = np.tile(np.expand_dims(norm_3, axis=-1), d.shape[-1])
                d_norm = self.G * d / norm_3
                a = -np.sum(d_norm, axis=1)

                self.dt = min(self.dt_min, np.min(norm_3)/10)
                self.dt = max(self.dt, 1e-5)
                count += self.dt

                # update equations
                self.v += a * self.dt
                self.p += self.v * self.dt
                self.p -= np.mean(self.p, axis=0)

                # tail for vis
                self.chain.append(self.p - self.last)
                if len(self.chain) > 100: self.chain.pop(0)
                chain = np.array(self.chain)

            yield self.p.T, rearrange(chain, 't b d -> t d b'), self.dt

def dist(x):
    x_ = np.expand_dims(x, axis=0)
    y1 = np.vstack([x_]*x.shape[0])
    y2 = rearrange(y1, 'b1 b2 d -> b2 b1 d')
    return y2 - y1

def update_graph(num):
    p, chain, dt = next(data_stream())
    graph._offsets3d = (p[0], p[1], p[2])
    title.set_text(f'dt = {dt:.3E}')
    #line, = ax.plot(chain[:,0,0], chain[:,1,0], chain[:,0,2], alpha=0.1)

if __name__ == '__main__':
    np.random.seed(0)
    data_stream = DataStream().stream

    lim = 1.5
    fig = plt.figure(figsize=(6,6))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)
    ax.set_zlim(-lim, lim)

    p, chain, dt = next(data_stream())
    graph = ax.scatter(p[0], p[1], p[2])
    title = ax.set_title(f'{dt:.3E}')
    #line, = ax.plot3D(chain[:,0,0], chain[:,1,0], chain[:,0,2], alpha=0.1)

    ani = animation.FuncAnimation(fig, update_graph, interval=40, blit=False)

    plt.show()
