import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from scipy.io import loadmat

def read_data(path):
    path = Path(path)
    if path.suffix == '.mat':
        d = loadmat(path)
    elif path.suffix == '.npz':
        with np.load(path) as f:
            d = {k: f[k] for k in f.files}
    else:
        raise ValueError('expected .mat or .npz file')
    return d

def metrics(d):
    pos = np.asarray(d['pos_s'])
    vel = np.asarray(d['vel_s'])
    if pos.ndim == 3: pos = pos[..., None]
    if vel.ndim == 3: vel = vel[..., None]
    if pos.shape != vel.shape: raise ValueError('Position and velocity shapes differ')
    norms = np.linalg.norm(vel, axis = 1, keepdims = True)
    with np.errstate(invalid = 'ignore', divide = 'ignore'):
        headings = vel/norms
    polar = np.linalg.norm(np.mean(headings, axis = 0), axis = 0).ravel()

    centered = pos - pos.mean(axis = 0, keepdims = True)
    cohesion = np.linalg.norm(centered, axis = 1).mean(axis = 0).ravel()
    
    meanvel = vel.mean(axis = 0)  
    theta = np.arctan2(meanvel[1], meanvel[0])
    x, y = centered[:, 0], centered[:, 1]
    xp = -np.sin(theta)[None, ...] * x + np.cos(theta)[None, ...] * y
    yp = -np.cos(theta)[None, ...] * x - np.sin(theta)[None, ...] * y
    rx = np.max(xp, axis = 0) - np.min(xp, axis = 0)
    ry = np.max(yp, axis = 0) - np.min(yp, axis = 0)
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        elong = (ry/rx).ravel()
    return {'polarization': polar, 'cohesion': cohesion, 'elongation': elong}

def make_plots(m, outdir):
    outdir = Path(outdir); outdir.mkdir(parents = True, exist_ok = True)
    configs = {
        'polarization': (np.arange(0, 1.00001, 0.04), 'Group polarisation', (0, 1.02)),
        'cohesion': (None, 'Cohesion', None),
        'elongation': (np.arange(0.1, 7.00001, 0.2), 'Elongation', (0, 7)),}
    for k, (edges, label, xlim) in configs.items():
        x = m[k][np.isfinite(m[k])]
        if edges is None:
            edges = np.arange(np.min(x), np.max(x) + 0.21, .2)
        fig, ax = plt.subplots(figsize = (6.5, 4.5))
        dens, edges = np.histogram(x, bins = edges, density = True)
        ax.plot((edges[:-1]+edges[1:])/2, dens, linewidth = 2)
        ax.set_xlabel(label); ax.set_ylabel('PDF')
        if xlim: ax.set_xlim(*xlim)
        ax.grid(alpha = 0.2)
        fig.tight_layout(); fig.savefig(outdir/f'{k}.png', dpi = 170)
        plt.close(fig)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help = 'Path to .mat or .npz')
    parser.add_argument('--output-dir', default = 'figures')
    args = parser.parse_args()
    d = read_data(args.data)
    m = metrics(d)
    for k,v in m.items():
        v = v[np.isfinite(v)]
        print(f'{k}: n = {v.size}, mean = {v.mean():.4f}, median = {np.median(v):.4f}')
    make_plots(m,args.output_dir)
    print(f'Figures saved to {args.output_dir}')
