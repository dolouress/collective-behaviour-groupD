# dog figures
import argparse
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
from analysis import read_data

def signed_angle(a, b):
    cross = a[..., 0] * b[..., 1] - a[..., 1] * b[..., 0]
    dot = np.sum(a * b, axis = -1)
    return np.degrees(np.arctan2(cross, dot))

def calculate_dog_metrics(data):
    sheep = np.asarray(data['pos_s'], dtype = float)
    vel_s = np.asarray(data['vel_s'], dtype = float)
    dog = np.asarray(data['pos_d'], dtype = float)
    vel_d = np.asarray(data['vel_d'], dtype = float)
    if sheep.ndim == 3:
        sheep = sheep[..., None]
        vel_s = vel_s[..., None]
    if dog.ndim == 2:
        dog = dog[..., None]
        vel_d = vel_d[..., None]
    if sheep.shape != vel_s.shape or dog.shape != vel_d.shape:
        raise ValueError('Position and velocity array shapes must agree')

    s = np.transpose(sheep, (2, 3, 0, 1))
    vs = np.transpose(vel_s, (2, 3, 0, 1))
    d = np.transpose(dog, (0, 2, 1))
    vd = np.transpose(vel_d, (0, 2, 1))
    center = s.mean(axis = 2)
    mean_heading = vs.mean(axis = 2)
    theta = np.arctan2(mean_heading[..., 1], mean_heading[..., 0])

    dx = d[..., 0] - center[..., 0]
    dy = d[..., 1] - center[..., 1]
    dog_lateral = np.sin(theta) * dx - np.cos(theta) * dy
    dog_longitudinal = np.cos(theta) * dx + np.sin(theta) * dy
    sx = s[..., 0] - center[..., 0, None]
    sy = s[..., 1] - center[..., 1, None]
    sheep_longitudinal = np.cos(theta)[..., None] * sx + np.sin(theta)[..., None] * sy
    rear_sheep = sheep_longitudinal.min(axis=2)
    dog_to_rear = rear_sheep - dog_longitudinal

    a = mean_heading[1:]
    b = vd[1:]
    bary_to_dog = d[1:] - center[1:]
    dog_to_bary = -bary_to_dog
    phi_bd = signed_angle(a, b)
    psi_bd = signed_angle(a, bary_to_dog)
    psi_db = signed_angle(b, dog_to_bary)
    return {
        'relative_dog_rear': dog_to_rear.ravel(),
        'dog_lateral': dog_lateral.ravel(),
        'psi_BD': psi_bd.ravel(),
        'psi_DB': psi_db.ravel(),
        'phi_BD': phi_bd.ravel(),
    }

def pdf_line(ax, values, edges, label = None, color = None, style = '-'):
    finite = values[np.isfinite(values)]
    if not finite.size:
        raise ValueError('No finite values available to plot')
    counts, bins = np.histogram(finite, bins = edges, density = True)
    centers = (bins[1:] + bins[:-1])/2
    ax.plot(centers, counts, style, label = label, color = color, linewidth = 2)
    return finite

def save_figures(metrics, outdir):
    outdir = Path(outdir)
    outdir.mkdir(parents = True, exist_ok = True)

    fig, ax = plt.subplots(figsize = (7, 4.8))
    pdf_line(ax, metrics['relative_dog_rear'], np.arange(-10, 10.001, 0.5), color = 'green')
    ax.set(xlabel = 'Relative distance: dog to rear sheep', ylabel = 'PDF', xlim = (-10, 10),
           title = 'Figure 6d - dog to rear sheep')
    ax.grid(alpha = 0.2)
    fig.tight_layout()
    fig.savefig(outdir/'figure_6d_dog_rear.png', dpi = 180)
    plt.close(fig)

    fig, ax = plt.subplots(figsize = (7, 4.8))
    pdf_line(ax, metrics['dog_lateral'], np.arange(-10, 10.001, 0.5), color = 'orange')
    ax.set(xlabel = 'Dog lateral movement', ylabel = 'PDF', xlim = (-10, 10), title = 'Figure 6e - dog lateral displacement')
    ax.grid(alpha=0.2)
    fig.tight_layout()
    fig.savefig(outdir/'figure_6e_dog_lateral.png', dpi = 180)
    plt.close(fig)

    edges = np.linspace(-180, 180, 50)
    fig, ax = plt.subplots(figsize=(7, 4.8))
    pdf_line(ax, metrics['psi_BD'], edges, r'$\psi_{BD}$', 'blue')
    pdf_line(ax, metrics['psi_DB'], edges, r'$\psi_{DB}$', 'red')
    pdf_line(ax, metrics['phi_BD'], edges, r'$\phi_{BD}$', '#0096FF', '--')
    ax.set(xlabel = 'Angle (degrees)', ylabel = 'PDF', xlim = (-180, 180), title = 'Figure 6f - relative viewing and heading angles')
    ax.legend()
    ax.grid(alpha = 0.2)
    fig.tight_layout()
    fig.savefig(outdir/'figure_6f_angles.png', dpi = 180)
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('data', help = 'MATLAB .mat or Python .npz data file')
    parser.add_argument('--output-dir', default = 'figures_dog')
    args = parser.parse_args()
    metrics = calculate_dog_metrics(read_data(args.data))
    for name, values in metrics.items():
        valid = values[np.isfinite(values)]
        print(f'{name}: n = {len(valid)}, mean = {valid.mean():.4f}, median = {np.median(valid):.4f}')
    save_figures(metrics, args.output_dir)
    print('Figures saved in:', args.output_dir)

if __name__ == '__main__':
    main()