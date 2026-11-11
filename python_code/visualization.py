# visualization code
from __future__ import annotations
import argparse
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
import numpy as np
from pathlib import Path

IMAGES_DIR = Path(__file__).resolve().parent/"images"
SHEEP_IMAGE = IMAGES_DIR/"sheep.png"
DOG_IMAGE = IMAGES_DIR/"dog.png"

def load_trajectory(path: Path, run: int):
    with np.load(path, allow_pickle = False) as data:
        sheep = np.asarray(data["pos_s"], dtype = float)
        dog = np.asarray(data["pos_d"], dtype = float)
        if sheep.ndim == 3:
            sheep = sheep[..., None]
        if dog.ndim == 2:
            dog = dog[..., None]
        if sheep.ndim != 4 or sheep.shape[1] != 2 or dog.ndim != 3 or dog.shape[1] != 2:
            raise ValueError("Unexpected shapes. Expected sheep (N,2,T,R), dog (T,2,R).")
        if sheep.shape[2] != dog.shape[0] or sheep.shape[3] != dog.shape[2]:
            raise ValueError("Sheep and dog time/run dimensions do not match.")
        if not 0 <= run < sheep.shape[3]:
            raise ValueError(f"Run index must be between 0 and {sheep.shape[3]-1}.")
        s = np.transpose(sheep[:, :, :, run], (2, 0, 1)).copy()
        d = dog[:, :, run].copy()
        modes = None
        keys = ("collect_t", "drive_t", "force_slow")
        if all(k in data for k in keys):
            indicator = []
            for key in keys:
                a = np.asarray(data[key])
                if a.ndim == 2:
                    a = a[:, run]
                elif a.ndim != 1:
                    indicator = []
                    break
                indicator.append(a)
            if len(indicator) == 3 and all(len(a) == len(d) for a in indicator):
                modes = []
                for i in range(len(d)):
                    active = [name for name, a in zip(("Collecting", "Driving", "Slowing down"), indicator)
                              if np.isfinite(a[i])]
                    modes.append(active[0] if len(active) == 1 else "—")
    if not np.isfinite(s).all() or not np.isfinite(d).all():
        raise ValueError("Trajectory includes non-finite coordinates.")
    return s, d, modes

def create_animation(path: Path, run: int, trail: int, interval: int,
                     camera: str, show_icons: bool, repeat: bool):
    s, d, modes = load_trajectory(path, run)
    T, n, _ = s.shape
    center = s.mean(axis = 1)
    fig, ax = plt.subplots(figsize = (8.5, 7.5))
    fig.subplots_adjust(top = 0.91, bottom = 0.10, left = 0.10, right = 0.97)
    ax.set_aspect("equal", adjustable = "box")
    ax.grid(alpha = 0.18, zorder = 0)
    ax.set_xlabel("X position")
    ax.set_ylabel("Y position")
    ax.set_title("Agent-based sheep herding", fontsize = 15, weight = "bold", pad = 14)

    sheep_trails = [ax.plot([], [], color = "#79acd0", lw = 1.15, alpha = 0.50, zorder = 2)[0]
                    for _ in range(n)]
    dog_trail, = ax.plot([], [], color = "#d84040", lw = 2.2, alpha = 0.85, zorder = 3)
    centre_marker, = ax.plot([], [], marker = "+", ms = 9, mew = 1.7,color = "#30425b", ls = "None", zorder = 5)

    sheep_icons, dog_icon = [], None
    sheep_png = SHEEP_IMAGE
    dog_png = DOG_IMAGE
    if show_icons and sheep_png.exists() and dog_png.exists():
        sheep_image = plt.imread(sheep_png)
        dog_image = plt.imread(dog_png)
        for _ in range(n):
            item = AnnotationBbox(OffsetImage(sheep_image, zoom = 0.12), (0, 0), frameon = False, zorder = 6, annotation_clip = False)
            ax.add_artist(item)
            sheep_icons.append(item)
        dog_icon = AnnotationBbox(OffsetImage(dog_image, zoom = 0.075), (0, 0), frameon = False, zorder = 7, annotation_clip = False)
        ax.add_artist(dog_icon)
    else:
        sheep_scatter = ax.scatter([], [], s = 55, c = "#73a5cb", edgecolors = "white", lw = 0.6, zorder = 6, label = "Sheep")
        dog_scatter = ax.scatter([], [], s = 120, c = "#d84040", edgecolors = "white",lw = 0.8, zorder = 7, label = "Dog")
        ax.legend(loc = "upper left")

    info = ax.text(0.02, 0.98, "", transform = ax.transAxes, va = "top", ha = "left", fontsize = 10, bbox = dict(facecolor = "white", alpha = 0.90,
            edgecolor = "none", boxstyle = "round,pad = 0.5"), zorder = 10)

    if camera == "fixed":
        all_points = np.concatenate([s.reshape(-1, 2), d], axis = 0)
        xy_min = all_points.min(axis = 0)
        xy_max = all_points.max(axis = 0)
        mid = (xy_min + xy_max)/2
        half = max(float(np.max(xy_max-xy_min))/2+5, 15)
        ax.set_xlim(mid[0] - half, mid[0] + half)
        ax.set_ylim(mid[1] - half, mid[1] + half)
    else:
        cam_centers = np.empty_like(center)
        cam_centers[0] = center[0]
        half_widths = np.empty(T)
        for t in range(T):
            cam_centers[t] = (0.86*cam_centers[t-1] + 0.14*center[t]) if t else center[t]
            pts = np.concatenate([s[t], d[t:t + 1]], axis = 0)
            half_widths[t] = max(23., np.max(np.abs(pts-cam_centers[t])) + 7.)
        for t in range(1, T):
            half_widths[t] = max(half_widths[t], 0.90*half_widths[t - 1])

    def update(t):
        start = max(0, t - trail + 1)
        for i, line in enumerate(sheep_trails):
            line.set_data(s[start:t + 1, i, 0], s[start:t + 1, i, 1])
        dog_trail.set_data(d[start:t + 1, 0], d[start:t + 1, 1])
        centre_marker.set_data([center[t, 0]], [center[t, 1]])
        if sheep_icons:
            for i, icon in enumerate(sheep_icons):
                icon.xybox = tuple(s[t, i])
                icon.xy = tuple(s[t, i])
            dog_icon.xybox = tuple(d[t])
            dog_icon.xy = tuple(d[t])
        else:
            sheep_scatter.set_offsets(s[t])
            dog_scatter.set_offsets(d[t:t + 1])
        status = f"  |  Dog: {modes[t]}" if modes and modes[t] != "—" else ""
        info.set_text(f"Step {t + 1}/{T}  |  Sheep: {n}{status}")
        if camera == "follow":
            cx, cy = cam_centers[t]
            half = half_widths[t]
            ax.set_xlim(cx - half, cx + half)
            ax.set_ylim(cy - half, cy + half)
        return []

    update(0)
    animation = FuncAnimation(fig, update, frames = T, interval = interval, repeat = repeat, blit = False)
    fig._herding_animation = animation
    plt.show()

def main():
    parser = argparse.ArgumentParser(description = __doc__)
    parser.add_argument("data", type = Path, help = "Simulation .npz file")
    parser.add_argument("--animate", action = "store_true", help = "Show interactive animation")
    parser.add_argument("--run", type = int, default = 0, help = "Zero-based run index (default 0)")
    parser.add_argument("--trail", type = int, default = 20, help = "Previous positions to display")
    parser.add_argument("--interval", type = int, default = 50, help = "Milliseconds between frames")
    parser.add_argument("--camera", choices = ["follow", "fixed"], default = "follow")
    parser.add_argument("--no-icons", action = "store_true", help = "Use colored dots instead of PNGs")
    parser.add_argument("--no-repeat", action = "store_true", help = "Stop at the end")
    args = parser.parse_args()
    if args.trail < 1 or args.interval < 1:
        parser.error("--trail and --interval must both be positive")
    create_animation(args.data, args.run, args.trail, args.interval, args.camera, not args.no_icons, not args.no_repeat)

if __name__ == "__main__":
    main()