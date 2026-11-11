import argparse
from dataclasses import fields
from pathlib import Path
import numpy as np
from herding_model import Parameters, herding_model

def simulate(runs: int = 1, seed: int = 42, p: Parameters | None = None):
    p = p or Parameters()
    rng = np.random.default_rng(seed)
    result = [herding_model(p, rng) for _ in range(runs)]
    arrays = {name: np.stack([run[name] for run in result], axis = -1)
              for name in result[0]}
    for f in fields(p):
        arrays[f.name] = np.asarray(getattr(p, f.name))
    arrays['no_it'] = np.asarray(runs)
    return arrays

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type = int, default = 1)
    parser.add_argument('--seed', type = int, default = 42)
    parser.add_argument('--output', default = 'python_baseline.npz')
    args = parser.parse_args()
    result = simulate(args.runs, args.seed)
    np.savez_compressed(args.output, **result)

    print(f'Saved {args.runs} run(s) to {Path(args.output).resolve()}')
    print('Sheep shape:', result['pos_s'].shape, 'dog shape:', result['pos_d'].shape)

    for key in ('pos_s','pos_d','vel_s','vel_d'):
        print(f'{key}: finite fraction = {np.mean(np.isfinite(result[key])):.4f}')
