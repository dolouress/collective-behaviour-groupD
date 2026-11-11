import numpy as np
data = np.load("test_run.npz")
sheep = data["pos_s"]
dog = data["pos_d"]

print("Sheep shape:", sheep.shape)
print("Dog shape:", dog.shape)
print("Sheep positions valid:", np.isfinite(sheep).all())
print("Dog positions valid:", np.isfinite(dog).all())

sheep_run = sheep[..., 0] if sheep.ndim == 4 else sheep
dog_run = dog[..., 0] if dog.ndim == 3 else dog
sheep_steps = np.linalg.norm(np.diff(sheep_run, axis = 2), axis = 1)
dog_steps = np.linalg.norm(np.diff(dog_run, axis = 0), axis = 1)

print("max sheep displacement:", sheep_steps.max())
print("max dog displacement:", dog_steps.max())
print("mean sheep displacement:", sheep_steps.mean())
print("mean dog displacement:", dog_steps.mean())