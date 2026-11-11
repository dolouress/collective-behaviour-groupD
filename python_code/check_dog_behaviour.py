# tests whether the implemented dog movement rules behave consistently
import numpy as np

data = np.load("python_corrected.npz")
dog = data["pos_d"]
collect = data["collect_t"]
drive = data["drive_t"]
slow = data["force_slow"]
collecting = np.isfinite(collect[1:])
driving = np.isfinite(drive[1:])
slowing = np.isfinite(slow[1:])

print("Collecting steps:", collecting.sum())
print("Driving steps:", driving.sum())
print("Slowdown steps:", slowing.sum())

events = (collecting.astype(int) + driving.astype(int) + slowing.astype(int))
print("Every step has exactly one mode:", np.all(events == 1))

displacement = np.linalg.norm(np.diff(dog, axis = 0), axis = 1)
print("Normal speed correct:", np.allclose(displacement[collecting | driving],1.5, atol = 1e-8))
print("Slowdown speed correct:", np.allclose(displacement[slowing], 0.05, atol = 1e-8))
print("All dog positions finite:", np.isfinite(dog).all())
