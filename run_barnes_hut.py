# run_barnes_hut.py
import numpy as np
from system import BarnesHutSystem
from animate import animate_simulation

def Barnes_Hut_demo():
    N = 10
    dt = 0.01
    total_steps = 2000

    # Create random-ish initial positions centered around 0, scaled so we see structure
    rng = np.random.default_rng(42)
    pos_array = (rng.random((N, 2)) - 0.5) * 100.0  # positions ~ [-50, 50]
    vel_array = (rng.random((N, 2)) - 0.5) * 0.1   # small initial velocities
    mass_array = rng.random(N) * 5.0 + 0.5          # masses in [0.5, 5.5]

    G = 1.0
    theta = 0.5    # Barnes-Hut opening parameter (0.5 is a good default)
    eps = 1e-2     # softening to keep integrator stable

    solar = BarnesHutSystem(N, pos_array, vel_array, mass_array, G, theta=theta, eps=eps)
    animate_simulation(solar, dt, total_steps)

if __name__ == "__main__":
    Barnes_Hut_demo()
