import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math
from system import System, System_with_Sun
from animate import animate_simulation


# --- Main Script Execution ---
def run_simulation():
    # Configuration
    N = 10 # Start with 3 bodies to see chaos clearly
    dt = 0.00001 # Smaller time step for better stability with Euler integration
    total_steps = 1000 # Number of frames/steps for the animation

    # Initial Conditions 
    # Use a more constrained, non-random setup for initial testing of chaotic motion (e.g., three-body figure eight)
    pos_array = np.random.rand(N, 2)
    vel_array = np.random.rand(N, 2)
    mass_array = np.ones(N)

    G = 1.0

    # Initialize and Run Animation
    Solar = System(N, pos_array, vel_array, mass_array, G)
    animate_simulation(Solar, dt, total_steps)

def sun_simulation():
    # Configuration
    N = 10 # Start with 3 bodies to see chaos clearly
    dt = 0.0001 # Smaller time step for better stability with Euler integration
    total_steps = 5000 # Number of frames/steps for the animation

    # Initial Conditions 
    # Use a more constrained, non-random setup for initial testing of chaotic motion (e.g., three-body figure eight)
    pos_array = np.random.rand(N, 2)
    vel_array = np.random.rand(N, 2)
    mass_array = np.ones(N)

    G = 1.0

    # Initialize and Run Animation
    Solar = System_with_Sun(N, pos_array, vel_array, mass_array, G)
    animate_simulation(Solar, dt, total_steps)


def Barnes_Hut():
    # time complexity O(N log N)
    N = 10
    dt = 0.0001
    total_steps = 5000
    pos_array = np.random.rand(N, 2)
    vel_array = np.random.rand(N, 2)

    G = 1.0


if __name__ == "__main__":
    print("--- Starting N-Body Simulation Driver ---")
    Barnes_Hut()
    print("--- Simulation Finished ---")