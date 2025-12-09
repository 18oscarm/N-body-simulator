# N-body-simulator

A simple 2D N-body gravitational simulator in Python — with support for both direct force simulation and a Barnes–Hut optimised version.  
Great for exploring chaotic dynamics, gravitational clustering, or learning how tree-based approximations accelerate N-body simulations.

## 🚀 Features

- **Direct (O(N²)) simulation** via naive force computation.  
- **Barnes–Hut tree (O(N log N)) implementation** for faster simulations with larger particle counts.  
- Configurable simulation parameters: gravitational constant `G`, time step `dt`, softening, etc.  
- Built-in animation of body trajectories using `matplotlib`.  
- Easy to extend: swap integrators, adjust initial conditions, add visualisation tweaks, or plug in different force laws.  



