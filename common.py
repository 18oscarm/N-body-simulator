import numpy as np
import matplotlib.pyplot as plt

class System:
    def __init__(
        self, num_particles: int, x: np.ndarray, v: np.ndarray, m: np.ndarray, G: float
    ) -> None:
        self.num_particles = num_particles
        self.x = x
        self.v = v
        self.m = m
        self.G = G

    def center_of_mass_correction(self) -> None:
        """Here we set the center of mass-position 
        and center of mass-velocity of the system to 0
        We do this to centre the system and make sure there is no energy runnaway"""
        x_cm = np.zeros(3) #0.000, center mass-position
        v_cm = np.zeros(3) #average/ center velocity position
        M = 0.0 #total mass
        for i in range(self.num_particles):
            x_cm += self.m[i] * self.x[i]
            v_cm += self.m[i] * self.v[i]
            M += self.m[i]

        x_cm /= M
        v_cm /= M
        self.x -= x_cm #here we have initialised/relativised the position
        self.v -= v_cm

    # def plot_initial_conditions(
    #     system: System,
    #     labels: list,
    #     colours: list,
    #     legend: bool,
    # ) -> None:
    #     """
    #     Plot the initial positions.

    #     Parameters
    #     ----------
    #     system : System
    #         System object.
    #     labels : list
    #         Labels for the particles.
    #     colors : list
    #         Colors for the particles.
    #     legend : bool
    #         Whether to show the legend.
    #     """
        
    #     fig, ax = plt.subplots()
    #     ax.set_xlabel("$x$ (AU)")
    #     ax.set_ylabel("$y$ (AU)")

    #     for i in range(system.num_particles):
    #         ax.scatter(
    #             system.x[i,0], system.x[i,1], marker="o", colour = colour[i], label = labels[i]
    #         )

    #     if legend:
    #         ax.legend()

    #     plt.show()
