import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import math 

# system.py

from Barnes_Hut_System import QuadTree, Body as BHBody

class System:
    """
    A class to represent and simulate an N-body system using Euler integration.
    """
    def __init__(self, num_particles: int, x: np.ndarray, v: np.ndarray, m: np.ndarray, G: float) -> None:
        self.num_particles = num_particles
        self.x = x.astype(float)
        self.v = v.astype(float)
        self.m = m.astype(float)
        self.G = float(G)

        self.a = np.zeros_like(self.v, dtype=float)
        for i in range(num_particles):
            self.a[i] = self.accel_of_object(i)

        self.history_x = [self.x.copy()]
        self.step_count = 0

    def accel_of_object(self, i: int) -> np.ndarray:
        """
        Naive O(N^2) acceleration calculation (used by base System).
        """
        G = self.G
        total_accel = np.array([0.0, 0.0], dtype=float)
        r_i = self.x[i]
        for j in range(self.num_particles):
            if j == i:
                continue
            r_j = self.x[j]
            r_vec = r_j - r_i
            dist = np.linalg.norm(r_vec)
            if dist < 1e-9:
                continue
            total_accel += self.m[j] * r_vec / (dist ** 3)
        return G * total_accel

    def update(self, dt: float):
        """Euler step (positions first, then velocities)."""
        for i in range(self.num_particles):
            self.x[i] = self.x[i] + dt * self.v[i]
            self.v[i] = self.v[i] + dt * self.a[i]
        # recompute accelerations for next step
        for i in range(self.num_particles):
            self.a[i] = self.accel_of_object(i)

        self.history_x.append(self.x.copy())
        self.step_count += 1


class System_with_Sun(System):
    def __init__(self, num_particles, x, v, m, G):
        super().__init__(num_particles, x, v, m, G)
        x_coord = np.zeros(2)
        v_coord = np.zeros(2)
        self.adding_sun(x_coord, v_coord, 100)

    def adding_sun(self, x_coord, v_corrd, m):
        sol = Sun(x_coord, v_corrd, m)
        self.x = np.vstack([self.x, sol.x])
        self.v = np.vstack([self.v, sol.v])
        self.m = np.append(self.m, sol.m)
        self.num_particles += 1
        self.a = np.zeros_like(self.v, dtype=float)
        for i in range(self.num_particles):
            self.a[i] = self.accel_of_object(i)
        self.history_x = [self.x.copy()]
        self.step_count = 0


class Sun:
    def __init__(self, x, v, m):
        self.x = x
        self.v = v
        self.m = m


# ------------------ Barnes-Hut System ------------------

class BarnesHutSystem(System):
    """
    System subclass that uses Barnes-Hut tree for O(N log N) approx force computation.
    Build a QuadTree per timestep and compute accelerations via the aggregate nodes.
    """
    def __init__(self, num_particles, x, v, m, G, theta=0.5, eps=1e-3):
        super().__init__(num_particles, x, v, m, G)
        self.theta = theta
        self.eps = eps

    def build_tree(self):
        """Construct a QuadTree that encloses all bodies and insert BHBody objects."""
        # compute bounding square that contains all bodies
        xs = self.x[:, 0]
        ys = self.x[:, 1]
        min_x, max_x = xs.min(), xs.max()
        min_y, max_y = ys.min(), ys.max()
        width = max(max_x - min_x, max_y - min_y)
        # If all bodies are at the same point, give a small width
        if width <= 0:
            width = 1.0
        # make square slightly larger to avoid edge boundary issues
        width *= 1.1
        center_x = (min_x + max_x) / 2.0
        center_y = (min_y + max_y) / 2.0

        root = QuadTree(center_x, center_y, width)

        # Create BHBody objects and insert them
        self._bh_bodies = []
        for i in range(self.num_particles):
            b = BHBody(self.x[i, 0], self.x[i, 1], self.m[i], idx=i, vx=self.v[i, 0], vy=self.v[i, 1])
            root.insert(b)
            self._bh_bodies.append(b)
        return root

    def update(self, dt: float):
        """Override: build tree, compute accelerations for all bodies (BH), then integrate (Euler)."""
        # 1) Build tree
        root = self.build_tree()

        # 2) Compute accelerations using the tree
        new_a = np.zeros_like(self.v, dtype=float)
        for b in self._bh_bodies:
            acc = root.compute_accel_on(b, theta=self.theta, G=self.G, eps=self.eps)
            # acc is the acceleration vector on body b
            new_a[b.idx] = acc

        self.a = new_a

        # 3) Integrate positions and velocities (Euler)
        for i in range(self.num_particles):
            self.x[i] = self.x[i] + dt * self.v[i]
            self.v[i] = self.v[i] + dt * self.a[i]

        self.history_x.append(self.x.copy())
        self.step_count += 1
