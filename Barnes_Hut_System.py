# Barnes_Hut.py
# Quadtree + simple Body type used by the Barnes-Hut system.

import numpy as np

class Body:
    """Simple body object used for building the quadtree.
       Optionally include an index to map back to System arrays.
    """
    def __init__(self, x, y, mass, idx=None, vx=0.0, vy=0.0):
        self.pos = np.array([x, y], dtype=float)
        self.mass = float(mass)
        self.idx = idx
        self.vel = np.array([vx, vy], dtype=float)

    def __repr__(self):
        return f"Body(idx={self.idx}, pos=({self.pos[0]:.2f},{self.pos[1]:.2f}), m={self.mass:.2f})"


class QuadTree:
    """A simple quadtree for Barnes-Hut 2D.

    Each node stores an aggregate total_mass and center_of_mass.
    Insert bodies with insert(body). Compute acceleration on a given body
    via compute_accel_on(body, theta, G, eps).
    """
    def __init__(self, x_center: float, y_center: float, width: float, capacity: int = 1):
        self.x_center = float(x_center)
        self.y_center = float(y_center)
        self.width = float(width)    # square region
        self.capacity = capacity

        # aggregate properties
        self.total_mass = 0.0
        self.center_of_mass = np.array([0.0, 0.0], dtype=float)

        # structure
        self.body = None      # single Body if leaf and occupied
        self.children = None  # list of 4 QuadTree nodes: [NW, NE, SW, SE]

    def contains(self, pos):
        """pos may be numpy array or sequence (x,y)."""
        x, y = float(pos[0]), float(pos[1])
        half = self.width / 2.0
        return (self.x_center - half <= x < self.x_center + half and
                self.y_center - half <= y < self.y_center + half)

    def subdivide(self):
        """Create 4 children: NW, NE, SW, SE"""
        half = self.width / 2.0
        quarter = self.width / 4.0
        new_width = half
        self.children = [
            QuadTree(self.x_center - quarter, self.y_center + quarter, new_width, self.capacity),  # NW
            QuadTree(self.x_center + quarter, self.y_center + quarter, new_width, self.capacity),  # NE
            QuadTree(self.x_center - quarter, self.y_center - quarter, new_width, self.capacity),  # SW
            QuadTree(self.x_center + quarter, self.y_center - quarter, new_width, self.capacity),  # SE
        ]

    def _update_aggregate_with(self, body):
        """Update total_mass and center_of_mass with new body's mass/pos."""
        old_mass = self.total_mass
        new_mass = body.mass
        new_total = old_mass + new_mass
        if new_total > 0.0:
            # weighted average
            self.center_of_mass = (self.center_of_mass * old_mass + body.pos * new_mass) / new_total
        self.total_mass = new_total

    def insert(self, body):
        """Insert a Body into the tree. Returns True if inserted."""
        if not self.contains(body.pos):
            return False

        # Update aggregate (do this for every node on path)
        self._update_aggregate_with(body)

        # If empty leaf -> store body
        if self.children is None and self.body is None:
            self.body = body
            return True

        # If leaf but occupied -> subdivide then re-insert existing body and new one
        if self.children is None and self.body is not None:
            # subdivide
            self.subdivide()
            # re-insert the existing body into one child
            existing = self.body
            self.body = None
            for child in self.children:
                if child.contains(existing.pos):
                    child.insert(existing)
                    break
            # insert the new body
            for child in self.children:
                if child.contains(body.pos):
                    child.insert(body)
                    break
            return True

        # If internal node -> pass down to the appropriate child
        if self.children is not None:
            for child in self.children:
                if child.contains(body.pos):
                    child.insert(body)
                    return True

        # fallback
        return False

    def compute_accel_on(self, target_body, theta=0.5, G=1.0, eps=1e-3):
        """Compute acceleration vector on target_body by recursively traversing the tree.

        Parameters:
            target_body: Body instance that is present in this tree (the same object inserted)
            theta: opening angle threshold
            G: gravitational constant
            eps: softening length (added to r^2 to avoid singularities)
        Returns:
            numpy array (2,) acceleration
        """
        # If there's no mass here, no force
        if self.total_mass == 0.0:
            return np.zeros(2, dtype=float)

        # Vector from target to this node's center-of-mass
        r_vec = self.center_of_mass - target_body.pos
        dist2 = np.dot(r_vec, r_vec) + eps * eps
        dist = np.sqrt(dist2)

        # If this node is a leaf and contains only the target body itself -> skip
        if self.children is None and self.body is not None and self.body is target_body:
            return np.zeros(2, dtype=float)

        # If node is far enough: treat as single mass
        if (self.width / dist) < theta or self.children is None:
            # acceleration = G * M * r_vec / |r|^3
            return G * self.total_mass * r_vec / (dist2 * dist)  # = G*M*r / (r^3)
        else:
            # otherwise sum contributions from children
            total = np.zeros(2, dtype=float)
            if self.children:
                for child in self.children:
                    total += child.compute_accel_on(target_body, theta, G, eps)
            return total

    # Optional: small repr for debugging
    def __repr__(self):
        return f"QuadTree(center=({self.x_center:.1f},{self.y_center:.1f}), w={self.width:.1f}, M={self.total_mass:.2f})"
