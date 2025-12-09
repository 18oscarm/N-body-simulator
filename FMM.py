# Note: You would need to install this package first: pip install fmm
import numpy as np
import fmm

def fmm_force_calculation(bodies, p=10, L=3):
    """
    Calculates forces using the Fast Multipole Method (FMM).
    
    Parameters:
    - bodies: A list of Body objects (or equivalent structure).
    - p: The order of the multipole expansion (controls accuracy). Higher is slower but more accurate.
    - L: The level of the Quad/Octree (controls resolution).
    """
    if not bodies:
        return []

    # 1. Prepare Source Data (Body Positions and Masses)
    source_coords = np.array([body.pos for body in bodies]) # Shape (N, 2 or 3)
    source_masses = np.array([body.mass for body in bodies]) # Shape (N,)
    
    # In an N-body simulation, the sources and targets are the same set of bodies
    target_coords = source_coords 
    
    # 2. Initialize and Execute FMM
    # The 'fmm' library handles the tree construction and multipole calculations internally.
    
    # Initialize the FMM evaluator object
    # K is the kernel (0 for Laplace/Gravity in 2D or 3D).
    fmm_obj = fmm.FMM(
        source_coords=source_coords, 
        source_masses=source_masses, 
        target_coords=target_coords,
        p=p, 
        L=L, 
        K=0, # Assuming Laplace/Gravity kernel (1/r potential)
        eps=1e-8 # Required tolerance
    )
    
    # Evaluate the potential (potential) and its gradient (gradient, which is acceleration)
    results = fmm_obj.evaluate()
    
    # The 'results' will contain the potential and the gradient (force/mass)
    # The gradient is the acceleration vector (ax, ay, az) for each body
    accelerations = results['target']['gradient'] # Shape (N, 3) or (N, 2)
    
    # 3. Apply Accelerations to Bodies (if your Body class allows)
    # This step integrates the FMM result back into your simulation structure.
    for i, body in enumerate(bodies):
        # Gravitational acceleration 'a' is what FMM computes for the 1/r potential.
        # Note: Depending on the library's exact conventions, you might need to adjust 
        # for G or sign. For a standard Laplace kernel (like gravity), the result is typically 
        # the acceleration field a.
        body.acc = accelerations[i] 

    return accelerations