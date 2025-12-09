from system import System, System_with_Sun
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np

def animate_simulation(Solar: System, dt: float, max_steps: int, render_stride: int = 10):
    """Runs the simulation and plots only every `render_stride` frame."""
    
    import matplotlib.pyplot as plt
    import matplotlib.animation as animation
    import numpy as np

    # ---- Setup Figure ----
    fig, ax = plt.subplots(figsize=(10, 10))
    plt.style.use('dark_background')

    max_range = np.max(np.abs(Solar.x)) * 1.5
    ax.set_xlim([-max_range, max_range])
    ax.set_ylim([-max_range, max_range])
    ax.set_aspect('equal', adjustable='box')
    ax.set_title('N-Body Chaotic Motion (Euler Integration)')
    ax.set_xlabel('X Position')
    ax.set_ylabel('Y Position')

    body_markers = ax.scatter(Solar.x[:, 0], Solar.x[:, 1], s=100, marker='o', c='yellow')

    # Lines for trails
    lines = []
    colors = plt.cm.jet(np.linspace(0, 1, Solar.num_particles))
    for i in range(Solar.num_particles):
        line, = ax.plot([], [], lw=1, color=colors[i], alpha=0.7)
        lines.append(line)

    # ---- Animation Update ----
    def update_plot(frame):
        """
        frame counts visible animation frames,
        but actual simulation steps = frame * render_stride
        """
        target_step = frame * render_stride

        # run enough simulation steps to catch up
        while Solar.step_count < target_step and Solar.step_count < max_steps:
            Solar.update(dt)

        # update point positions
        body_markers.set_offsets(Solar.x)

        # update trails
        history = np.array(Solar.history_x)
        upto = min(target_step + 1, len(history))

        for i in range(Solar.num_particles):
            lines[i].set_data(history[:upto, i, 0], history[:upto, i, 1])

        ax.set_title(f'N-Body Chaotic Motion | Step: {Solar.step_count}/{max_steps}')
        return [body_markers] + lines

    # Number of animation frames is reduced
    total_frames = max_steps // render_stride + 1

    ani = animation.FuncAnimation(
        fig,
        update_plot,
        frames=total_frames,
        interval=20,
        blit=False,
        repeat=False
    )

    plt.show()
