import numpy as np
import matplotlib.pyplot as plt
from py_opt_control import min_time_bvp

# Constants
v_min =  -10
v_max =   10
a_min =   -5
a_max =    5
j_min = -100
j_max =  100

sync_w = False

example_index = 4

if example_index == 1:
    # Returns "Error: Could not find valid timed solution!" with sync_w=False,
    # works with sync_w=True. Works in Matlab. Works if jerk limits modified to
    # (-99.9, 100). Solution is still fairly high quality.
    (p0, v0, a0) = (np.array([ 0.1,  0.2,  0.5]), np.array([0., 0., 0.]), np.array([0., 0., 0.]))
    (p1, v1, a1) = (np.array([ 0.0,  0.0,  0.0]), np.array([0., 0., 0.]), np.array([0., 0., 0.]))
elif example_index == 2:
    # Returns "Error: Could not find valid timed solution!" with sync_w=False,
    # works with sync_w=True. Works in Matlab. Works if jerk limits modified to
    # (-99.9, 100). Solution is still fairly high quality.
    (p0, v0, a0) = (np.array([ 0.6 ,  0.9]), np.array([0., 0.]), np.array([0., 0.]))
    (p1, v1, a1) = (np.array([ 1.0,   1.0]), np.array([0., 0.]), np.array([0., 0.]))
elif example_index == 3:
    # Core dump in "evaluate_to_time" with both sync_w=False and sync_w=True.
    # Works in Matlab. Works if jerk limits modified to (-99.9, 100). Error
    # text: "Index exceeds array dimensions.  Index value 1 exceeds valid range
    # [1-0] of array index_setp_max. Error in evaluate_to_time (line 25) Aborted
    # (core dumped)".
    (p0, v0, a0) = (np.array([ 1.0 , -0.8]), np.array([-1.0,  0.5]), np.array([0., 0.]))
    (p1, v1, a1) = (np.array([ 1.0,   1.0]), np.array([ 1.0, -1.0]), np.array([0., 0.]))
elif example_index == 4:
    # Returns "Error: Could not find valid timed solution!" with sync_w=False,
    # works with sync_w=True. Works in Matlab. Works if jerk limits modified to
    # (-97, 100), but (-99.9, 100) still doesn't work.
    (p0, v0, a0) = (np.array([-0.2,  0.6]), np.array([0.1, 1. ]), np.array([-0.1, -0.6]))
    (p1, v1, a1) = (np.array([0., 0.]), np.array([0., 0.]), np.array([0., 0.]))
else:
    assert False, "Must select a valid example_index."

# Compute the jerk input sequence. The initial state (p0, v0, a0) and input
# sequence (t, j) is the smallest description of a solution trajectory.
(t, j) = min_time_bvp.min_time_bvp(
    p0, v0, a0,
    p1, v1, a1,
    v_min, v_max, a_min, a_max, j_min, j_max,
    sync_w=sync_w)

# Analytically integrate the full state at the switching times. Then densely
# sample the trajectory over time for plotting purposes.
a, v, p = min_time_bvp.switch_states(p0, v0, a0, t, j)
st, sj, sa, sv, sp = min_time_bvp.uniformly_sample(p0, v0, a0, t, j, dt=0.001)

# Plot the state over time.
fig, axes = plt.subplots(4, 1, sharex=True)
for i in range(sp.shape[0]):
    for ax, s, l in zip(axes, [sp, sv, sa, sj], ('pos', 'vel', 'acc', 'jerk')):
        ax.plot(st, s[i,:])
        ax.set_ylabel(l)
axes[3].set_xlabel('time')
fig.suptitle('Full State over Time')

# Show plots.
plt.show()
