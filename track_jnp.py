# import numpy as np 
import matplotlib.pyplot as plt
import matplotlib
from scipy.interpolate import CubicSpline
from scipy.interpolate import CubicHermiteSpline
import jax.numpy as jnp
from jax import jit
from functools import partial
gate_dimensions = [1.6, 1.6]

class Splined_Track():
    def __init__(self, waypoints, track_width):
        """
        :param waypoints: Array of track center waypoints in the form [[x1, y1], [x2, y2], ...]
        :param track_width: Width of the track
        
        n_gates: Number of gates (waypoints)
        arc_length: Array of arc lengths between the gates. spline maps from arc_length to waypoints
        track: Cubic spline object
        track_centers: Array of track center points extrapolated from the spline
        """

        dists = jnp.linalg.norm(waypoints[1:, :] - waypoints[:-1, :], axis=1)
        self.arc_length = jnp.zeros(shape=jnp.size(waypoints, 0))
        self.arc_length = self.arc_length.at[1:].set(jnp.cumsum(dists))
        self.waypoints = waypoints
        self.track_width = track_width
        self.track = CubicSpline(self.arc_length, waypoints, bc_type='periodic')
        dists = jnp.linalg.norm(waypoints[1:, :] - waypoints[:-1, :], axis=1)
        taus = jnp.linspace(0, self.arc_length[-1], 2**12)
        self.track_centers = jnp.array(self.track(taus))
        # track_center_dist = np.linalg.norm(self.track_centers[1:, :] - self.track_centers[:-1, :], axis=1)
        # self.progress = np.zeros_like(self.track_centers)
        # self.progress[1:] = np.cumsum(track_center_dist)
        self.track_tangent = self.track.derivative(nu=1)(taus)
        self.track_tangent /= jnp.linalg.norm(self.track_tangent, axis=1)[:, jnp.newaxis]
        self.track_normals = jnp.zeros_like(self.track_tangent)
        self.track_normals = self.track_normals.at[:, 0].set(-self.track_tangent[:, 1])
        self.track_normals = self.track_normals.at[:, 1].set(self.track_tangent[:, 0])
        self.track_normals /= jnp.linalg.norm(self.track_normals, axis=1)[:, jnp.newaxis]

    # @jit
    @partial(jit, static_argnums=(0,))
    def nearest_trackpoint(self, p): 
        """Find closest track frame to a reference point p.
        :param p: Point of reference
        :return: Index of track frame, track center, tangent and normal.

        function s in the paper
        """
        # print(p)
        i = jnp.linalg.norm(self.track_centers - p, axis=1).argmin()
        return i, self.track_centers[i], self.track_tangent[i], self.track_normals[i]
    
    def plot_waypoints_2d(self, ax):
        ax.plot(self.waypoints[:, 0], self.waypoints[:, 1], 'x')

    def plot_track(self, ax, draw_boundaries=False):
            # Plot the spline
            ax.plot(self.track_centers[:, 0], self.track_centers[:, 1], '--')
            # Plot the track width
            if draw_boundaries:
                left_boundary = self.track_centers + self.track_normals*self.track_width
                right_boundary = self.track_centers - self.track_normals*self.track_width
                ax.plot(left_boundary[:, 0], left_boundary[:, 1], 'b-')
                ax.plot(right_boundary[:, 0], right_boundary[:, 1], 'b-')

def plot_tangents(ax, track):
    random_samples = jnp.random.randint(0, track.track_centers.shape[0], 10)
    ax.quiver(track.track_centers[random_samples, 0], track.track_centers[random_samples, 1], track.track_tangent[random_samples, 0], track.track_tangent[random_samples, 1], color='r')
    ax.quiver(track.track_centers[random_samples, 0], track.track_centers[random_samples, 1], track.track_normals[random_samples, 0], track.track_normals[random_samples, 1], color='g')

if __name__ == "__main__":
    gates = jnp.array([])
    track_width = 0.2
    track = Splined_Track(gates, track_width)
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    # plot_waypoints_2d(ax, gates)
    # plot_track(ax, track)
    # plot_tangents(ax, track)
    random_points = (jnp.random.rand(20, 2) - 0.2)*5
    for rp in random_points:
        i, p, t, n = track.nearest_trackpoint(rp)
        ax.plot([rp[0], p[0]], [rp[1], p[1]], 'ro')
        ax.plot([rp[0], p[0]], [rp[1], p[1]], 'r-')
    plt.show()