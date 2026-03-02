# -*- coding: utf-8 -*-
"""
Created on Mon Mar  2 10:51:31 2026

@author: zks524
"""

import astropy.units as u
import matplotlib.pyplot as plt
import numpy as np

from plasmapy.diagnostics.charged_particle_radiography import (
    synthetic_radiography as cpr,
)
from plasmapy.plasma.grids import CartesianGrid


# Create a Cartesian grid
L = 1 * u.mm
grid = CartesianGrid(-L, L, num=100)

# Create a spherical potential with a Gaussian radial distribution
radius = np.linalg.norm(grid.grid, axis=3)
arg = (radius / (L / 3)).to(u.dimensionless_unscaled)
potential = 2e5 * np.exp(-(arg**2)) * u.V

# Calculate E from the potential
Ex, Ey, Ez = np.gradient(potential, grid.dax0, grid.dax1, grid.dax2)
Ex = -np.where(radius < L / 2, Ex, 0)
Ey = -np.where(radius < L / 2, Ey, 0)
Ez = -np.where(radius < L / 2, Ez, 0)

# Add those quantities to the grid
grid.add_quantities(E_x=Ex, E_y=Ey, E_z=Ez, phi=potential)


# Plot the E-field
fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
ax.view_init(30, 30)

# skip some points to make the vector plot intelligible
s = tuple([slice(None, None, 6)] * 3)

ax.quiver(
    grid.pts0[s].to(u.mm).value,
    grid.pts1[s].to(u.mm).value,
    grid.pts2[s].to(u.mm).value,
    grid["E_x"][s],
    grid["E_y"][s],
    grid["E_z"][s],
    length=1e-6,
)

ax.set_xlabel("X (mm)")
ax.set_ylabel("Y (mm)")
ax.set_zlabel("Z (mm)")
ax.set_xlim(-1, 1)
ax.set_ylim(-1, 1)
ax.set_zlim(-1, 1)
ax.set_title("Gaussian Potential Electric Field");

plt.show()


source = (0 * u.mm, -10 * u.mm, 0 * u.mm)
detector = (0 * u.mm, 1000 * u.mm, 0 * u.mm)

sim = cpr.Tracker(grid, source, detector, verbose=True)


sim.create_particles(1e2, 3 * u.MeV, max_theta=np.pi / 15 * u.rad, particle="p", distribution="uniform")

sim.run();


fig = plt.figure(figsize=(8, 8))
ax = fig.add_subplot(111, projection="3d")
ax.view_init(30, 150)
ax.set_xlabel("X (cm)")
ax.set_ylabel("Y (cm)")
ax.set_zlabel("Z (cm)")

# Plot the source-to-detector axis
ax.quiver(
    sim.source[0] * 100,
    sim.source[1] * 100,
    sim.source[2] * 100,
    sim.detector[0] * 100,
    sim.detector[1] * 100,
    sim.detector[2] * 100,
    color="black",
)

# Plot the simulation field grid volume
ax.scatter(0, 0, 0, color="green", marker="s", linewidth=5, label="Simulated Fields")

# Plot the proton source and detector plane locations
ax.scatter(
    sim.source[0] * 100,
    sim.source[1] * 100,
    sim.source[2] * 100,
    color="red",
    marker="*",
    linewidth=5,
    label="Source",
)

ax.scatter(
    sim.detector[0] * 100,
    sim.detector[1] * 100,
    sim.detector[2] * 100,
    color="blue",
    marker="*",
    linewidth=10,
    label="Detector",
)


# Plot the final proton positions of some (not all) of the protons
ind = slice(None, None, 200)
ax.scatter(
    sim.x[ind, 0] * 100,
    sim.x[ind, 1] * 100,
    sim.x[ind, 2] * 100,
    label="Protons",
)

ax.legend();
plt.show()




# A function to reduce repetitive plotting


def plot_radiograph(hax, vax, intensity):
    fig, ax = plt.subplots(figsize=(8, 8))
    plot = ax.pcolormesh(
        hax.to(u.cm).value,
        vax.to(u.cm).value,
        intensity.T,
        cmap="Blues_r",
        shading="auto",
    )
    cb = fig.colorbar(plot)
    cb.ax.set_ylabel("Intensity")
    ax.set_aspect("equal")
    ax.set_xlabel("X (cm), Image plane")
    ax.set_ylabel("Z (cm), Image plane")
    ax.set_title("Synthetic Proton Radiograph")


size = np.array([[-1, 1], [-1, 1]]) * 1.5 * u.cm
bins = [200, 200]
hax, vax, intensity = cpr.synthetic_radiograph(sim, size=size, bins=bins)
plot_radiograph(hax, vax, intensity)
plt.show()


max_deflection = sim.max_deflection
print(f"Maximum deflection α = {np.rad2deg(max_deflection):.2f}")

a = 1 * u.mm
l = np.linalg.norm(sim.source * u.m).to(u.mm)
mu = l * max_deflection.value / a
print(f"a = {a}")
print(f"l = {l:.1f}")
print(f"μ = {mu:.2f}")
