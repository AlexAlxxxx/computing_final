# -*- coding: utf-8 -*-
"""
Created on Fri Jul 21 20:33:32 2023

@author: Alex A
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^−1 s^−2)
m_star1 = 1.989e30  # Mass of star 1 (Sun) (kg)
m_star2 = 5.972e30  # Mass of star 2 (Earth) (kg)
m_star3 = 6.39e30  # Mass of star 3 (Mars) (kg)
au = 1.496e11  # Astronomical unit (m)
one_year = 3.154e7  # One year in seconds (s)

# Initial conditions
x0 = [
    1*au,1*au, 0,np.sqrt(G * m_star1 / au), np.sqrt(G * m_star1 / au), 0,  # Initial conditions for star 1 [x, y, z, vx, vy, vz]
    1.5*au, 0.5*au, 0, np.sqrt(G * m_star1 / au), np.sqrt(G * m_star1 / au), 0,  # Initial conditions for star 2 [x, y, z, vx, vy, vz]
    2.0*au, -1*au, 0, np.sqrt(G * m_star1 / au), np.sqrt(G * m_star1 / au), 0  # Initial conditions for star 3 [x, y, z, vx, vy, vz]
]

def three_body_system(x, t):
    # Unpacking the state variables
    (
        x_star1, y_star1, z_star1, vx_star1, vy_star1, vz_star1,
        x_star2, y_star2, z_star2, vx_star2, vy_star2, vz_star2,
        x_star3, y_star3, z_star3, vx_star3, vy_star3, vz_star3
    ) = x

    # Distance between objects
    r_star1_star2 = np.sqrt((x_star1 - x_star2) ** 2 + (y_star1 - y_star2) ** 2 + (z_star1 - z_star2) ** 2)
    r_star1_star3 = np.sqrt((x_star1 - x_star3) ** 2 + (y_star1 - y_star3) ** 2 + (z_star1 - z_star3) ** 2)
    r_star2_star3 = np.sqrt((x_star2 - x_star3) ** 2 + (y_star2 - y_star3) ** 2 + (z_star2 - z_star3) ** 2)

    # Gravitational forces
    fx_star1 = (
        (-G * m_star2 * (x_star1 - x_star2)) / r_star1_star2 ** 3
    ) + (
        (-G * m_star3 * (x_star1 - x_star3)) / r_star1_star3 ** 3
    )
    fy_star1 = (
        (-G * m_star2 * (y_star1 - y_star2)) / r_star1_star2 ** 3
    ) + (
        (-G * m_star3 * (y_star1 - y_star3)) / r_star1_star3 ** 3
    )
    fz_star1 = (
        (-G * m_star2 * (z_star1 - z_star2)) / r_star1_star2 ** 3
    ) + (
        (-G * m_star3 * (z_star1 - z_star3)) / r_star1_star3 ** 3
    )

    fx_star2 = (
        (G * m_star1 * (x_star1 - x_star2)) / r_star1_star2 ** 3
    ) + (
        (-G * m_star3 * (x_star2 - x_star3)) / r_star2_star3 ** 3
    )
    fy_star2 = (
        (G * m_star1 * (y_star1 - y_star2)) / r_star1_star2 ** 3
    ) + (
        (-G * m_star3 * (y_star2 - y_star3)) / r_star2_star3 ** 3
    )
    fz_star2 = (
        (G * m_star1 * (z_star1 - z_star2)) / r_star1_star2 ** 3
    ) + (
        (-G * m_star3 * (z_star2 - z_star3)) / r_star2_star3 ** 3
    )

    fx_star3 = (
        (G * m_star1 * (x_star1 - x_star3)) / r_star1_star3 ** 3
    ) + (
        (G * m_star2 * (x_star2 - x_star3)) / r_star2_star3 ** 3
    )
    fy_star3 = (
        (G * m_star1 * (y_star1 - y_star3)) / r_star1_star3 ** 3
    ) + (
        (G * m_star2 * (y_star2 - y_star3)) / r_star2_star3 ** 3
    )
    fz_star3 = (
        (G * m_star1 * (z_star1 - z_star3)) / r_star1_star3 ** 3
    ) + (
        (G * m_star2 * (z_star2 - z_star3)) / r_star2_star3 ** 3
    )

    # Equations of motion for objects
    dx_star1_dt = vx_star1
    dy_star1_dt = vy_star1
    dz_star1_dt = vz_star1
    dvx_star1_dt = fx_star1
    dvy_star1_dt = fy_star1
    dvz_star1_dt = fz_star1

    dx_star2_dt = vx_star2
    dy_star2_dt = vy_star2
    dz_star2_dt = vz_star2
    dvx_star2_dt = fx_star2
    dvy_star2_dt = fy_star2
    dvz_star2_dt = fz_star2

    dx_star3_dt = vx_star3
    dy_star3_dt = vy_star3
    dz_star3_dt = vz_star3
    dvx_star3_dt = fx_star3
    dvy_star3_dt = fy_star3
    dvz_star3_dt = fz_star3

    return [
        dx_star1_dt, dy_star1_dt, dz_star1_dt, dvx_star1_dt, dvy_star1_dt, dvz_star1_dt,
        dx_star2_dt, dy_star2_dt, dz_star2_dt, dvx_star2_dt, dvy_star2_dt, dvz_star2_dt,
        dx_star3_dt, dy_star3_dt, dz_star3_dt, dvx_star3_dt, dvy_star3_dt, dvz_star3_dt
    ]

# Time points
t = np.linspace(0, 3.154e7 * 5, 50000)  # 5 years (seconds)

# Solving the ODE system
sol = odeint(three_body_system, x0, t)

# Extracting the positions of objects
x_star1, y_star1, z_star1, _, _, _, x_star2, y_star2, z_star2, _, _, _, x_star3, y_star3, z_star3, _, _, _ = sol.T

# Plotting the orbits
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_star1 / au, y_star1 / au, z_star1 / au, label='Star 1')
ax.plot(x_star2 / au, y_star2 / au, z_star2 / au, label='Star 2')
ax.plot(x_star3 / au, y_star3 / au, z_star3 / au, label='Star 3')
ax.set_xlabel('x [AU]')
ax.set_ylabel('y [AU]')
ax.set_zlabel('z [AU]')
ax.set_zlim(0.006, -0.006)
ax.set_ylim(0, 10)
ax.set_xlim(0, 5)
ax.set_title('Three-Body System Orbits')
ax.legend()
#plt.show()
plt.savefig('3stars.png')

