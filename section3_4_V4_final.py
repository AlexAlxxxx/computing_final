# -*- coding: utf-8 -*-
"""
Created on Sun Jul 23 18:05:21 2023

@author: Alex A
"""
import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^−1 s^−2)
m_sun = 1.989e30  # Mass of the Sun (kg)
m_star = 5.972e30  # Mass of the Star (kg)
m_mars = 6.39e23  # Mass of Mars (kg)
au = 1.496e11  # Astronomical unit (m)
one_year = 3.154e7  # One year in seconds (s)

# Initial conditions
x0 = [
    -0.1*au,0, 0,-np.sqrt(G * m_sun / au), -np.sqrt(G * m_sun / au), np.sqrt(G * m_sun / au),  # Initial conditions for Sun [x, y, z, vx, vy, vz]
    0.1*au, 0, 0, np.sqrt(G * m_sun / au), np.sqrt(G * m_sun / au), np.sqrt(G * m_sun / au),  # Initial conditions for Star [x, y, z, vx, vy, vz]
    0, 0.6*au, 0, np.sqrt(G * m_sun / au), np.sqrt(G * m_sun / au), np.sqrt(G * m_sun / au)  # Initial conditions for Mars [x, y, z, vx, vy, vz]
]

def three_body_system(x, t):
    # Unpacking the state variables
    (
        x_sun, y_sun, z_sun, vx_sun, vy_sun, vz_sun,
        x_star, y_star, z_star, vx_star, vy_star, vz_star,
        x_mars, y_mars, z_mars, vx_mars, vy_mars, vz_mars
    ) = x

    # Distance between objects
    r_sun_star = np.sqrt((x_sun - x_star) ** 2 + (y_sun - y_star) ** 2 + (z_sun - z_star) ** 2)
    r_sun_mars = np.sqrt((x_sun - x_mars) ** 2 + (y_sun - y_mars) ** 2 + (z_sun - z_mars) ** 2)
    r_star_mars = np.sqrt((x_star - x_mars) ** 2 + (y_star - y_mars) ** 2 + (z_star - z_mars) ** 2)

    # Gravitational forces
    fx_sun = (
        (-G * m_star * (x_sun - x_star)) / r_sun_star ** 3
    ) + (
        (-G * m_mars * (x_sun - x_mars)) / r_sun_mars ** 3
    )
    fy_sun = (
        (-G * m_star * (y_sun - y_star)) / r_sun_star ** 3
    ) + (
        (-G * m_mars * (y_sun - y_mars)) / r_sun_mars ** 3
    )
    fz_sun = (
        (-G * m_star * (z_sun - z_star)) / r_sun_star ** 3
    ) + (
        (-G * m_mars * (z_sun - z_mars)) / r_sun_mars ** 3
    )

    fx_star = (
        (G * m_sun * (x_sun - x_star)) / r_sun_star ** 3
    ) + (
        (-G * m_mars * (x_star - x_mars)) / r_star_mars ** 3
    )
    fy_star = (
        (G * m_sun * (y_sun - y_star)) / r_sun_star ** 3
    ) + (
        (-G * m_mars * (y_star - y_mars)) / r_star_mars ** 3
    )
    fz_star = (
        (G * m_sun * (z_sun - z_star)) / r_sun_star ** 3
    ) + (
        (-G * m_mars * (z_star - z_mars)) / r_star_mars ** 3
    )

    fx_mars = (
        (G * m_sun * (x_sun - x_mars)) / r_sun_mars ** 3
    ) + (
        (G * m_star * (x_star - x_mars)) / r_star_mars ** 3
    )
    fy_mars = (
        (G * m_sun * (y_sun - y_mars)) / r_sun_mars ** 3
    ) + (
        (G * m_star * (y_star - y_mars)) / r_star_mars ** 3
    )
    fz_mars = (
        (G * m_sun * (z_sun - z_mars)) / r_sun_mars ** 3
    ) + (
        (G * m_star * (z_star - z_mars)) / r_star_mars ** 3
    )

    # Equations of motion for objects
    dx_sun_dt = vx_sun
    dy_sun_dt = vy_sun
    dz_sun_dt = vz_sun
    dvx_sun_dt = fx_sun
    dvy_sun_dt = fy_sun
    dvz_sun_dt = fz_sun

    dx_star_dt = vx_star
    dy_star_dt = vy_star
    dz_star_dt = vz_star
    dvx_star_dt = fx_star
    dvy_star_dt = fy_star
    dvz_star_dt = fz_star

    dx_mars_dt = vx_mars
    dy_mars_dt = vy_mars
    dz_mars_dt = vz_mars
    dvx_mars_dt = fx_mars
    dvy_mars_dt = fy_mars
    dvz_mars_dt = fz_mars

    return [
        dx_sun_dt, dy_sun_dt, dz_sun_dt, dvx_sun_dt, dvy_sun_dt, dvz_sun_dt,
        dx_star_dt, dy_star_dt, dz_star_dt, dvx_star_dt, dvy_star_dt, dvz_star_dt,
        dx_mars_dt, dy_mars_dt, dz_mars_dt, dvx_mars_dt, dvy_mars_dt, dvz_mars_dt
    ]

# Time points
t = np.linspace(0, 3.154e7 * 5, 5000)  # 5 years (seconds)

# Solving the ODE system
sol = odeint(three_body_system, x0, t)

# Extracting the positions of objects
x_sun, y_sun, z_sun, _, _, _, x_star, y_star, z_star, _, _, _, x_mars, y_mars, z_mars, _, _, _ = sol.T

# Plotting the orbits
fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')
ax.plot(x_sun / au, y_sun / au, z_sun / au, label='Sun')
ax.plot(x_star / au, y_star / au, z_star / au, label='Star')
ax.plot(x_mars / au, y_mars / au, z_mars / au, label='Mars')
ax.set_xlabel('x [AU]')
ax.set_ylabel('y [AU]')
ax.set_zlabel('z [AU]')
ax.set_title('Three-Body System Orbits')
ax.legend()
#plt.show()
plt.savefig('2star1planet.png')


