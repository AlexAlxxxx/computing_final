import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^−1 s^−2)
m_sun = 1.989e30  # Mass of the Sun (kg)
m_star = 1.989e30  # Mass of the star (kg)
m_earth = 5.972e24  # Mass of the Earth (kg)
au = 1.496e11  # Astronomical unit (m)
one_year = 3.154e7  # One year in seconds (s)

# Initial conditions
x0 = [
    -0.1*au, 0, 0, -np.sqrt(G * m_sun / au),  # Initial conditions for Sun [x, y, vx, vy]
    0.1*au, 0, 0, np.sqrt(G * m_sun / au),  # Initial conditions for Star [x, y, vx, vy]
    0, 0.9*au, np.sqrt(G * m_sun / au), np.sqrt(G * m_sun / au),  # Initial conditions for earth [x, y, vx, vy]
]


def three_body_system(x, t):
    #  Unpacking the state variables
    x_sun, y_sun, vx_sun, vy_sun, x_star, y_star, vx_star, vy_star, x_earth, y_earth, vx_earth, vy_earth = x

    # Distance between objects
    r_sun_star = np.sqrt((x_sun - x_star) ** 2 + (y_sun - y_star) ** 2)
    r_sun_earth = np.sqrt((x_sun - x_earth) ** 2 + (y_sun - y_earth) ** 2)
    r_star_earth = np.sqrt((x_star - x_earth) ** 2 + (y_star - y_earth) ** 2)

    # Gravitational forces
    fx_sun = (
        (-G * m_star * (x_sun - x_star)) / r_sun_star ** 3
    ) + (
        (-G * m_earth * (x_sun - x_earth)) / r_sun_earth ** 3
    )
    fy_sun = (
        (-G * m_star * (y_sun - y_star)) / r_sun_star ** 3
    ) + (
        (-G * m_earth * (y_sun - y_earth)) / r_sun_earth ** 3
    )

    fx_star = (
        (G * m_sun * (x_sun - x_star)) / r_sun_star ** 3
    ) + (
        (-G * m_earth * (y_star - y_earth)) / r_star_earth ** 3
    )
    fy_star = (
        (G * m_sun * (y_sun - y_star)) / r_sun_star ** 3
    ) + (
        (-G * m_earth * (y_star - y_earth)) / r_star_earth ** 3
    )

    fx_earth = (
        (G * m_sun * (x_sun - x_earth)) / r_sun_earth ** 3
    ) + (
        (G * m_star * (x_star - x_earth)) / r_star_earth ** 3
    )
    fy_earth = (
        (G * m_sun * (y_sun - y_earth)) / r_sun_earth ** 3
    ) + (
        (G * m_star * (y_star - y_earth)) / r_star_earth ** 3
    )

    # Equations of motion for objects
    dx_sun_dt = vx_sun
    dy_sun_dt = vy_sun
    dvx_sun_dt = fx_sun
    dvy_sun_dt = fy_sun

    dx_earth_dt = vx_earth
    dy_earth_dt = vy_earth
    dvx_earth_dt = fx_earth
    dvy_earth_dt = fy_earth

    dx_star_dt = vx_star
    dy_star_dt = vy_star
    dvx_star_dt = fx_star
    dvy_star_dt = fy_star

    return [
        dx_sun_dt,
        dy_sun_dt,
        dvx_sun_dt,
        dvy_sun_dt,
        dx_star_dt,
        dy_star_dt,
        dvx_star_dt,
        dvy_star_dt,
        dx_earth_dt,
        dy_earth_dt,
        dvx_earth_dt,
        dvy_earth_dt
    ]


# Time points
t = np.linspace(0, 3.154e7 * 5, 5000)  # 5 years (seconds)

# Solving the ODE system
sol = odeint(three_body_system, x0, t)

# Extracting the positions of objects
x_sun, y_sun, _, _, x_star, y_star, _, _, x_earth, y_earth, _, _ = sol.T

# Plotting the orbits
plt.plot(x_sun / au, y_sun / au, label='Star 1')
plt.plot(x_star / au, y_star / au, label='Star 2')
plt.plot(x_earth / au, y_earth / au, label='Earth-like object')
plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.title('Three-Body System Orbits')
plt.legend()
plt.axis('equal')
plt.grid(True)
#plt.show()
plt.savefig('2heavy_onelight.png')

