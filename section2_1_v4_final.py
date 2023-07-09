import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^−1 s^−2)
m_sun = 1.989e30  # Mass of the Sun (kg)
m_earth = 5.972e24  # Mass of the Earth (kg)
m_mars = 6.39e23  # Mass of Mars (kg)
au = 1.496e11  # Astronomical unit (m)
one_year = 3.154e7  # One year in seconds (s)

# Initial conditions as array
x0 = [
    0, 0, 0, 0,  # Initial conditions for Sun [x, y, vx, vy]
    0.3*au, 0, 0, np.sqrt(G * m_sun / au),  # Initial conditions for Light 1 earth [x, y, vx, vy]
    0.4*au, 0, 0, np.sqrt(G * m_sun / au),  # Initial conditions for Light 2 mars [x, y, vx, vy]
]


def three_body_system(x, t):
    # Unpacking the state variables
    x_sun, y_sun, vx_sun, vy_sun, x_earth, y_earth, vx_earth, vy_earth, x_mars, y_mars, vx_mars, vy_mars = x

    # Distance between objects
    r_sun_earth = np.sqrt((x_sun - x_earth) ** 2 + (y_sun - y_earth) ** 2)
    r_sun_mars = np.sqrt((x_sun - x_mars) ** 2 + (y_sun - y_mars) ** 2)
    r_earth_mars = np.sqrt((x_earth - x_mars) ** 2 + (y_earth - y_mars) ** 2)

    # Gravitational forces
    fx_sun = (
        (-G * m_earth * (x_sun - x_earth)) / r_sun_earth ** 3
    ) + (
        (-G * m_mars * (x_sun - x_mars)) / r_sun_mars ** 3
    )
    fy_sun = (
        (-G * m_earth * (y_sun - y_earth)) / r_sun_earth ** 3
    ) + (
        (-G * m_mars * (y_sun - y_mars)) / r_sun_mars ** 3
    )

    fx_earth = (
        (G * m_sun * (x_sun - x_earth)) / r_sun_earth ** 3
    ) + (
        (-G * m_mars * (y_earth - y_mars)) / r_earth_mars ** 3
    )
    fy_earth = (
        (G * m_sun * (y_sun - y_earth)) / r_sun_earth ** 3
    ) + (
        (-G * m_mars * (y_earth - y_mars)) / r_earth_mars ** 3
    )

    fx_mars = (
        (G * m_sun * (x_sun - x_mars)) / r_sun_mars ** 3
    ) + (
        (G * m_earth * (x_earth - x_mars)) / r_earth_mars ** 3
    )
    fy_mars = (
        (G * m_sun * (y_sun - y_mars)) / r_sun_mars ** 3
    ) + (
        (G * m_earth * (y_earth - y_mars)) / r_earth_mars ** 3
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

    dx_mars_dt = vx_mars
    dy_mars_dt = vy_mars
    dvx_mars_dt = fx_mars
    dvy_mars_dt = fy_mars

    return [
        dx_sun_dt,
        dy_sun_dt,
        dvx_sun_dt,
        dvy_sun_dt,
        dx_earth_dt,
        dy_earth_dt,
        dvx_earth_dt,
        dvy_earth_dt,
        dx_mars_dt,
        dy_mars_dt,
        dvx_mars_dt,
        dvy_mars_dt,
    ]


# Time points as array
t = np.linspace(0, 3.154e7 * 5, 5000)  # 5 years (seconds)

# Solving the ODE system
sol = odeint(three_body_system, x0, t)

# Extracting the positions of objects
x_sun, y_sun, _, _, x_earth, y_earth, _, _, x_mars, y_mars, _, _ = sol.T

# Plotting the orbits
plt.plot(x_sun / au, y_sun / au,'o',markersize=4, color='red', label='Sun')
plt.plot(x_earth / au, y_earth / au, label='Earth-like')
plt.plot(x_mars / au, y_mars / au, label='Mars-like')
plt.xlabel('x [AU]')
plt.ylabel('y [AU]')
plt.title('Three-Body System Orbits')
plt.legend(loc='upper right')
plt.axis('equal')
plt.grid(True)
#plt.show()
plt.savefig('sun_2light.png')

