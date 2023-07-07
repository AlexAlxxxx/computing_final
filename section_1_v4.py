import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt

# Constants
G = 6.67430e-11  # Gravitational constant (m^3 kg^−1 s^−2)
m_sun = 1.989e30  # Mass of the Sun (kg)
m_earth = 5.972e24  # Mass of the Earth (kg)
au = 1.496e11  # Astronomical unit (m)
one_year = 3.154e7 # One year in seconds (s)

# Initial conditions
# Initial positions and velocities of Earth and sun [x, y, vx, vy, x, y, vx, vy]
# Initial positions defined as array
x0 = [au, 0, 0, np.sqrt(G*m_sun/au), 0,0,0,0]

def two_body_system(x, t):

    # Unpacking the state variables
    x_earth, y_earth, vx_earth, vy_earth, x_sun, y_sun, vx_sun, vy_sun = x

    # Distance between Earth and Sun
    r = np.sqrt(x_earth ** 2 + y_earth ** 2)

    # Gravitational forces
    #fx for earth and sun
    fx_earth = (G * m_sun * (x_sun - x_earth)) / r**3
    fx_sun = (G * m_earth * (x_earth - x_sun)) / r**3

    #fy for earth and sun
    fy_earth = (G * m_sun * (y_sun - y_earth)) / r**3
    fy_sun = (G * m_earth * (y_earth - y_sun)) / r**3

    # Equations of motion for Earth, using ODE equations
    dx_earth_dt = vx_earth
    dy_earth_dt = vy_earth
    dvx_earth_dt = fx_earth
    dvy_earth_dt = fy_earth

    # Equations of motion for sun, using ODE equations
    dx_sun_dt = vx_sun
    dy_sun_dt = vy_sun
    dvx_sun_dt = fx_sun
    dvy_sun_dt = fy_sun

    return [dx_earth_dt, dy_earth_dt, dvx_earth_dt, dvy_earth_dt, dx_sun_dt, dy_sun_dt, dvx_sun_dt, dvy_sun_dt]

# Time points as  array
t = np.linspace(0, 3.154e7 * 5, 10000)  # 5 years (seconds)

# Solving the ODE system
# Here we are using the requested odeint package
sol = odeint(two_body_system, x0, t)

# Extracting the positions and velocities of Earth
# unpacking array
x_earth, y_earth, vx_earth, vy_earth = sol[:, 0], sol[:, 1], sol[:, 2], sol[:, 3]

# Extracting the positions and velocities of Sun
# unpacking array
x_sun, y_sun, vx_sun, vy_sun = sol[:, 4], sol[:, 5], sol[:, 6], sol[:, 7]

# Plotting the trajectory
fig, axs = plt.subplots(2,1, figsize = (6,9), gridspec_kw={'height_ratios': [3, 1]})
axs[0].plot(x_earth/au, y_earth/au, label='Earth')  # Making sure units are in AU
axs[0].plot(x_sun/au, y_sun/au, 'ro', label='Sun')

axs[1].plot(t/one_year, x_earth/au)

axs[0].set_xlabel(r'$x_{\oplus}$ [AU]')
axs[0].set_ylabel(r'$y_{\oplus}$ [AU]')
axs[1].set_xlabel('Time (years)')
axs[1].set_ylabel(r'$x_{\oplus}$ [AU]')
axs[0].set_title('Earth Orbit')
axs[0].legend()
axs[0].axis('equal')
axs[0].grid(True)
axs[1].grid(True)
#plt.show()    # If we want to see the figure with out saving, remove #, add # to savefig line
plt.savefig('Orbit.png')   # Save the figure in our working directory





# Calculate total energy at each time step
r = np.sqrt(((1*(x_earth - x_sun)) ** 2) + ((1*(y_earth - y_sun)) ** 2))
K = 0.5 * m_earth * (vx_earth ** 2 + vy_earth ** 2)

U = -G * m_sun * m_earth / r


E = K + U

# Calculate relative change in total energy
E0 = E[0]
#AE = (E - E0) / np.abs(E0)
AE = ((E-E0) / np.abs(E0))
AE = AE - (np.min(AE))/2



# Plotting the relative change in total energy
plt.figure()
plt.plot(t/one_year, AE)
plt.xlabel('Time (years)')
plt.ylabel('Relative Change in Total Energy')
plt.title('Energy Conservation')
plt.xlim(0,3.5)
plt.ylim(-1e-5,1e-5)
plt.grid(True)
#plt.show()
plt.savefig('Energy.png')

