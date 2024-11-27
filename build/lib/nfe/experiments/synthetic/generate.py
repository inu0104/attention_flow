import numpy as np
import scipy.signal
import scipy.integrate
from pathlib import Path

DATA_DIR = Path(__file__).parents[1] / 'data/synth'
DATA_DIR.mkdir(parents=True, exist_ok=True)

np.random.seed(123)

NUM_SEQUENCES = 1000
NUM_POINTS = 100
MAX_TIME = 10
EXTRAPOLATION_TIME = 20

def get_inital_value(extrap_space):
    if extrap_space:
        return np.random.uniform(-4, -2, (1,)) if np.random.rand() > 0.5 else np.random.uniform(2, 4, (1,))
    else:
        return np.random.uniform(-2, 2, (1,))

def get_inital_value2d(extrap_space):
    if extrap_space:
        return np.random.uniform(1, 2, (2,))
    else:
        return np.random.uniform(0, 1, (2,))

def get_data(func, time_min, time_max, NF=NUM_POINTS, extrap_space=False, name=None, NU=NUM_SEQUENCES,):
    initial_values = []
    times = []
    sequences = []

    for _ in range(NU):
        t = np.sort(np.random.uniform(time_min, time_max, NF))
        y0, y = func(t, extrap_space)
        times.append(t)
        initial_values.append(y0)
        sequences.append(y)

    initial_values, times, sequences = np.array(initial_values), np.array(times), np.array(sequences)
    if name is None:
        return initial_values, times, sequences
    else:
        np.savez(DATA_DIR / f'{name}.npz', init=initial_values, seq=sequences, time=times)

def generate(NU=None, NF=None):
    # SINE
    def sine_func(t, extrap_space=False):
        y = get_inital_value(extrap_space)
        return y, np.sin(t[:,None]) + y
    if not (DATA_DIR / 'sine.npz').exists():
        get_data(sine_func, 0, MAX_TIME, name='sine')
        get_data(sine_func, MAX_TIME, MAX_TIME + EXTRAPOLATION_TIME, name='sine_extrap_time')
        get_data(sine_func, 0, MAX_TIME, extrap_space=True, name='sine_extrap_space')

    # SQUARE
    def square_func(t, extrap_space=False):
        y = get_inital_value(extrap_space)
        return y, np.sign(np.sin(t[:,None])) + y
    if not (DATA_DIR / 'square.npz').exists():
        get_data(square_func, 0, MAX_TIME, name='square')
        get_data(square_func, MAX_TIME, MAX_TIME + EXTRAPOLATION_TIME, name='square_extrap_time')
        get_data(square_func, 0, MAX_TIME, extrap_space=True, name='square_extrap_space')

    # SAWTOOTH
    def sawtooth_func(t, extrap_space=False):
        y = get_inital_value(extrap_space)
        return y, scipy.signal.sawtooth(t[:,None]) + y
    if not (DATA_DIR / 'sawtooth.npz').exists():
        get_data(sawtooth_func, 0, MAX_TIME, name='sawtooth')
        get_data(sawtooth_func, MAX_TIME, MAX_TIME + EXTRAPOLATION_TIME, name='sawtooth_extrap_time')
        get_data(sawtooth_func, 0, MAX_TIME, extrap_space=True, name='sawtooth_extrap_space')

    # TRIANGLE
    def triangle_func(t, extrap_space=False):
        y = get_inital_value(extrap_space)
        return y, np.abs(scipy.signal.sawtooth(t[:,None])) + y
    if not (DATA_DIR / 'triangle.npz').exists():
        get_data(triangle_func, 0, MAX_TIME, name='triangle')
        get_data(triangle_func, MAX_TIME, MAX_TIME + EXTRAPOLATION_TIME, name='triangle_extrap_time')
        get_data(triangle_func, 0, MAX_TIME, extrap_space=True, name='triangle_extrap_space')


    # SINK
    def sink_func(t, extrap_space=False):
        y = get_inital_value2d(extrap_space)
        ode = lambda y, t: np.array([[-4, 10], [-3, 2]]) @ y
        return y, scipy.integrate.odeint(ode, y, t)
    if not (DATA_DIR / 'sink.npz').exists():
        get_data(sink_func, 0, MAX_TIME, name='sink')
        get_data(sink_func, MAX_TIME, MAX_TIME + EXTRAPOLATION_TIME, name='sink_extrap_time')
        get_data(sink_func, 0, MAX_TIME, extrap_space=True, name='sink_extrap_space')

    # ELLIPSE (Lotka-Volterra)
    def ellipse_func(t, extrap_space=False):
        y = get_inital_value2d(extrap_space)
        ode = lambda y, t: np.array([2/3 * y[0] - 2/3 * y[0] * y[1], y[0] * y[1] - y[1]])
        return y, scipy.integrate.odeint(ode, y, t) - 1
    if not (DATA_DIR / 'ellipse.npz').exists():
        get_data(ellipse_func, 0, MAX_TIME, name='ellipse')
        get_data(ellipse_func, MAX_TIME, MAX_TIME + EXTRAPOLATION_TIME, name='ellipse_extrap_time')
        get_data(ellipse_func, 0, MAX_TIME, extrap_space=True, name='ellipse_extrap_space')
    
    # heat
    def heat_func(t, extrap_space=False):
        """
        Solve the Heat equation (1D) using finite difference with time-stepping.
        """
        nu = 0.01  # Diffusion coefficient
        L = 1.0  # Domain length
        N = 100  # Number of spatial points
        dx = L / N  # Spatial step size
        dt = 0.0001  # Time step size
        x = np.linspace(0, L, N)  # Spatial grid

        # Initial condition: u(x, 0) = sin(pi * x)
        u = np.sin(np.pi * x)

        # Add noise to the initial condition
        noise = np.random.normal(0, 0.1, u.shape)
        u += noise  # Add random noise to the initial condition

        # Precompute coefficient for finite difference
        alpha = nu * dt / dx**2

        # Time-stepping loop for heat equation
        solutions = [u.copy()]  # Store solutions for all time steps
        for time in t[1:]:
            u_new = u.copy()
            for i in range(1, N - 1):
                # Finite difference update
                u_new[i] = u[i] + alpha * (u[i + 1] - 2 * u[i] + u[i - 1])
            u = u_new
            solutions.append(u.copy())

        return solutions[0], np.array(solutions)
    if not (DATA_DIR / 'heat.npz').exists():
        get_data(heat_func, 0, MAX_TIME, name='heat')
        get_data(heat_func, MAX_TIME, MAX_TIME + EXTRAPOLATION_TIME, name='heat_extrap_time')
        get_data(heat_func, 0, MAX_TIME, extrap_space=True, name='heat_extrap_space')
        
    # BURGERS (1D Burgers' Equation)
    def burgers_func(t, extrap_space=False):
        # Set up the spatial grid
        L_x = 10
        dx = 0.02
        N_x = int(L_x / dx)
        x = np.linspace(0, L_x, N_x)
        
        # Initial condition: Gaussian wave
        center = get_inital_value(extrap_space)[0]
        y = np.exp(-(x - center)**2 / 2)
        
        noise = np.random.normal(0, 0.1, y.shape)  # Random noise
        y += noise  # Add the noise to the wave

        
        # Define Burgers' equation system
        def burg_system(y, t, k, mu, nu):
            y_hat = np.fft.fft(y)
            y_hat_x = 1j * k * y_hat
            y_hat_xx = -k**2 * y_hat

            y_x = np.fft.ifft(y_hat_x)
            y_xx = np.fft.ifft(y_hat_xx)

            dydt = -mu * y * y_x.real + nu * y_xx.real
            return dydt

        # Set up Fourier wave numbers
        k = 2 * np.pi * np.fft.fftfreq(N_x, d=dx)
        mu, nu = 1, 0.01  # Nonlinear and viscosity coefficients

        # Solve Burgers' equation
        y_t = scipy.integrate.odeint(burg_system, y, t, args=(k, mu, nu), mxstep=5000)
        return y, y_t
    if not (DATA_DIR / 'burgers.npz').exists():
        get_data(burgers_func, 0, MAX_TIME, NU=NU, NF=NF, name='burgers')
        get_data(burgers_func, MAX_TIME, MAX_TIME + EXTRAPOLATION_TIME, NU=NU, NF=NF, name='burgers_extrap_time')
        get_data(burgers_func, 0, MAX_TIME, NU=NU, NF=NF, extrap_space=True, name='burgers_extrap_space')

if __name__ == '__main__':
    generate()