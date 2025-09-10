import numpy as np
from scipy.optimize import minimize

g = 9.81
rho = 1.225
m = 0.057
D = 0.067
A = np.pi * (D/2)**2
Cd = 0.55

def drag_force(v):
    return 0.5 * Cd * rho * A * v**2

def equations(t, state):
    x, y, vx, vy = state
    v = np.sqrt(vx**2 + vy**2)
    if v == 0:
        drag_x = 0
        drag_y = 0
    else:
        Fd = drag_force(v)
        drag_x = -Fd * (vx/v) / m
        drag_y = -Fd * (vy/v) / m
    return np.array([vx, vy, drag_x, -g + drag_y])

def rk4_step(f, t, state, dt):
    k1 = f(t, state)
    k2 = f(t + dt/2, state + dt*k1/2)
    k3 = f(t + dt/2, state + dt*k2/2)
    k4 = f(t + dt, state + dt*k3)
    return state + dt*(k1 + 2*k2 + 2*k3 + k4)/6

def simulate_projectile(v0, theta, h0, max_time=5, dt=0.001):
    vx0 = v0 * np.cos(theta)
    vy0 = v0 * np.sin(theta)
    state = np.array([0, h0, vx0, vy0])
    t = 0
    states = [state]
    times = [0]

    while t < max_time:
        state = rk4_step(equations, t, state, dt)
        t += dt
        states.append(state)
        times.append(t)
        if state[1] <= 0:
            break

    states = np.array(states)
    times = np.array(times)

    # Linear interpolate landing time and distance for better accuracy
    y = states[:,1]
    x = states[:,0]
    if y[-1] < 0 and y[-2] > 0:
        t_land = times[-2] + (times[-1] - times[-2]) * (0 - y[-2]) / (y[-1] - y[-2])
        x_land = x[-2] + (x[-1] - x[-2]) * (0 - y[-2]) / (y[-1] - y[-2])
    else:
        t_land = times[-1]
        x_land = x[-1]

    return t_land, x_land

def objective(vars, h0, t_target, d_target):
    v0, theta = vars
    if v0 < 0 or theta < 0 or theta > np.pi/2:
        return 1e6
    t_land, x_land = simulate_projectile(v0, theta, h0)
    time_error = (t_land - t_target)**2
    dist_error = (x_land - d_target)**2
    # Uncomment for debugging:
    # print(f"v0={v0:.2f}, theta={np.degrees(theta):.2f}°, t_land={t_land:.3f}, x_land={x_land:.3f}")
    return time_error + dist_error

def find_initial_speed_and_angle(h0, t_flight, d, initial_guess=(30, np.radians(10))):
    res = minimize(objective, initial_guess, args=(h0, t_flight, d), 
                   bounds=[(0, 100), (0, np.pi/2)], method='L-BFGS-B',
                   options={'ftol':1e-8, 'maxiter':1000})
    if res.success:
        v0, theta = res.x
        return float(v0), float(np.degrees(theta))  # Convert to regular Python floats
    else:
        raise RuntimeError("Optimization failed")

if __name__ == "__main__":
    h0 = 1.5
    t_flight = 1.2333333333333334
    d = 27.050416946411133
    v0, angle_deg = find_initial_speed_and_angle(h0, t_flight, d)
    print(f"Initial speed: {v0:.2f} m/s, Launch angle: {angle_deg:.2f} degrees")
