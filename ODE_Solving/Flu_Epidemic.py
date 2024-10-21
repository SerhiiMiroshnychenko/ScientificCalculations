import numpy as np
from scipy.integrate import odeint
import matplotlib.pyplot as plt


# Define the SIR model function
def sir_model(Y, t, r=0.00218, a=0.5):
    S, I, R = Y

    # Define the differential equations
    dSdt = -r * S * I
    dIdt = r * S * I - a * I
    dRdt = a * I

    return [dSdt, dIdt, dRdt]


# Set up the time range
t_range = np.linspace(0, 14, 1000)  # 1000 points from 0 to 14 days

# Initial conditions
Y0 = [999, 1, 0]  # Initial S, I, R populations

# Solve the ODE system
solution = odeint(sir_model, Y0, t_range)

# Extract S, I, R from the solution
S = solution[:, 0]
I = solution[:, 1]
R = solution[:, 2]

# Plot S, I, R over time
plt.figure(figsize=(10, 6))
plt.plot(t_range, S, label='Susceptible')
plt.plot(t_range, I, label='Infected')
plt.plot(t_range, R, label='Recovered')
plt.xlabel('Time (days)')
plt.ylabel('Population')
plt.legend()
plt.grid(True)
plt.show()

# Find maximum infected population and when it occurred
max_I = np.max(I)
peak_flu = t_range[np.argmax(I)]
print(f"Maximum infected population: {max_I:.2f}")
print(f"Peak occurred at day: {peak_flu:.2f}")

# Find number of people who never got flu
no_flu = S[-1]
print(f"People who never got flu: {no_flu:.2f}")

# Check population conservation
pop_error = np.sum(solution, axis=1) - 1000
plt.figure(figsize=(10, 4))
plt.plot(t_range, pop_error)
plt.xlabel('Time (days)')
plt.ylabel('Population Error')
plt.title('Population Conservation Error')
plt.grid(True)
plt.show()

# Check population conservation with rounded numbers
pop_error_round = np.sum(np.round(solution), axis=1) - 1000
plt.figure(figsize=(10, 4))
plt.plot(t_range, pop_error_round)
plt.xlabel('Time (days)')
plt.ylabel('Population Error (Rounded)')
plt.title('Population Conservation Error (Rounded Numbers)')
plt.grid(True)
plt.show()
