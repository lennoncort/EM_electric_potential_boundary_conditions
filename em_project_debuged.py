# -*- coding: utf-8 -*-
"""
Created on Tue Mar  4 14:31:39 2025

@author: Pablo Herrador
"""

"Boundary Condition Problem. Potential in a Region"

import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import RegularGridInterpolator

# %% SETUP AND MATRIX PROBLEM
#Space of the Problem
delta_x = 0.002 #m
x_max = 0.14 #m
x_min = -0.14 #m
n_x = int((x_max - x_min) / delta_x)
x_points = np.linspace(x_min, x_max, n_x)

delta_y = 0.002 #m
y_max = 0.1 #m
y_min = -0.1 #m
n_y = int((y_max - y_min) / delta_y)
y_points = np.linspace(y_min, y_max, n_y)

potential_grid = np.zeros([n_y, n_x])
potential_fixed = np.zeros([n_y, n_x], dtype = bool)

potential_max = 5 #volts
ground = 0 #volts
potential_min = -5 #volts

#Bounday Conditions
## External Boundary
x_c, y_c = 0.0, 0.0 #m
r_inner = 0.098 #bm
r_outer = 0.109 #m

for i in range(n_y):
    for j in range(n_x):
        x, y = x_points[j], y_points[i]
        r = np.sqrt((x-x_c)**2 + (y-y_c)**2)
        
        if r_inner <= r <= r_outer:
            potential_grid[i,j] = ground
            potential_fixed[i,j] = True
            
## Internal Boundaries
###Vertical Bar
bar_x_min = -0.005 #m
bar_x_max = 0.005 #m
bar_y_min = -0.05 #m
bar_y_max = 0.05 #m

for i in range(n_y):
    for j in range(n_x):
        x, y = x_points[j], y_points[i]
        
        if bar_x_min <= x <= bar_x_max and bar_y_min <= y <= bar_y_max :
            potential_grid[i,j] = potential_max
            potential_fixed[i,j] = True
            
### Smal Circles
x_c_1, y_c_1 = 0.04 - 0.002, 0.0 - 0.002 #m
x_c_2, y_c_2 = -0.04 - 0.002, 0.0 - 0.002 #m
radius = 0.02/2 #m

for i in range(n_y):
    for j in range(n_x):
        x, y = x_points[j], y_points[i]
        r1 = np.sqrt((x-x_c_1)**2 + (y - y_c_1)**2)
        r2 = np.sqrt((x-x_c_2)**2 + (y - y_c_2)**2)
        
        if r1 <= radius or r2 <= radius:
            potential_grid[i,j] = potential_min #volts
            potential_fixed[i,j] = True
            
# %% VISUAL CHECK
# plt.figure(figsize=(6, 5))
# plt.imshow(potential_grid, extent=[x_min, x_max, y_min, y_max], origin='lower', cmap='grey')
# plt.colorbar(label="Potential (V)")
# plt.xlabel("x (m)")
# plt.ylabel("y (m)")
# plt.title("Potential Boundaries")
# plt.show()

# %% SOR
"Laplace Equation"

tolerance = 1e-4
or_factor = 0.9 
max_change =1.0 
coeff = (1.0 + or_factor) / 4.0
count = 0 

while max_change > tolerance:
    old_potential = np.copy(potential_grid)
    for i in range(1, n_y - 1):
        for j in range(1, n_x - 1):
            if not potential_fixed[i,j]:
                potential_grid[i,j] = (coeff * (potential_grid[i-1,j] +
                                                potential_grid[i+1,j] +
                                                potential_grid[i,j-1] +
                                                potential_grid[i,j+1]) -
                                               or_factor * potential_grid[i,j])
                max_change = np.max(np.abs(potential_grid - old_potential))
                count += 1
                if (count % 100000) == 0 :
                    print('Iteration Number:', count, 'Max Change:=', max_change)
                    
print('Iteration Number:', count, 'Max Change:=', max_change)
# %% VISUAL CHECK
# Plot the results
# plt.close('all')
# plt.pcolormesh(x_points, y_points, potential_grid, cmap='inferno')
# plt.gca().set_aspect('equal')
# plt.colorbar()
# plt.xlabel('X (m)')
# plt.ylabel('Y (m)')
# plt.tight_layout()

# plt.figure()
# levels = np.linspace(potential_min, potential_max, 25)
# plt.contour(x_points, y_points, potential_grid, levels, cmap='hsv')
# plt.gca().set_aspect('equal')
# plt.colorbar()
# plt.xlabel('X (m)')
# plt.ylabel('Y (m)')
# plt.tight_layout()

# X_grid, Y_grid = np.meshgrid(x_points, y_points)
# fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
# ax.plot_surface(X_grid, Y_grid, potential_grid, cmap='inferno')
# plt.xlabel('X (m)')
# plt.ylabel('Y (m)')
# ax.set_zlabel('Potential (V)')

# plt.show()
# %% EXPERIMENTAL DATA
data_exp = np.loadtxt("C:/Users/lenno/Desktop/Physics 24-25/Spring/EM/Projects/Projects/data_1.txt", delimiter=";")
data_dim = data_exp.shape
n_y_exp, n_x_exp = data_dim
x_exp_points = np.linspace(-0.14, 0.14, n_x_exp)
y_exp_points = np.linspace(-0.1, 0.1, n_y_exp)
# %% VISUAL CHECK 2D & 3D
##2D heat map
# plt.figure(figsize=(6, 5))
# plt.imshow(data_exp, extent=[-0.14, 0.14, -0.1, 0.1], origin='lower', cmap='turbo')
# plt.colorbar(label="Experimental Potential (V)")
# plt.xlabel("x (m)")
# plt.ylabel("y (m)")
# plt.title("Experimental Potential Map")
# plt.show()

## 3D Surface Plot
# X_exp_grid, Y_exp_grid = np.meshgrid(x_exp_points, y_exp_points)

# fig = plt.figure(figsize=(8, 6))
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(X_exp_grid, Y_exp_grid, data_exp, cmap='turbo')

# ax.set_xlabel("X (m)")
# ax.set_ylabel("Y (m)")
# ax.set_zlabel("Potential (V)")
# ax.set_title("3D Experimental Potential Surface")

# plt.show()
# %% Comparission Model vs Experimental Data
data_model = potential_grid
data_model_dim = data_model.shape
## Heat Map 2D
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

im1 = axes[0].imshow(data_exp, cmap='inferno', aspect='auto')
axes[0].set_title("Experimental Data")
fig.colorbar(im1, ax=axes[0])

im2 = axes[1].imshow(data_model, cmap='inferno', aspect='auto')
axes[1].set_title("Model")
fig.colorbar(im2, ax=axes[1])

plt.show()

#3D and Interpolation
x_model = np.linspace(x_min, x_max, data_model.shape[1])
y_model = np.linspace(y_min, y_max, data_model.shape[0])

# Interpolation function
X_exp_grid, Y_exp_grid = np.meshgrid(x_exp_points, y_exp_points)
interp_model = RegularGridInterpolator((y_model, x_model), data_model, method='linear', bounds_error=False, fill_value=None)


data_model_interpol = interp_model((Y_exp_grid, X_exp_grid))  # Usar las coordenadas de la malla experimental

# Shape of the result. We need dim = (21,29) = data_ext.shape
print(data_model_interpol.shape)
fig = plt.figure(figsize=(12, 6))

# Gráfico 3D de datos experimentales
ax1 = fig.add_subplot(121, projection='3d')
ax1.plot_surface(X_exp_grid, Y_exp_grid, data_exp, cmap='inferno')
ax1.set_title("Experimental Data")

# Gráfico 3D del modelo
ax2 = fig.add_subplot(122, projection='3d')
ax2.plot_surface(X_exp_grid, Y_exp_grid, data_model_interpol, cmap='inferno')
ax2.set_title("Model Data")

plt.show()

# print(data_exp[13,17])
# print(data_model_interpol[13,17])
# %% ERROR
error_absolute = np.abs(data_exp - data_model_interpol)
rmse = np.sqrt(np.mean((data_exp - data_model_interpol)**2))

print("RMSE:", rmse)

# Visualization
plt.figure(figsize=(6, 5))
plt.imshow(error_absolute, extent=[-0.14, 0.14, -0.1, 0.1], origin='lower', cmap='inferno')
plt.colorbar(label="Absolute Error (V)")
plt.xlabel("x (m)")
plt.ylabel("y (m)")
plt.title("Absolute Error")
plt.show()

# 3D Abs Error
X_exp_grid, Y_exp_grid = np.meshgrid(x_exp_points, y_exp_points)

fig = plt.figure(figsize=(8, 6))
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(X_exp_grid, Y_exp_grid, error_absolute, cmap='inferno')

ax.set_xlabel("X (m)")
ax.set_ylabel("Y (m)")
ax.set_zlabel("Absolute Error (V)")
ax.set_title("3D Absolute Error")

plt.show()