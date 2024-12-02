# date: 2024 11 17 written by fuyi li , beyesian optimization regression program: a program in scientific maching learning course
# this program using beyesian estimation to calculate parameter liklyhood and choosing the best parameter group

import numpy as np
import jax.numpy as jnp
import jax
from jax import jit, grad, vmap
import time
from jax.scipy.linalg import inv
from jax.scipy.linalg import svd
from scipy.optimize import minimize
import matplotlib.pyplot as plt
import math

# Global define
CHARACTERISTIC_LEN = 1.0
SIGMA_L = 1.0
SIGMA = 0.1
np.random.seed(1)

x_interval_left = -8.0
x_interval_right = 8.0
generated_points_num = 20
x = np.random.uniform(x_interval_left, x_interval_right, size=(20, 1)).flatten()
x_mesh, x_mesh_transpose = np.meshgrid(x, x, indexing='ij')
initial_covariance_mat = SIGMA_L**2 * np.exp(-(x_mesh - x_mesh_transpose)**2 / (2 * CHARACTERISTIC_LEN**2)) + SIGMA**2 * np.eye(len(x))
initial_mean = np.array([0.0] * len(x), dtype='float64')
prior_y = np.random.multivariate_normal(initial_mean, initial_covariance_mat, size=1).flatten()
# plt.plot(x, prior_y, 'x')
# plt.show()

# Convert to JAX arrays
x_jax = jnp.array(x)
x_mesh_jax, x_mesh_transpose_jax = jnp.meshgrid(x_jax, x_jax, indexing='ij')
initial_mean_jax = jnp.array(initial_mean)
prior_y_jax = jnp.array(prior_y)

def covariance_mat_calculation(theta):
    char_len = theta[0]
    sigma_f = theta[1]
    sigma = theta[2]
    covariance_mat = sigma_f**2 * jnp.exp(-(x_mesh_jax - x_mesh_transpose_jax)**2 / (2 * char_len**2)) + sigma**2 * jnp.eye(len(x_jax)) #square kernel func
    # abs_x_i_minus_x_j = jnp.abs(x_mesh_jax - x_mesh_transpose_jax)
    # covariance_mat = sigma_f**2*(1+jnp.sqrt(3)/char_len*abs_x_i_minus_x_j)*jnp.exp(-jnp.sqrt(3)/char_len*abs_x_i_minus_x_j) + sigma**2*jnp.eye(len(x_jax)) #martin kernel func
    # print('present max value of mat is:')
    # print([theta[0].item(), theta[1].item(), theta[2].item()])
    # print(jnp.max(covariance_mat).item())
    return covariance_mat

def optimization_object(theta):
    covariance_mat = covariance_mat_calculation(theta)
    y_minus_mean = prior_y_jax - initial_mean_jax
    inverse_covariance_mat = inv(covariance_mat)
    # u,s,vt = svd(covariance_mat)
    # s = 1/s
    # inverse_covariance_mat = vt.T @ jnp.diag(s) @ u.T
    sign, log_complexity = jnp.linalg.slogdet(covariance_mat)
    data_match_term = y_minus_mean @ inverse_covariance_mat @ y_minus_mean.T
    optimized_project = data_match_term + sign*log_complexity
    # print((sign*log_complexity).item())
    # print(data_match_term.item())
    return optimized_project

gradient = grad(optimization_object)

def gradient_jnp(theta):
    x_jax = jnp.array(theta)
    grad_jax = gradient(x_jax)
    return np.array(grad_jax)

def get_levels(value_mat):
    max_value = np.max(value_mat)
    min_value = np.min(value_mat)
    # levels = np.concatenate([np.linspace(max_value, max_value - 0.1 * (max_value - min_value), 10), 
                        #  np.linspace(max_value - 0.1 * (max_value - min_value), min_value, 5)])
    levels = np.concatenate([np.linspace(min_value, max_value - 0.05*(max_value - min_value), 8)
                            ,np.linspace(max_value - 0.05*(max_value - min_value)-(1e-10), max_value, 10)])
    levels = np.unique(levels)
    return levels

start_time = time.time()
print('Program begins')

# Define the bounds for the parameters
parabounds = [(0.1, 10), (1e-2, 5), (1e-2, 10)]

# Generate multiple initial states
num_initial_states = 50
initial_states = np.random.uniform(low=[1, 1e-2, 1e-2], high=[2, 1, 1], size=(num_initial_states, 3))

# Perform optimization from multiple initial states
best_result = None
best_value = float('inf')

for initial_state in initial_states:
    result = minimize(optimization_object, initial_state, method='BFGS', jac=gradient_jnp)
    if result.fun < best_value:
        best_value = result.fun
        best_result = result
        print('present optimization objection value is:')
        print(best_value)
        print(best_result.x[0], best_result.x[1], best_result.x[2])

print("Best parameter group:", best_result.x)
print("Best objective value:", best_result.fun)
print("Optimization time:", time.time() - start_time, "seconds")

# with determined spirit and keep trying, we find the best iteration method is conjudgate gradient method

# fixed one para each time and get three para table
best_theta = best_result.x
[best_char_len, best_sigma_f, best_sigma] = best_theta
left_interval = 5*1e-2
mid_intervel = 1
right_interval = 10
char_len_list = sigma_f_list = sigma_list =np.hstack((np.arange(left_interval, mid_intervel, 1*1e-2),np.arange(mid_intervel, right_interval, 0.1)))
test_num = len(char_len_list)
sigma_f_vs_sigma_plane = np.zeros((test_num, test_num), dtype='float64')
char_len_vs_sigma = np.copy(sigma_f_vs_sigma_plane)
char_len_vs_sigma_f = np.copy(sigma_f_vs_sigma_plane)

# calculate data used for contour only run as main program
if __name__ == "__main__":
    for i in range(test_num):
        for j in range(test_num):
            sigma_f_vs_sigma_plane[i][j] = -(optimization_object([best_char_len, sigma_f_list[j], sigma_list[i]])+generated_points_num*np.log(2*math.pi))/2
            char_len_vs_sigma[i][j] = -(optimization_object([char_len_list[j], best_sigma_f, sigma_list[i]])+generated_points_num*np.log(2*math.pi))/2
            char_len_vs_sigma_f[i][j] = -(optimization_object([char_len_list[j], sigma_f_list[i], best_sigma])+generated_points_num*np.log(2*math.pi))/2
    print('log likelyhood calculation has been done')
    
    #save log marginal likelyhood as file
    file_path = './output/martin_kernel_func/'
    data_name = ['sigmaf_vs_sigma.txt', 'char_len_vs_sigma.txt', 'char_len_vs_sigma_f.txt']
    file = [file_path+data_name[i] for i in range(3)]
    with open(file[0], 'w') as f:
        np.savetxt(f, sigma_f_vs_sigma_plane, fmt='%f')
    f.close()
    with open(file[1], 'w') as f:
        np.savetxt(f, char_len_vs_sigma, fmt='%f')
    f.close()
    with open(file[2], 'w') as f:
        np.savetxt(f, char_len_vs_sigma_f, fmt='%f')
    f.close()
    print("all data has been stored! that's all")





    # # Plot figure: contour map with one parameter fixed

    # # plot plane value on sigma_f vs sigma 
    # sigma_f_sigma_mesh1, sigma_f_sigma_mesh2 = np.meshgrid(sigma_f_list, sigma_list)
    # char_len_sigma_mesh1, char_len_sigma_mesh2 = np.meshgrid(char_len_list, sigma_list)
    # char_len_sigma_f_mesh1, char_len_sigma_f_mesh2 = np.meshgrid(char_len_list, sigma_f_list)
    # levels = get_levels(sigma_f_vs_sigma_plane)
    # # contours = plt.contour(sigma_f_sigma_mesh1, sigma_f_sigma_mesh2, sigma_f_vs_sigma_plane, cmap='rainbow', linewidths=1, levels=levels)
    # contours = plt.contour(sigma_f_sigma_mesh1, sigma_f_sigma_mesh2, sigma_f_vs_sigma_plane, cmap='rainbow', linewidths=1, norm=plt.matplotlib.colors.LogNorm(vmin=sigma_f_vs_sigma_plane.min(), vmax=sigma_f_vs_sigma_plane.max()))
    
    # plt.clabel(contours, inline=True, fontsize=8)
    # plt.title('log likelyhood on sigma_f vs sigma plane')
    # plt.xlabel(r'$\sigma_f$')
    # plt.ylabel(r'$\sigma$')
    # # plt.colorbar(label='log likelyhood')
    # plt.xscale('log')
    # plt.yscale('log')
    # plt.savefig('./output/beyesian_regression/log_likelyhood_on_sigma_vs_siamg_f_plane.png', dpi=300)
    # plt.close()

    # # plot plane value on char_len vs sigma
    # levels = get_levels(char_len_vs_sigma)
    # contours = plt.contour(char_len_sigma_mesh1, char_len_sigma_mesh2, char_len_vs_sigma, cmap='rainbow', linewidths=1, levels=levels)
    # plt.clabel(contours, inline=True, fontsize=8)
    # plt.title('log likelyhood on charlen vs sigma plane')
    # plt.xlabel('characteristic len scale')
    # plt.ylabel(r'\sigma')
    # plt.xscale('log')
    # plt.yscale('log')
    # # plt.colorbar(label='log likelyhood')
    # plt.savefig('./output/beyesian_regression/log_likelyhood_on_charlen_vs_sigma_plane.png', dpi=300)
    # plt.close()

    # # plot plane value on char_len vs sigma_f
    # levels = get_levels(char_len_vs_sigma_f)
    # contours = plt.contour(char_len_sigma_f_mesh1, char_len_sigma_f_mesh2, char_len_vs_sigma_f, cmap='rainbow', linewidths=1, levels=levels)
    # plt.clabel(contours, inline=True, fontsize=8)
    # plt.title('log likelyhood on charlen vs sigma_f plane')
    # plt.xlabel('characteristic len scale')
    # plt.ylabel(r'\sigma_f')
    # plt.xscale('log')
    # plt.yscale('log')
    # # plt.colorbar(label='log likelyhood')
    # plt.savefig('./output/beyesian_regression/log_likelyhood_on_charlen_vs_sigma_f_plane.png', dpi=300)
    # plt.close()

    # end_time = time.time()
    # print(f'all work has been done, Total time cost is: {end_time-start_time:.3f}')

