#date:2024 11 16 scientific maching learning program, written by fuyi li
#this code is a simple application of guassian regression with a kernel function

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import best_paragroup_calculate as bpc

#guassian regression parameters
charactistic_len_scale = 1.0
expection = 0.0
sigma_f = 1.0 #prior distribution variance
sigma = 0.1 #regularizaiton coef

#processing random value of f(x)
np.random.seed(1) #make sure generate same value each time
x_interval_left = -8 
x_interval_right = 8
generate_points_num = 20
mean = np.array([0.0]*generate_points_num, dtype='float32')
delta_x = (x_interval_right - x_interval_left)/generate_points_num
# x = np.arange(x_interval_left, x_interval_right, delta_x)
x = np.random.uniform(x_interval_left, x_interval_right, size=(generate_points_num, 1)).flatten()
x_mesh, x_mesh_tranpose = np.meshgrid(x, x)
covariance_mat = sigma_f**2*np.exp(-(x_mesh - x_mesh_tranpose)**2/(2*charactistic_len_scale**2)) + sigma**2*np.eye(len(x))
# print(x.shape)
y = np.random.multivariate_normal(mean=mean, cov=covariance_mat, size=1)
y = np.squeeze(y)
print(y.shape)

#plot figure of initial generate random value
plt.plot(x, y, 'x', color='blue', alpha=0.5, label='prior value')
plt.title('prior value')
plt.xlabel('x axis')
plt.ylabel('y axis')
# plt.legend(loc='best')
plt.xlim(x_interval_left-0.5, x_interval_right+0.5)
plt.ylim(-5*sigma_f, 5*sigma_f)
# plt.savefig('./output/beyesian_regression/prior_value.png', dpi=300)
# print('prior value figure has been stored')

#posterior guassiam parameters, there are three groups:(1.0, 1.0, 0.1), (0.3, 1.08, 5e-5), (3.0, 1.16, 0.89)
best_theta = bpc.best_theta
post_characteristic_len_scale = best_theta[0]
post_sigma_f = best_theta[1]
post_sigma = best_theta[2]

#posterior estimation with given prior generated value
posterior_points_num = 2000
post_delta_x = (x_interval_right-x_interval_left)/posterior_points_num
x_posterior_value = np.arange(x_interval_left, x_interval_right, post_delta_x)
x = np.append(x, x_posterior_value)
print(x_posterior_value.shape)
x_mesh, x_mesh_tranpose = np.meshgrid(x, x) #to achieve x-x_hat and make vectorization calculation
n=15 #biggest n characteristic value and its vector
covariance_mat = post_sigma_f**2*np.exp(-(x_mesh - x_mesh_tranpose)**2/(2*post_characteristic_len_scale**2)) #+ post_sigma**2*np.eye(len(x))
covariance_mat_x_x = covariance_mat[0:generate_points_num, 0:generate_points_num] + post_sigma**2*np.eye(generate_points_num)
covariance_mat_x_xhat = covariance_mat[0:generate_points_num, generate_points_num-1:-1]
covariance_mat_xhat_x = covariance_mat[generate_points_num-1:-1, 0:generate_points_num]
covariance_mat_xhat_xhat = covariance_mat[generate_points_num-1:-1, generate_points_num-1:-1]
u,s,vt = sparse.linalg.svds(covariance_mat_x_x, n)
s = 1/s #calculate inverse of diag mat s
covariance_mat_inverse = vt.T@np.diag(s)@u.T
# print(covariance_mat_x_x.shape)
post_mean = covariance_mat_xhat_x@covariance_mat_inverse@y
post_covariance_mat = covariance_mat_xhat_xhat - covariance_mat_xhat_x@covariance_mat_inverse@covariance_mat_x_xhat
print(post_mean.shape)
print(post_covariance_mat.shape)
post_y = np.random.multivariate_normal(mean=post_mean, cov=post_covariance_mat, size=1)
post_y = np.squeeze(post_y)
# print(post_y.shape)

#plot the posterior estimation value figure
# plt.plot(x_posterior_value, post_y, '-', color='black', alpha=0.5, label='posterior value')
std_dev = np.sqrt(np.diag(post_covariance_mat))
plt.plot(x_posterior_value, post_mean, '-', color='red', alpha=0.6, label='mean value')
# plt.plot(X_test, mean_values, 'b-', lw=2, label='Mean prediction')  # 均值曲线
plt.fill_between(x_posterior_value.flatten(), post_mean - 2*std_dev, post_mean + 2*std_dev,
                 alpha=0.5, color='gray', label='double std var')  # 置信区间
plt.legend(loc='best')
# plt.savefig('./output/beyesian_regression/prior_and_posterior_value2.png', dpi=300)
# plt.savefig('./output/square_kernel_func/best_theta_guass_posterior.png', dpi=300)
plt.savefig('./output/martin_kernel_func/best_theta_guass_posterior.png', dpi=300)
# plt.show()
print('prior and posterior value figure has been stored')