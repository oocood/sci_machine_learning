#date: 2024 11 17 written by fuyi li , beyesian optimization regression program: a program in scientific maching learning course
#this program using beyesian estimation to calculate parameter liklyhood and choosing the best parameter group

import numpy as np
import matplotlib.pyplot as plt
from scipy import sparse
import time

#Global define
CHARACTERISTIC_LEN = 1.0
SIGMA_L = 1.0
SIGMA = 0.1
np.random.seed(1)

def inverse_mat_svd(mat, truncated_num):
    u, s, vt = sparse.linalg.svds(mat, truncated_num)
    s = 1/s
    inverse_mat = vt.T@np.diag(s)@u.T
    return inverse_mat

class likely_hood:
    def __init__(self, x, y):
        self.prior_x = x
        self.prior_y = y
        self.characteristic_len_scale = 1.0
        self.sigma_l = 1.0
        self.sigma = 0.1
        print('likelyhood class initialization has been done!')
    def covariance_mat_calculate(self):
        x_mesh, x_mesh_transpose = np.meshgrid(self.prior_x, self.prior_x)
        mat = self.sigma_l**2*np.exp(-(x_mesh - x_mesh_transpose)**2/(2*self.characteristic_len_scale**2))+self.sigma**2*np.eye(len(self.prior_x))
        return mat
    def mean_array_calculate(self):
        mean = np.array([0.0]*len(self.prior_x), dtype='float32')
        return mean
    def likelyhood_calculate(self, estimation_para): #make two of them fixed and change one of them each time
        self.characteristic_len_scale, self.sigma_l, self.sigma = estimation_para
        covariance_mat = self.covariance_mat_calculate()
        inverse_covariance_mat = inverse_mat_svd(covariance_mat, int(len(self.prior_x)*0.75))
        mean = self.mean_array_calculate()
        y_minus_mean_column = (self.prior_y-mean)
        # print(y_minus_mean_column.shape)
        log_likelyood = -y_minus_mean_column@inverse_covariance_mat@y_minus_mean_column.T
        # print(log_likelyood.shape)
        return np.squeeze(log_likelyood)

if __name__ == "__main__": #start as main program
    start_time = time.time()
    print('program begins')
    x_interval_left = -8.0
    x_interval_right = 8.0
    generated_points_num = 20
    delta_x = (x_interval_right - x_interval_left)/generated_points_num
    x = np.arange(x_interval_left, x_interval_right, delta_x)
    based_beyesian_optimization = likely_hood(x, x)
    initial_covariance_mat = based_beyesian_optimization.covariance_mat_calculate()
    initial_mean = based_beyesian_optimization.mean_array_calculate()
    prior_y = np.random.multivariate_normal(initial_mean, initial_covariance_mat, size=1) #generate initial points y
    
    #create class according to the upper text
    beyesian_optimizaiton = likely_hood(x, prior_y)
    delta_para = 1e-3
    characteristic_len_scale_list = np.arange(0+delta_para, 5, delta_para)
    sigma_l_list = np.arange(0+delta_para, 100, delta_para)
    sigma_list = np.arange(0+delta_para, 100, delta_para)
    char_len_scale_len = len(characteristic_len_scale_list)
    sigma_l_len = len(sigma_l_list)
    sigma_len = len(sigma_list)

    #create three marginal likelyhood array according to upper three para value list    
    marginal_likelyhood_characteristic_len_scale = np.array([0.0]*char_len_scale_len, dtype='float32')
    marginal_likelyhood_sigma_l = np.array([0.0]*sigma_l_len, dtype='float32')
    marginal_likelyhood_sigma = np.array([0.0]*sigma_len, dtype='float32')
    
    #three loops to calculate data relative to upper para lists
    for i in range(char_len_scale_len):
        marginal_likelyhood_characteristic_len_scale[i] = beyesian_optimizaiton.likelyhood_calculate((characteristic_len_scale_list[i], SIGMA_L, SIGMA))
    for i in range(sigma_l_len):
        marginal_likelyhood_sigma_l[i] = beyesian_optimizaiton.likelyhood_calculate((CHARACTERISTIC_LEN, sigma_l_list[i], SIGMA))
    for i in range(sigma_len):
        marginal_likelyhood_sigma[i] = beyesian_optimizaiton.likelyhood_calculate((CHARACTERISTIC_LEN, SIGMA_L, sigma_list[i]))
    print('all the calculation has done')
    
    #plot figure: marginal likelyhood vs (characteristic len/sigma l/sigma)
    plt.plot(characteristic_len_scale_list, marginal_likelyhood_characteristic_len_scale, '-', color='blue')
    plt.title('marginal likelyhood vs charactistic len')
    plt.xlabel('characteristic len')
    plt.ylabel('log likelyhood')
    plt.xscale('log')
    plt.savefig('./output/beyesian_regression/log_marginal_likelyhood_vs_char_len.png', dpi=300)
    plt.close()

    plt.plot(sigma_l_list, marginal_likelyhood_sigma_l, '-', color='blue')
    plt.title('marginal likelyhood vs sigma l')
    plt.xlabel(r'$\sigma_l$')
    plt.ylabel('log likelyhood')
    plt.xscale('log')
    plt.savefig('./output/beyesian_regression/log_marginal_likelyhood_vs_sigma_l.png', dpi=300)
    plt.close()

    plt.plot(sigma_list, marginal_likelyhood_sigma, '-', color='blue')
    plt.title('marginal likelyhood vs sigma')
    plt.xlabel(r'$\sigma$')
    plt.ylabel('log likelyhood')
    plt.xscale('log')
    plt.savefig('./output/beyesian_regression/log_marginal_likelyhood_vs_sigma.png', dpi=300)
    plt.close()

    end_time = time.time()
    print('all figure has been plotted')
    print(f'total time cost is:{end_time - start_time:.3f}')
    
                                                                            