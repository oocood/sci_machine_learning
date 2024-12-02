#这个程序在于对在程序simple_diffusion_equation.py中的cfd的数据进行koopman分解
#也就是通过连接前后时步的线性变换（存疑）矩阵来进行动力学分解

import numpy as np
import math
import scipy
import matplotlib.pyplot as plt
import os
import parameters as p
from scipy import sparse
import time 
import gc
import multiprocessing as mp

#图像存储路径

fig_path = './output/fig_koopman_decomposition/' 

#图像并行绘制函数
def parallel_plot(koopman_mode_data):
    # print(koopman_mode_data[1])
    (eignvector, eignvalue, i) = koopman_mode_data
    # eignvector = koopman_mode_data[0]
    # eignvalue = koopman_mode_data[1]
    # i = koopman_mode_data[2]
    plt.imshow(np.abs(eignvector).reshape(p.nelements_x, -1), cmap='coolwarm', extent=[0, 1, 0, 1])
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    if eignvalue.imag<0:
        plt.title(f'eignvalue = {eignvalue.real:.5f}{eignvalue.imag:.5f}i')
    else:
        plt.title(f'eignvalue = {eignvalue.real:.5f}+{eignvalue.imag:.5f}i')
    plt.colorbar(label='value')
    plt.savefig(fig_path + str(i) + '.png', dpi=300)
    plt.close()

if __name__ == '__main__':
    #文件路径选取和文件批量读取
    data_path = './output/data/'
    files = sorted(os.listdir(data_path))
    print(files)
    for i in range(len(files)):
        files[i] = os.path.join(data_path, files[i])

    #加载数据
    start_time = time.time()
    backward_data = np.zeros((p.nelements_x*p.nelements_y, len(files)-1), dtype='float32') #为-(last_step-1)的数据
    forward_data = np.copy(backward_data) #1-last_step的数据
    for i in range(len(files) - 1): #数据present_data_t为当前时步的数据，present_data_t_plus_1是下一时步的数据
        present_data_t = np.loadtxt(files[i]).reshape(p.nelements_x*p.nelements_y, -1)
        present_data_t = np.squeeze(present_data_t)
        present_data_t_plus_1 = np.loadtxt(files[i+1]).reshape(p.nelements_x*p.nelements_y, -1)
        present_data_t_plus_1 = np.squeeze(present_data_t_plus_1)
        backward_data[:, i] = present_data_t
        forward_data[:, i] = present_data_t_plus_1

    #对backward_data组成的数据矩阵进行svd分解,并且仅仅保存前10个特征值和其特征向量
    # backward_data = sparse.csc_matrix(backward_data) #流场是一个稀疏矩阵，所以采用列压缩的方式
    n = 25 #设置保留最大特征值的个数
    u, s, vt = sparse.linalg.svds(backward_data, n, which='LM') #svd分解
    # u = sparse.csc_matrix(u)
    # vt = sparse.csr_matrix(vt)
    print(s)

    #构建koopman矩阵
    print(s.shape, u.shape, vt.shape)
    for i in range(s.shape[0]): #奇异值对角矩阵的逆，假设其可逆
        s[i] = 1.0/s[i]
    print(s)
    s_diag = np.diag(s) #由于svd分解求得的s为数组，将其转化为对角矩阵
    # s_diag = sparse.csc_matrix(s_diag)
    print(forward_data.shape)
    # koopman_mat = forward_data @ vt.conj().T @ s_diag @ u.conj().T #似乎没用到

    #koopman分解
    koopman_mat_hat = u.conj().T @ forward_data @ vt.conj().T @ s_diag #减小计算量，使用该矩阵进行特征值计算并且根据关系反推出koopman_mat的特征值和特征向量
    k = 30 #计算前k大大特征值及其右特征向量
    reignvalues, reignvectors = scipy.linalg.eig(koopman_mat_hat)
    print(reignvalues.shape, reignvectors.shape)
    koopman_reignvalues = reignvalues
    koopman_reignvectors = np.dot(forward_data, vt.conj().T) @ s_diag @ reignvectors

    # sorted_indices = np.argsort(koopman_reignvalues.real)
    # koopman_reignvalues = koopman_reignvalues[sorted_indices]
    # koopman_reignvectors = koopman_reignvectors[:, sorted_indices] #通过特征值的实部排序，对应特征向量也进行排序
    print(koopman_reignvalues)
    print('koopman decomposition has been done!')

    #koopman特征向量绘制
    # fig_path = './output/fig_koopman_decomposition/'
    # for i in range(len(koopman_reignvalues)):
    #     plt.imshow(np.abs(koopman_reignvectors[:, i]).reshape(p.nelements_x, -1), cmap='coolwarm', extent=[0, 1, 0, 1])
    #     plt.xlabel('x axis')
    #     plt.ylabel('y axis')
    #     if koopman_reignvalues[i].imag<0:
    #         plt.title(f'eignvalue = {koopman_reignvalues[i].real:.5f}{koopman_reignvalues[i].imag:.5f}i')
    #     else:
    #         plt.title(f'eignvalue = {koopman_reignvalues[i].real:.5f}+{koopman_reignvalues[i].imag:.5f}i')
    #     plt.colorbar(label='value')
    #     plt.savefig(fig_path + str(i) + '.png', dpi=300)
    #     plt.close()

    #启用并行绘图，缩短运行时间
    koopman_modes_data =[(koopman_reignvectors[:, i], koopman_reignvalues[i], i) for i in range(koopman_reignvectors.shape[1])]
    print(koopman_modes_data[0])
    num_cores = mp.cpu_count()
    print(f'available cores in this computer is: {num_cores}')
    with mp.Pool(processes = num_cores) as pool:
        pool.map(parallel_plot, koopman_modes_data)
    print('koopman eignvector map has been plotted!')

    #koopman特征值的谱图
    koopman_reignvalues_real = koopman_reignvalues.real
    koopman_reignvalues_imag = koopman_reignvalues.imag
    plt.plot(koopman_reignvalues_real, koopman_reignvalues_imag, 'o', markerfacecolor='none', color = 'blue', alpha=0.5)
    plt.plot(np.sin(math.pi*np.arange(-1, 1, 0.001)), np.cos(math.pi*np.arange(-1, 1, 0.001)), '--', color = 'black')
    plt.xlabel('eignvalue real part')
    plt.ylabel('eignvalue imag part')
    plt.axis('equal')
    plt.savefig('./output/fig_koopman_decomposition/eignvalue_cycle.png', dpi=300)
    plt.close()
    print('koopman eignvalues map has been plotted!')

    #分别存储N维函数空间的基（koopman矩阵的左特征值）、koopman模态（koopman矩阵的右特征值）、谱（koopman矩阵的特征值）
    # leignvalues, leignvectors = sparse.linalg.eigs(koopman_mat_hat.conj().T, k, which='LM') #求解左特征值
    # koopman_leignvalues = leignvalues
    # koopman_leignvectors = u @ leignvectors
    # sorted_indices = np.argsort(koopman_leignvalues.real)
    # koopman_leignvalues = koopman_leignvalues[sorted_indices]
    # koopman_leignvectors = koopman_leignvectors[:, sorted_indices] #按照特征值的实部排序

    # save_path1 = './output/koopman_left_eignvector.txt' #存储左特征向量
    # save_path2 = './output/koopman_right_eignvector.txt' #存储右特征向量
    # save_path3 = './output/koopman_eignvalue.txt' #存储特征值
    # with open(save_path1, 'w') as f1, open(save_path2, 'w') as f2, open(save_path3, 'w') as f3:
    #     np.savetxt(f1, np.abs(koopman_leignvectors), fmt = '%f')
    #     np.savetxt(f2, np.abs(koopman_reignvectors), fmt = '%f')
    #     np.savetxt(f3, np.abs(koopman_reignvalues), fmt = '%f')
    # f1.close()
    # f2.close()
    # f3.close()

    #重构流场
    # gc.collect() #清理内存
    # print(koopman_reignvalues-koopman_leignvalues)
    # sorted_indices = np.argsort(np.abs(koopman_reignvalues - koopman_leignvalues))
    # koopman_reignvalues = koopman_reignvalues[sorted_indices]
    # koopman_reignvectors = koopman_reignvectors[:, sorted_indices]
    # koopman_leignvectors = koopman_leignvectors[:, sorted_indices]
    # truncated_num = int(k/2)

    # koopman_eignvalues = koopman_reignvalues[0:truncated_num] #为了简化计算，截断实特征值最大的部分
    # koopman_reignvectors = koopman_reignvectors[:, 0:truncated_num]
    # koopman_leignvectors = koopman_leignvectors[:, 0:truncated_num]

    refactor_path = './output/koopman_refactor_cfd/' #重构数据的存储路径
    # print(backward_data[:, 0])
    with open(refactor_path+'0.txt', 'w') as f:
        np.savetxt(f, np.loadtxt(files[0]), fmt='%f')
    f.close()
    t_list_k = [20, 50, 100, 150]
    print('校验是否垂直')
    print(koopman_reignvectors.conj().T @ koopman_reignvectors)
    phi = np.squeeze(np.linalg.inv(koopman_reignvectors.conj().T @ koopman_reignvectors) @ koopman_reignvectors.conj().T @ backward_data[:, 0]) #求出phi(u0)
    print(phi.shape)
    eignvalues = koopman_reignvalues.real
    print(eignvalues.shape)
    for i in range(len(t_list_k)): #根据koopman分解的结果及其性质重构流场
        # eignvalue_diag_mat = np.diag(koopman_eignvalues**t_list_k[i]) #构建特征值对角矩阵
        # c_refactor = koopman_reignvectors @ eignvalue_diag_mat @ koopman_leignvectors.conj().T @ backward_data[:, 0]
        print(eignvalues.shape)
        mid_tmp = eignvalues**t_list_k[i]*phi
        print(mid_tmp.shape)
        c_refactor = koopman_reignvectors @ mid_tmp
        print(c_refactor.shape)
        with open(refactor_path+str(f'{t_list_k[i]:03d}')+'.txt', 'w') as f:
            np.savetxt(f, np.abs(c_refactor), fmt='%f')
        f.close()

    #结束
    end_time = time.time()
    print('all the calculation has been done!')
    print('time cost is : ', end_time - start_time, ' s')




