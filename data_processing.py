# 这个程序是用来进行 进行对simple_diffusion_equation.py得到的数据进行后处理

import numpy as np
import matplotlib.pyplot as plt
import os
# import simple_diffusion_equation as se
import parameters as p
from mpl_toolkits.mplot3d import Axes3D
import time
import multiprocessing as mp
#全局参数
# fig_save_path = './output/fig_koopman_refactor_cfd/'
fig_save_path = './output/fig/'
# t_list_k = [0, 20, 50, 100, 150]

#创建并行工作函数
def parallel_func(tuple1):
    path, i = tuple1
    data = np.loadtxt(path).reshape(p.nelements_y, p.nelements_x)
    plt.imshow(data, cmap='coolwarm', extent=[0, 1, 0, 1])
    plt.xlabel('x axis')
    plt.ylabel('y axis')
    # plt.title(f't={p.delta_t*t_list_k[i]*100:.2f}')
    plt.title(f't={p.delta_t*i*1:.2f}')
    plt.colorbar(label='value')
    plt.savefig(fig_save_path+str(i)+'.png', dpi=300)
    plt.close()

if __name__=='__main__':
    # data_save_path = './output/koopman_refactor_cfd'
    data_save_path = './output/data/'
    files = os.listdir(data_save_path)
    files = sorted(files) #按照文件名进行文件数组的排序
    print(files)
    for i in range(len(files)): #路径拼接
        files[i] = os.path.join(data_save_path, files[i])
    x_mesh, y_mesh = np.meshgrid(p.x[0:-1]+p.h/2, p.y[0:-1]+p.h/2)
    start_time = time.time()
    # for i in range(len(files)):
    #     data = np.loadtxt(files[i]).reshape(p.nelements_y, p.nelements_x)
    #     # fig = plt.figure()
    #     # ax = fig.add_subplot(1, 1, 1, projection = '3d')
    #     # ax.plot_surface(x_mesh, y_mesh, data, cmap='viridis')
    #     plt.imshow(data, cmap='coolwarm', extent=[0, 1, 0, 1])
    #     plt.xlabel('x axis')
    #     plt.ylabel('y axis')
    #     plt.title(f't={p.delta_t*i*100:.2f}')
    #     plt.colorbar(label='value')
    #     # plt.show()
    #     plt.savefig(fig_save_path+str(i)+'.png', dpi=300)
    #     plt.close()
    num_cores = mp.cpu_count()
    print(f'available cores in this computer is: {num_cores}')
    with mp.Pool(processes = num_cores) as pool:
        pool.map(parallel_func, zip(files, range(len(files))))
    end_time = time.time()
    print('all fig has been created!')
    print('time cost is', end_time - start_time, ' s')



