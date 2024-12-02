#这个程序是更根据本征正交分解结合降阶模型来进行计算的
#构建降阶模型的矩阵来源于快照矩阵的svd分解的左奇异向量，其中快照矩阵S来源于
#simple_diffusion_method程序输出的结果，先进行均匀取样，使用简单的梯形公式进行分片积分
#作为数值积分的方式，得到的积分具有一阶代数精度

import numpy as np
# import matplotlib.pyplot as plt
import os
# import maths
import time
import parameters as p
import simple_diffusion_equation as sde
import scipy
from scipy import sparse

#数据输入的路径
data_path = './output/data'
files = os.listdir(data_path)
files = sorted(files)
for i in range(len(files)):
    files[i] = os.path.join(data_path, files[i])

#数据存储的路径
save_path = './output/data_pod_k5/pod_method'

#均匀采样，每两个数据选取一个数据作为输入,构建快照矩阵
input_data = np.zeros((p.nelements_x*p.nelements_y, int(len(files)/1)), dtype='float32') #存储快照矩阵
for i in range(int(len(files)/1)):
    input_data[:, i] = np.squeeze(np.loadtxt(files[i*1]).reshape(p.nelements_x*p.nelements_y, -1))
input_data = input_data*np.sqrt(p.delta_t) #乘以积分的步长开根号

#对于快照矩阵进行svd分解得到左正交矩阵
snap_shot_mat = sparse.csc_matrix(input_data)
n = 5 #设置保留前n个最大的奇异值
u, s, vt = scipy.sparse.linalg.svds(snap_shot_mat, n)
print(s)
print(u.shape, s.shape, vt.shape)

#基矩阵V构建,V根据选取的k个数进行v = u[:,0:k]
v = sparse.csc_matrix(u)

#矩阵计算：v*dq/dt = A*v*q, dq/dt = vt*A*v*q
iter_mat = v.T @ sparse.csr_matrix(sde.mat) @ v #提前计算迭代矩阵 vt*A*v

#时间离散选取一阶显式欧拉
delta_t = 1e-4 #设置迭代步长
total_t = p.total_t
u0 = np.loadtxt(files[0]).reshape(p.nelements_x*p.nelements_y, -1)
q0 = v.T @ u0 #初始值计算

#开始迭代
# if "__name__" == "__main__":
start_time = time.time() #开始
q = q0
with open(save_path+'0.txt', 'w') as f: #存储初始值
    u0 = v @ q
    np.savetxt(f, u0, fmt='%f')
    f.close()
for i in range(int(total_t/delta_t)):
    q = iter_mat @ q
    if(i%100==0):
        flag = int(i/100)
        save_path_tmp = save_path + str(f'{flag:03d}') + '.txt'
        with open(save_path_tmp, 'w') as f:
            u = v @ q
            np.savetxt(f, u, fmt='%f')
            f.close()
end_time = time.time() #结束
print('all the calculation has been done!(pod method k=4)')
print('time cost = ', end_time - start_time, ' s')



    
