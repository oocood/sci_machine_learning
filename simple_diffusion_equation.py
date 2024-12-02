#这个程序用于求解区域为[0,1]*[0,1]区域的二维扩散方程。
#求解方法：有限体积方法,边界面上取单个点的guass积分，左端时间偏导数项的体积分平均用单元中心的取值来近似
#对流项采用一阶迎风格式，扩散项的梯度采用中心差分，时间离散采用向前差分

import numpy as np
import matplotlib.pyplot as plt
import math
import time
from scipy import sparse
import parameters as p
#区域离散化
h = p.h
dx = dy = h
x_len = y_len = p.x_len
x = np.arange(0, x_len + dx, dx)
y = np.arange(0, y_len + dy, dy)
nelements_x = len(x) - 1
nelements_y = len(y) - 1

#设置步长和总时间
delta_t = p.delta_t
total_t = p.total_t

#扩散系数
diff_coef = p.diff_coef
delta_t_mult_diff_coef_subt_h_2 = p.delta_t_mult_diff_coef_subt_h_2
delta_t_subt_h_2 = p.delta_t_subt_h_2

#设置数据存储路径
save_path = './output/data/diffusion_fvm' #非完整路径

#有限体积中心点取值数组
c = np.array([0.0]*(nelements_x*nelements_y), dtype='float32') #c为浓度数组，初始化为0

#初始化
left_boundary_idx = np.array([0]*(nelements_y), dtype = 'int32')
right_boundary_idx = np.array([0]*(nelements_x), dtype = 'int32')
for i in range(nelements_y): #给左边界赋值：1/3<y<2/3取1， 其他部分取0
    left_boundary_idx[i] = i*nelements_x
    right_boundary_idx[i] = left_boundary_idx[i] + (nelements_x - 1)
    if i*dy>(1/3) and i*dy<(2/3):   c[left_boundary_idx[i]] = 1

#构建代数方程组及其矩阵
mat = np.zeros((nelements_x*nelements_y, nelements_x*nelements_y), dtype='float32')
four_corner_id = [0, nelements_x-1, nelements_x*(nelements_y - 1), nelements_x*nelements_y - 1] #四个特殊的边界点的id，在迭代矩阵的构建的时候需要特殊处理

#填充中间部分，idx从1到n-2， idy也是从1到n-2
# central_mat_id = np.array([0]*((nelements_x-1)*(nelements_y-1)), dtype='int32')
for i in range(1, nelements_y-1, 1):
    for j in range(1, nelements_x-1, 1):
        central_id = i*nelements_x + j
        mat[central_id][central_id] = (1 - delta_t_subt_h_2*(4*diff_coef + h))
        mat[central_id][central_id - nelements_x] = delta_t_mult_diff_coef_subt_h_2
        mat[central_id][central_id - 1] = delta_t_subt_h_2*(diff_coef + h)
        mat[central_id][central_id + 1] = delta_t_mult_diff_coef_subt_h_2
        mat[central_id][central_id + nelements_x] = delta_t_mult_diff_coef_subt_h_2

#左边界对应的行赋值,左边界采用dirichilet边界，取值始终都不变，等于初始值
for i in range(nelements_y):
    mat[left_boundary_idx[i]][left_boundary_idx[i]] = 1

#右边界对应的行赋值，右边界采用neumann边界，浓度梯度为零，也就是无朝右边的通量
for i in range(nelements_y):
    if right_boundary_idx[i]==four_corner_id[1]: #达到了右上边界
        mat[right_boundary_idx[i]][right_boundary_idx[i]] = (1 - delta_t_subt_h_2*(3*diff_coef + h))
        # mat[right_boundary_idx[i]][right_boundary_idx[i] - nelements_x] = delta_t_subt_h_2*(diff_coef + h)
        mat[right_boundary_idx[i]][four_corner_id[3]] = delta_t_mult_diff_coef_subt_h_2 #周期性边界条件
        mat[right_boundary_idx[i]][right_boundary_idx[i] - 1] = delta_t_subt_h_2*(diff_coef + h)
        # mat[right_boundary_idx[i]][right_boundary_idx[i] + 1] = delta_t_mult_diff_coef_subt_h_2 #零通量边界条件
        mat[right_boundary_idx[i]][right_boundary_idx[i] + nelements_x] = delta_t_mult_diff_coef_subt_h_2
    elif right_boundary_idx[i]==four_corner_id[3]: #达到了右下边界
        mat[right_boundary_idx[i]][right_boundary_idx[i]] = (1 - delta_t_subt_h_2*(3*diff_coef + h))
        mat[right_boundary_idx[i]][right_boundary_idx[i] - nelements_x] = delta_t_mult_diff_coef_subt_h_2
        mat[right_boundary_idx[i]][right_boundary_idx[i] - 1] = delta_t_subt_h_2*(diff_coef + h)
        # mat[right_boundary_idx[i]][right_boundary_idx[i] + 1] = delta_t_mult_diff_coef_subt_h_2 #零通量边界条件
        # mat[right_boundary_idx[i]][right_boundary_idx[i] + nelements_x] = delta_t_mult_diff_coef_subt_h_2 #周期性边界条件
        mat[right_boundary_idx[i]][four_corner_id[1]] = delta_t_mult_diff_coef_subt_h_2 #周期性边界条件

    else:
        mat[right_boundary_idx[i]][right_boundary_idx[i]] = (1 - delta_t_subt_h_2*(3*diff_coef + h))
        mat[right_boundary_idx[i]][right_boundary_idx[i] - nelements_x] = delta_t_mult_diff_coef_subt_h_2
        mat[right_boundary_idx[i]][right_boundary_idx[i] - 1] = delta_t_subt_h_2*(diff_coef + h)
        # mat[right_boundary_idx[i]][right_boundary_idx[i] + 1] = delta_t_mult_diff_coef_subt_h_2 #零通量边界条件
        mat[right_boundary_idx[i]][right_boundary_idx[i] + nelements_x] = delta_t_mult_diff_coef_subt_h_2

#上边界和下边界的周期性边界条件
# up_boundary_id = np.array([0]*(nelements_x-2), dtype = 'int32')
# bottom_boundary_id = np.copy(up_boundary_id)
for i in range(1, nelements_x-1, 1):
    up_boundary_periodic_id = (nelements_y - 1)*nelements_x + i
    # bottom_boundary_id = i 
    #上边界赋值
    mat[i][i] = (1 - delta_t_subt_h_2*(4*diff_coef + h))
    mat[i][up_boundary_periodic_id] = delta_t_mult_diff_coef_subt_h_2 #周期性边界条件
    mat[i][i-1] = delta_t_subt_h_2*(diff_coef + h)
    mat[i][i+1] = delta_t_mult_diff_coef_subt_h_2
    mat[i][i + nelements_x] = delta_t_mult_diff_coef_subt_h_2
    #下边界赋值
    mat[up_boundary_periodic_id][up_boundary_periodic_id] = (1 - delta_t_subt_h_2*(4*diff_coef + h))
    mat[up_boundary_periodic_id][up_boundary_periodic_id - nelements_x] = delta_t_mult_diff_coef_subt_h_2
    mat[up_boundary_periodic_id][up_boundary_periodic_id - 1] = delta_t_subt_h_2*(diff_coef + h)
    mat[up_boundary_periodic_id][up_boundary_periodic_id + 1] = delta_t_mult_diff_coef_subt_h_2
    mat[up_boundary_periodic_id][i] = delta_t_mult_diff_coef_subt_h_2

#打印mat和c的维度
print(mat.shape, c.shape)

#开始迭代
if __name__ == "__main__":
    start_time = time.time() #计时开始
    mat = sparse.csr_matrix(mat) #按照行压缩为稠密矩阵
    solve_t_list = np.arange(0, total_t, delta_t)
    with open(save_path + '0.txt', 'w') as f: #保存初始值
        np.savetxt(f, c, fmt='%f')
        f.close()
    print('intial value has been stored')
    for i in range(1, len(solve_t_list), 1):
        c = mat.dot(c)
        if(i%1==0):
            print(str(i)+' th calculation has been done!')
            flag = int(i/1)
            save_path_tmp = save_path + str(f'{flag:03d}') + '.txt'
            with open(save_path_tmp, 'w') as f:
                np.savetxt(f, c, fmt='%f')
                f.close()

    #计时结束
    end_time = time.time()
    print('all the calculation has been done!')
    print('time cost is',end_time - start_time, '\t s')
        


        