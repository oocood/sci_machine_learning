#参数设置
import numpy as np

#区域离散化
h = 5e-3
dx = dy = h
x_len = y_len = 1
x = np.arange(0, x_len + dx, dx)
y = np.arange(0, y_len + dy, dy)
nelements_x = len(x) - 1
nelements_y = len(y) - 1

#设置步长和总时间
delta_t = 1e-2
total_t = 2

#扩散系数
diff_coef = 1e-2
delta_t_mult_diff_coef_subt_h_2 = delta_t*diff_coef/h**2
delta_t_subt_h_2 = delta_t/h**2