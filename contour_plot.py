import matplotlib.pyplot as plt
from matplotlib.colors import SymLogNorm
import numpy as np
import time
import best_paragroup_calculate as bpc #reuse same parameters

#data loading
best_theta = bpc.best_theta
[best_char_len, best_sigma_f, best_sigma] = best_theta
print([best_char_len, best_sigma_f, best_sigma])
start_time = time.time() 
file_folder = './output/martin_kernel_func/'
sigma_f_vs_sigma = np.loadtxt(file_folder+'sigmaf_vs_sigma.txt').reshape(bpc.sigma_f_vs_sigma_plane.shape)
char_len_vs_sigma = np.loadtxt(file_folder+'char_len_vs_sigma.txt').reshape(bpc.char_len_vs_sigma.shape)
char_len_vs_sigma_f = np.loadtxt(file_folder+'char_len_vs_sigma_f.txt').reshape(bpc.char_len_vs_sigma_f.shape)

# Plot figure: contour map with one parameter fixed

# plot plane value on sigma_f vs sigma 
sigma_f_sigma_mesh1, sigma_f_sigma_mesh2 = np.meshgrid(bpc.sigma_f_list, bpc.sigma_list)
char_len_sigma_mesh1, char_len_sigma_mesh2 = np.meshgrid(bpc.char_len_list, bpc.sigma_list)
char_len_sigma_f_mesh1, char_len_sigma_f_mesh2 = np.meshgrid(bpc.char_len_list, bpc.sigma_f_list)

levels = bpc.get_levels(sigma_f_vs_sigma)
contours = plt.contour(sigma_f_sigma_mesh1, sigma_f_sigma_mesh2, sigma_f_vs_sigma, levels=levels, cmap='rainbow', linewidths=1, norm=SymLogNorm(linthresh=50))
# contours = plt.contour(sigma_f_sigma_mesh1, sigma_f_sigma_mesh2, sigma_f_vs_sigma, cmap='rainbow', linewidths=1, norm=plt.matplotlib.colors.LogNorm(vmin=sigma_f_vs_sigma_plane.min(), vmax=sigma_f_vs_sigma_plane.max()))
plt.clabel(contours, inline=False, fontsize=8, colors='black')
plt.title('log likelyhood on sigma_f vs sigma plane')
plt.xlabel(r'$\sigma_f$')
plt.ylabel(r'$\sigma$')
plt.scatter([best_sigma_f], [best_sigma], marker='*', color='black')
# plt.colorbar(label='log likelyhood')
plt.xscale('log')
plt.yscale('log')
# plt.savefig('./output/beyesian_regression/log_likelyhood_on_sigma_vs_sigma_f_plane.png', dpi=300)
plt.savefig('./output/beyesian_regression/martin_log_likelyhood_on_sigma_vs_sigma_f_plane.png', dpi=300)
plt.close()

# plot plane value on char_len vs sigma
levels = bpc.get_levels(char_len_vs_sigma)
contours = plt.contour(char_len_sigma_mesh1, char_len_sigma_mesh2, char_len_vs_sigma, cmap='rainbow', linewidths=1, levels=levels, norm=SymLogNorm(linthresh=50))
plt.clabel(contours, inline=False, fontsize=8, colors='black')
plt.title('log likelyhood on charlen vs sigma plane')
plt.xlabel('characteristic len scale')
plt.ylabel(r'$\sigma$')
plt.scatter([best_char_len], [best_sigma], marker='*', color='black')
plt.xscale('log')
plt.yscale('log')
# plt.colorbar(label='log likelyhood')
# plt.savefig('./output/beyesian_regression/log_likelyhood_on_charlen_vs_sigma_plane.png', dpi=300)
plt.savefig('./output/beyesian_regression/martin_log_likelyhood_on_charlen_vs_sigma_plane.png', dpi=300)
plt.close()

# plot plane value on char_len vs sigma_f
levels = bpc.get_levels(char_len_vs_sigma_f)
contours = plt.contour(char_len_sigma_f_mesh1, char_len_sigma_f_mesh2, char_len_vs_sigma_f, cmap='rainbow', linewidths=1, levels=levels, norm=SymLogNorm(linthresh=50))
plt.clabel(contours, inline=False, fontsize=8, colors='black')
plt.title('log likelyhood on charlen vs sigma_f plane')
plt.xlabel('characteristic len scale')
plt.ylabel(r'$\sigma_f$')
plt.scatter([best_char_len], [best_sigma_f], marker='*', color='black')
plt.xscale('log')
plt.yscale('log')
# plt.colorbar(label='log likelyhood')
# plt.savefig('./output/beyesian_regression/log_likelyhood_on_charlen_vs_sigma_f_plane.png', dpi=300)
plt.savefig('./output/beyesian_regression/martin_log_likelyhood_on_charlen_vs_sigma_f_plane.png', dpi=300)
plt.close()

end_time = time.time()
print(f'all work has been done, Total time cost is: {end_time-start_time:.3f}')

