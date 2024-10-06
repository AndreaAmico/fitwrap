from collections import OrderedDict

try:
    from collections.abc import Sized
except:
    from collections import Sized

from scipy.stats import t as scipy_stats_t

import inspect
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.signal as signal


from .fit import fit



class fit_gauss(fit):
    def __init__(self, x, y, sigma=None, off=None, x0=None, amp=None, sx=None,
             plot_results=True, print_results=True, **kwargs):


        data_array = np.array([x, y]).T
        x_sorted_data = data_array[np.argsort(data_array[:,0])]
        y_sorted_data = data_array[np.argsort(data_array[:,1])]

        min_y = y_sorted_data[0, :]
        max_y = y_sorted_data[-1, :]

        min_x = x_sorted_data[0, :]
        max_x = x_sorted_data[-1, :]


        if np.abs((min_x[1] + max_x[1])/2 - min_y[1]) < np.abs((min_x[1] + max_x[1])/2 - max_y[1]):
            off_auto, x0_auto, amp_sign = min_y[1], max_y[0], 1
        else:
            off_auto, x0_auto, amp_sign = max_y[1], min_y[0], -1


        if not off:
            off = off_auto
        if not x0:
            x0 = x0_auto
        if not amp:
            amp = amp_sign*np.abs(max_y[1] - min_y[1])

        if not sx:
            right_side = x_sorted_data[x_sorted_data[:, 0]>x0, :]
            left_side = x_sorted_data[x_sorted_data[:, 0]<x0, :]

            right_side[:, 1] = np.abs(right_side[:, 1] - off - amp/2)
            right_fwhm = right_side[np.where(right_side[:, 1]==np.min(right_side[:, 1]))[0],0]

            left_side[:, 1] = np.abs(left_side[:, 1] - off - amp/2)
            left_fwhm = left_side[np.where(left_side[:, 1]==np.min(left_side[:, 1]))[0],0]
            sx = ((right_fwhm - left_fwhm) / (2*np.sqrt(2 * np.log(2))))[0]

        def gauss(x, off=off, amp=amp, x0=x0, sx=sx):
            return  off + amp * np.exp(-(x-x0)**2/(2*sx**2))

        if print_results:
            print('Fitting function model: y = off + amp * exp(-(x-x0)^2 / (2 * sx^2))')
            
        super(fit_gauss, self).__init__(gauss, x, y, sigma=sigma, plot_results=plot_results, print_results=print_results, **kwargs)

