from collections import OrderedDict, Sized
from scipy.stats import t as scipy_stats_t

import inspect
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.signal as signal


from .fit import fit

def lomb_spectrum(x, y, frequency_span=None, grid_size=1000):
    x = np.array(x)
    y = np.array(y)
    if frequency_span:
        min_frequency, max_frequency = [f*2*np.pi for f in frequency_span]
    else:
        total_time = np.max(x) - np.min(x)
        min_frequency = 1/(total_time*2)
        max_frequency = min_frequency*x.shape[0]
    frequency_grid = np.linspace(min_frequency, max_frequency, grid_size)
    pgram = signal.lombscargle(x, y, frequency_grid)
    lombscargle_spectrum = np.sqrt(4*(pgram/x.shape[0]))
    return frequency_grid/(2*np.pi), lombscargle_spectrum

class fit_sin(fit):
    def __init__(self, x, y, sigma=None, off=None, freq=None, amp=None, phase=None,
             plot_results=True, print_results=True, lomb_frequency_span=None,
             lomb_grid_size=10000, **kwargs):
        '''
        args:
            - x, y : data pointS to fit with the function y(x) = off + amp * sin(2 * pi * freq * x + phase)
            - sigma=None : error of y data (default is no error)
            - off : initial guess on the offset. (if None the initial guess is the mean of the y points)
            - freq : initial guess on the frequency. (if None the initial guess is estimated finding
                        the maximum of the lombscargle spectrum)
            - amp : initial guess on the amplitude. (if None the initial guess is set to 2.8*standard_dev(y))
            - phase : initial guess on the phase. (if None the initial guess is np.pi/8)
            - plot_result : plot the data points and the fit result
            - print_results : print the results table
            - lomb_frequency_span: set the frequency span of the lombscargle spectrum to estimate the frequency.
                        (if None, the min frequency is given by 1/((np.max(x) - np.min(x))*2) and the max frequency
                        by min_frequency*size_of_the_x_array)
            - lomb_grid_size: size of the lombscargle spectrum grid. (default is 10000. Higher is slower but more precise)

        '''

        if not off:
            off = np.mean(y)

        if not freq:
            frequency_grid, lombscargle_spectrum = lomb_spectrum(x, y,
                            frequency_span=lomb_frequency_span, grid_size=lomb_grid_size)
            maximum_indices = np.where(lombscargle_spectrum==np.max(lombscargle_spectrum))    
            freq = frequency_grid[maximum_indices[0]][0]

        if not amp:
            amp = np.std(y)*2.8

        if not phase:
            phase = np.pi/8

        def sine(x, off=off, amp=amp, freq=freq, phase=phase):
            return  off + amp * np.sin(2*np.pi*freq*x + phase)

        if print_results:
            print('Fitting function model: y = off + amp * sin(2 * pi * freq * x + phase)')
            
        super(fit_sin, self).__init__(sine, x, y, sigma=sigma, plot_results=plot_results, print_results=print_results, **kwargs)

