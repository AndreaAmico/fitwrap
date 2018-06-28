from collections import OrderedDict, Sized
from scipy.stats import t as scipy_stats_t

import inspect
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.signal as signal


from .fit import fit

class fit2d(fit):
    def __init__(self, f, yx_data, z_data, scatter_kwargs={}, plot_kwargs={}, imshow_kwargs={},
                 contour_kwargs={}, plotting_function=None, fitting_function=None,
                 x_rescale=1, y_rescale=1, **kwargs):
        
        self.f = f
        self.yx_data = yx_data
        self.z_data = z_data
        self.scatter_kwargs = {**{'zorder':1, 'color':'green', 'alpha':0.5}, **scatter_kwargs}
        self.plot_kwargs = {**{'zorder':3}, **plot_kwargs}
        self.imshow_kwargs = {**{'aspect':y_rescale/x_rescale}, **imshow_kwargs}
        self.contour_kwargs = {**{'colors':'w', 'alpha':0.4}, **contour_kwargs}
        self.x_rescale = x_rescale
        self.y_rescale = y_rescale


        if not plotting_function:
            plotting_function = self.plotting_function_2d

        if not fitting_function:
            fitting_function = self.curve_fit_wrap
        
        super(fit2d, self).__init__(f, yx_data, np.ravel(z_data), fitting_function=fitting_function,
                                    plotting_function=plotting_function, **kwargs)
    
    def curve_fit_wrap(self, f, *args, **kwargs):
        def fit_function_wrap(*args, **kwargs):
            return np.ravel(f(*args, **kwargs))
        return scipy.optimize.curve_fit(fit_function_wrap, **kwargs)

    def plotting_function_2d(self, f, val, yx, z):

        z = z.reshape(yx[0].shape)

        left, width = 0.1, 0.65
        bottom, height = 0.1, 0.65
        bottom_h = left_h = left + width + 0.02
        
        image_ratio = z.shape[0]/z.shape[1]

        rect_imshow = [left, bottom, width, height]
        rect_projx = [left, bottom_h, width, 0.2]
        rect_projy = [left_h, bottom, 0.2, height]
        rect_colorbar = [left_h, bottom_h, 0.1, 0.2]

        if image_ratio>1:
            self.fig = plt.figure(1, figsize=(8*self.x_rescale/image_ratio, 8*self.y_rescale))
        else:
            self.fig = plt.figure(1, figsize=(12*self.x_rescale, 12*image_ratio*self.y_rescale))

        self.ax_imshow = plt.axes(rect_imshow)
        self.ax_projx = plt.axes(rect_projx)
        self.ax_projy = plt.axes(rect_projy)
        self.ax_colorbar = plt.axes(rect_colorbar)
        
        self.ax_projx.xaxis.set_major_formatter(plt.NullFormatter())
        self.ax_projy.yaxis.set_major_formatter(plt.NullFormatter())

        z_data_imshow = self.ax_imshow.imshow(z, **self.imshow_kwargs)
        self.fit_result_data = f(yx, *val).reshape(z.shape)
        self.ax_imshow.contour(yx[1], yx[0], self.fit_result_data, 10, **self.contour_kwargs)

        self.ax_projy.scatter(np.sum(z, 1), np.arange(z.shape[0]), **self.scatter_kwargs)
        self.ax_projy.plot(np.sum(self.fit_result_data, 1), np.arange(self.fit_result_data.shape[0]), **self.plot_kwargs)
        
        self.ax_projx.scatter(np.arange(z.shape[1]), np.sum(z, 0),**self.scatter_kwargs)
        self.ax_projx.plot(np.arange(self.fit_result_data.shape[1]), np.sum(self.fit_result_data, 0), **self.plot_kwargs)
        
        plt.colorbar(z_data_imshow, cax=self.ax_colorbar)
        
        self.ax_projx.set_xlim(-0.5, z.shape[1]-0.5)
        self.ax_projx.grid(alpha=0.3)
        self.ax_projy.set_ylim(z.shape[0]-0.5, -0.5)
        self.ax_projy.grid(alpha=0.3)
        self.ax_imshow.grid(alpha=0.3)
