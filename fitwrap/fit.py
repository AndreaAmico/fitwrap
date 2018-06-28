from collections import OrderedDict, Sized
from scipy.stats import t as scipy_stats_t

import inspect
import matplotlib.pyplot as plt
import numpy as np
import scipy.optimize
import scipy.signal as signal


class fit(object):
    ''' Fitting wrapper to use the default parameters of the function to initialize the fit parameters.

    With defaults it uses the scipy.optimize.curve_fit function as fitting method. The fit variables are
    all the function parameters except for the first argument, that is the x variable and the last one
    if its name is fixed_args. In the latter case, all the variable contained in the fixed_args tuple
    are considered fixed parameters, and they are not considered by the fitting method.
    '''
    def __init__(self, function, xdata, ydata,
        print_results = True,
        plot_results = True,
        fitting_function = scipy.optimize.curve_fit,
        plotting_function = False,
        plot_range_x = False,
        plot_range_y = False,
        fig_ax = False,
        debug = False,
        **kwargs):

        ''' Fitting method wrapper (scipy.optimize.curve_fit if the default)

        Args:
        function (func): Fit function model. Initial guesses can be set as default value of the variable. Boundaries
        can be set as touple of 3 elements: (initial_guess, min value, max value). The special variable name
        variable_values can be used to exclude one or more variables from the fit.
        Example function:
            def model_function(x, off, amp=2, freq=(10, 8, 18), phi=0, fixed_values=['phi']):
                return off + amp * np.cos(x*freq*2*np.pi + phi)
        xdata (iterable): x data set.
        ydata (iterable): y data set.
        print_results (bool): print the fit results.
        plot_results (bool): plot the fit result using the method plotting_function
        fitting_function (func): fit function, default is scipy.optimize.curve_fit. {f=self.clean_function,
            xdata=self.xdata, ydata=self.ydata, p0=self.variable_values, **kwargs} will be passed as arguments
            to the fitting_function.
        plotting_function: plotting function method that take {self.clean_function, self.val,
            self.xdata, self.ydata, **self.kwargs} as arguments. If False, a default plotting function will be used.
        plot_range_x (2 sized iterable): x limits for the plot
        plot_range_y (2 sized iterable): y limits for the plot
        fig_ax (matplotlib.pyplot figure, matplotlib.pyplot axes): axex and figures where the plot will be placed,
            if False, a new figure and axes will be created.
        debug (bool): If True the plot function will produce 4 axes showing seperately the data points, the fit model
            with the initial guess, the fit model with the initial guess plus the data points and the fit result plus
            the data points.

        Full usage example:
            def f(x, a=-1, b=6 , c=0.5, d=3):
                return a + b*x + c*x**2 + d*x**3

            xx = np.linspace(-1, 1, 50)
            yy = f(xx)+(np.random.random(xx.shape[0])-0.5)*2
            y_errs = (np.random.random(xx.shape[0])-0.5)*2

            def g(x, a, b=6 , c=0.5, d=3, fixed_args=['b']):
                return a + b*x + c*x**2 + d*x**3

            def my_plot(f, p0, x, y):
                xx = np.linspace(0,3, 30)
                yy = f(xx, *p0)
                plt.scatter(x, y)
                plt.plot(xx, yy, color='red')

            fit(g, xx, yy, debug=True, sigma=y_errs) # plot debug mode
            plt.show()

            fit(g, xx, yy, plotting_function=my_plot) # plot with custom function
            plt.show()

            fit(g, xx, yy) # default plot
            plt.show()
        '''

        self.function = function
        self.xdata = np.array(xdata)
        self.ydata = np.array(ydata)
        self.plot_range_x = plot_range_x
        self.plot_range_y = plot_range_y
        self.fig_ax = fig_ax
        self.kwargs = kwargs
        self.params = OrderedDict()

        if plotting_function:
            self.plotting_function = plotting_function
        elif debug:
            self.plotting_function = self._plotting_debug
        else:
            self.plotting_function = self._plotting_function


        self.fit_plot_params = {
            'FILL_BETWEEN_COLOR' : [x/255 for x in (223, 180, 175)],
            'BACKGROUND_COLOR' : [x/255 for x in (254, 251, 243, 255)],
            'MARKER_COLOR' : (0.35,0.77,0.55),
            'BASE_COLOR' : 'brown',
            'LINEWIDTH' : 1.5,
            'PLOT_POINTS' : 500,
            'FIGSIZE' : (8,5),
            'PADDING' : 0.1,
            'MARKER' : 'o',
            'MARKEREDGEWIDTH_BIG' : 0.8,
            'MARKEREDGEWIDTH_SMALL' : 0.6,
            'MARKERSIZE_BIG' : 7,
            'MARKERSIZE_SMALL' : 4,
            'NUM_POINT_LIMIT' : 100,
            'GRID' : True,
            'GRID_LINEWIDTH' : 0.5,
            'GRID_STYLE' : ':'}

        self._calculate_function_parameters()   #parse the function parameters to extract self.parameters_index, 
                                                #self.function_parameters, self.fixed_values, self.fixed_index,
                                                #self.variable_values, self.variable_index, self.fixed_args_name
                                                #self.bounds, self.is_bounded

        self.clean_function = self._wrap_function() #wrap function to remove the fixed parameters from the argument


        def _try_fit():
            if self.is_bounded:
                self.out = fitting_function(f=self.clean_function, xdata=self.xdata, ydata=self.ydata,
                    p0=self.variable_values, bounds=list(zip(*self.bounds)), **kwargs)
            else:
                self.out = fitting_function(f=self.clean_function, xdata=self.xdata, ydata=self.ydata,
                    p0=self.variable_values, **kwargs)


            self.val, self.cov = self.out
            self.err = np.sqrt((np.diag(self.cov)))

        self.fit_ok = False
        if debug:
            try:
                _try_fit()
                self.fit_ok = True
            except:
                print('Fit failed')
                self.val = self.variable_values
                self.err = np.zeros(len(self.variable_values))
        else:
            _try_fit()
            self.fit_ok = True
            
        if self.fit_ok:
            self._save_fitted_parameters()

        if plot_results:
            self._plot_results()
        if print_results:
            self._print_results()
        
    def _get_first_item(self, variable):
        if isinstance(variable, Sized):
            return variable[0]
        else:
            return variable

    def _wrap_function(self):
        new_args = [0]*len(self.parameters_index)
        def new_function(x, *args):
            for value, index in zip(args+tuple(self.fixed_values), self.parameters_index):
                new_args[index] = value
            return self.function(x, *new_args)
        return new_function

    def _save_fitted_parameters(self):
        class dotdict(OrderedDict):
            __getattr__ = dict.get
            __setattr__ = dict.__setitem__
            __delattr__ = dict.__delitem__

        out_value = iter(self.val)
        out_err = iter(self.err)
        max_name_length = max([len(name) for name in self.function_parameters])

        for name in self.function_parameters:
            self.params[name] = dotdict()
            if name in self.fixed_args_name:
                self.params[name]['val'] = self.function_parameters[name].default
                self.params[name]['err'] = 0
            else:
                self.params[name]['val'] = next(out_value)
                self.params[name]['err'] = next(out_err)
        return None

    def _calculate_function_parameters(self):
        self.function_parameters = OrderedDict(inspect.signature(self.function).parameters)
        self.fixed_values, self.fixed_index = [], []
        self.variable_values, self.variable_index  = [], []
        self.bounds, self.is_bounded = [], False

        self.function_parameters.popitem(last=False) #remove x
        if tuple(self.function_parameters.items())[-1][0]=='fixed_args': #check for fixed arguments
            self.fixed_args_name = self.function_parameters['fixed_args'].default
            self.function_parameters.popitem(last=True)
        else:
            self.fixed_args_name = []

        for index, param in enumerate(self.function_parameters):
            current_param = self.function_parameters[param]

            if current_param.name in self.fixed_args_name:
                self.fixed_index.append(index)
                self.fixed_values.append(self._get_first_item(current_param.default))
            else:          
                if current_param.default == inspect._empty:
                    value = 1
                    bound = [-np.inf, np.inf]
                elif isinstance(current_param.default, Sized):
                    value = current_param.default[0]
                    bound = [current_param.default[1], current_param.default[2]]
                    self.is_bounded = True
                else:
                    value = current_param.default
                    bound = [-np.inf, np.inf]

                self.bounds.append(bound)
                self.variable_index.append(index)
                self.variable_values.append(value)

        self.parameters_index = self.variable_index + self.fixed_index


    def _confidence_interval(self, x, confidence_probability=0.95): 
        grad_array = np.zeros(len(self.val))
        sqrt_machine_precision = np.sqrt(np.finfo(type(1.0)).eps)

        for index, _ in enumerate(self.val):
            start_param = self.val.copy()
            end_param = self.val.copy()
            start_param[index] = start_param[index] + sqrt_machine_precision
            end_param[index] = end_param[index] - sqrt_machine_precision
            grad_array[index] = (self.clean_function(x, *end_param)-self.clean_function(x, *start_param))/(2*sqrt_machine_precision)

        # Student's t distribution
        nfree = self.xdata.shape[0] - len(self.val)
        t = np.abs(scipy_stats_t.ppf(1-confidence_probability, nfree))

        # Confidence interal estimation: grad(fun)_all_param.T * covariance_matrix * grad(fun)_all_param
        # reduced chi squared is considered to be 1
        return np.sqrt(np.dot(np.dot(self.cov, grad_array), grad_array)) * t

    def round_sig(self, val, err, significative_digits=3):
        try:
            if err == 0:
                return (val, 0)
            elif not np.isfinite(err):
                return (val, np.inf)
            else:
                return (round(val, significative_digits-int(np.floor(np.log10(abs(err))))-1),
                        round(err, significative_digits-int(np.floor(np.log10(abs(err))))-1))
        except:
            print('Can not compute the error')
            return (val, 0)

    def _print_results(self):
        out_value = iter(self.val)
        out_err = iter(self.err)
        max_name_length = max([len(name) for name in self.function_parameters])

        for name in self.function_parameters:
            if name in self.fixed_args_name:
                print('{name:>{name_length}}:  {default:<7} Fixed'.format(name=name,
                    name_length=max_name_length, default=self.function_parameters[name].default))
            else:
                initial = (1 if self.function_parameters[name].default == inspect._empty else self.function_parameters[name].default)
                val, err = self.round_sig(next(out_value), next(out_err))
                print('{name:>{name_length}}:  {val:<7} +/- {err:<7} {perc_err:>9}  initial:{initial}'.format(name=name,
                    name_length=max_name_length,  val=val, err=err, perc_err='({:.1f}%)'.format(err/val*100), initial=initial))
        return None

    def _set_plot_range(self):
        if not self.plot_range_x:
            x_min, x_max = np.min(self.xdata), np.max(self.xdata)
            self.plot_range_x = (x_min - (x_max - x_min)*self.fit_plot_params['PADDING'],
                x_max + (x_max - x_min)*self.fit_plot_params['PADDING'])
        if not self.plot_range_y:
            y_min, y_max = np.min(self.ydata), np.max(self.ydata)
            self.plot_range_y = (y_min - (y_max - y_min)*self.fit_plot_params['PADDING'],
                y_max + (y_max - y_min)*self.fit_plot_params['PADDING']) 

    def _plotting_function(self, *args, **kwargs):
        if self.fig_ax:
            self.fig, self.ax = self.fig_ax
        else:
            self.fig, self.ax = plt.subplots(figsize=self.fit_plot_params['FIGSIZE'])

        self._set_plot_range()
        self.ax.set_xlim(self.plot_range_x)
        self.ax.set_ylim(self.plot_range_y)
        
        x_plot = np.linspace(*self.plot_range_x, num=self.fit_plot_params['PLOT_POINTS'])
        y_plot =  self.clean_function(x_plot, *self.val)
        self.ax.plot(x_plot,  y_plot, c=self.fit_plot_params['BASE_COLOR'], linewidth=self.fit_plot_params['LINEWIDTH'])
        
        # Plot 95% confidence interval
        y_error = np.array([self._confidence_interval(x, confidence_probability=0.95) for x in x_plot])
        self.ax.fill_between(x_plot, y_plot-y_error, y_plot+y_error, interpolate=True, color=self.fit_plot_params['FILL_BETWEEN_COLOR'])
        
        # Plot datapoints
        if 'sigma' in kwargs:
            self.ax.errorbar(self.xdata, self.ydata, kwargs['sigma'], linestyle='None', color=self.fit_plot_params['BASE_COLOR'] )

        self.ax.plot(self.xdata, self.ydata, linestyle='None',marker=self.fit_plot_params['MARKER'], mfc=self.fit_plot_params['MARKER_COLOR'], mec=self.fit_plot_params['BASE_COLOR'],
                 markersize = self.fit_plot_params['MARKERSIZE_BIG'] if self.xdata.shape[0]<100 else self.fit_plot_params['MARKERSIZE_SMALL'],
                 markeredgewidth = self.fit_plot_params['MARKEREDGEWIDTH_BIG'] if self.xdata.shape[0]<self.fit_plot_params['NUM_POINT_LIMIT'] else self.fit_plot_params['MARKEREDGEWIDTH_SMALL'])

        self.fig.patch.set_facecolor(self.fit_plot_params['BACKGROUND_COLOR'])
        self.ax.set_facecolor(self.fit_plot_params['BACKGROUND_COLOR'])
        self.ax.grid(b=self.fit_plot_params['GRID'], linestyle=self.fit_plot_params['GRID_STYLE'], linewidth=self.fit_plot_params['GRID_LINEWIDTH'])

    def _plotting_debug(self, *args, **kwargs):
        self.fig, self.axs = plt.subplots(2, 2, figsize=(self.fit_plot_params['FIGSIZE'][0], self.fit_plot_params['FIGSIZE'][1]))
        self._set_plot_range()

        x_plot = np.linspace(*self.plot_range_x, num=self.fit_plot_params['PLOT_POINTS'])
        y_plot =  self.clean_function(x_plot, *self.val)
        y_plot_p0 =  self.clean_function(x_plot, *self.variable_values)
        if 'sigma' in kwargs:
            self.axs[0, 0].errorbar(self.xdata, self.ydata, yerr=kwargs['sigma'], linestyle='None', color='red')
            self.axs[1, 0].errorbar(self.xdata, self.ydata, yerr=kwargs['sigma'], linestyle='None', color='red')
            self.axs[1, 1].errorbar(self.xdata, self.ydata, yerr=kwargs['sigma'], linestyle='None', color='red')
        self.axs[0, 0].scatter(self.xdata, self.ydata, label='data', color='purple')
        self.axs[1, 0].scatter(self.xdata, self.ydata, label='data', color='purple')
        self.axs[1, 1].scatter(self.xdata, self.ydata, label='data', color='purple')

        self.axs[0, 1].plot(x_plot, y_plot_p0, label='initial guess', color='green')
        self.axs[1, 0].plot(x_plot, y_plot_p0, label='initial guess', color='green')
        if self.fit_ok:
            self.axs[1, 1].plot(x_plot, y_plot, label='fit result', color='orange')
            self.axs[1, 1].set_xlim(np.min(self.xdata), np.max(self.xdata))
            self.axs[1, 1].set_ylim(np.min(self.ydata), np.max(self.ydata))

        [[ax.legend() for ax in v_ax] for v_ax in self.axs]

    def _plot_results(self):
        self.plotting_function(self.clean_function, self.val, self.xdata, self.ydata, **self.kwargs)






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

class fit_sine(fit):
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
            
        super(fit_sine, self).__init__(sine, x, y, sigma=sigma, plot_results=plot_results, print_results=print_results, **kwargs)


if __name__ == '__main__':
    def f(x, a=-1, b=6 , c=0.5, d=3):
        return a + b*x + c*x**2 + d*x**3

    xx = np.linspace(-1, 1, 50)
    yy = f(xx)+(np.random.random(xx.shape[0])-0.5)*2
    y_errs = (np.random.random(xx.shape[0])-0.5)*2

    def g(x, a, b=6 , c=0.5, d=3, fixed_args=['b']):
        return a + b*x + c*x**2 + d*x**3

    def my_plot(f, p0, x, y):
        xx = np.linspace(0,3, 30)
        yy = f(xx, *p0)
        plt.scatter(x, y)
        plt.plot(xx, yy, color='red')

    fit(g, xx, yy, debug=True, sigma=y_errs) # plot debug mode
    plt.show()

    fit(g, xx, yy, plotting_function=my_plot) # plot with custom function
    plt.show()

    fit(g, xx, yy) # default plot
    plt.show()