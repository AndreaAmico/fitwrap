import numpy as np
import matplotlib.pyplot as plt
import fitwrap as fw



### TEST FIT 1D #######################################################################################

SIZE = 100
random_noise = np.ones(SIZE) * 2**22
for i in range(SIZE-1):
    random_noise[i+1] = (random_noise[i]*1664525 + 1013904223) % 2**23 # from numerical recepies
random_noise = random_noise/2**23

xx = np.linspace(1,5,SIZE)
yy = 5*xx**2 + random_noise/10

def sq(x, a=4):
	return a * x**2

out = fw.fit(sq, xx, yy, print_results=False)

if out.val[0] == 5.002784126698641:
	print('Test fit value PASSED')
else:
	print('Test fit value FAILED')

if out.err[0] == 0.0003096773812548323:
	print('Test fit error PASSED')
else:
	print('Test fit error FAILED')


### TEST FIT 2D #######################################################################################

def g2(yx, x0=21, y0=32, sx=8, sy=5):
    return np.exp(-(yx[1]-x0)**2/(2*sx**2) -(yx[0]-y0)**2/(2*sy**2))
    
def gf(yx, x0=21, y0=32, sx=8, sy=(9,8.5,10)):
    return np.ravel(np.exp(-(yx[1]-x0)**2/(2*sx**2) -(yx[0]-y0)**2/(2*sy**2)))
    
yx = np.mgrid[:100,:40]
SIZE = 100*40

random_noise = np.ones(SIZE) * 2**22
for i in range(SIZE-1):

    random_noise[i+1] = (random_noise[i]*1664525 + 1013904223) % 2**23 # from numerical recepies
random_noise = random_noise/2**23


gg = g2(yx, 20, 30, 5, 5)+(0.5-random_noise.reshape(yx[0].shape))*0.2
out = fw.fit2d(gf, yx, gg, print_results=False)


if all([fit == expected for fit, expected in zip(out.val,
	[20.035323877504496, 30.051743101909096, 3.850099586254743, 8.500000000000002])]):
	print('Test fit 2D value PASSED')
else:
	print('Test fit 2D value FAILED')


if all([fit == expected for fit, expected in zip(out.err,
	[0.04597313775168646, 0.10149675561129788, 0.03981399732057903, 0.08790081271428242])]):
	print('Test fit 2D error PASSED')
else:
	print('Test fit 2D error FAILED')

### TEST SINE  #######################################################################################

def my_signal(t, off, amp, freq, phase):
    return  off + amp * np.sin(2*np.pi*freq*t + phase)

SIZE = 100
random_noise = np.ones(SIZE) * 2**22
for i in range(SIZE-1):
    random_noise[i+1] = (random_noise[i]*1664525 + 1013904223) % 2**23 # from numerical recepies
random_noise = random_noise/2**23

times = random_noise
signals = my_signal(times, 0.5, 2, 4, np.pi/4) + (random_noise-0.5)*2
errors = np.abs(my_signal(times, 0, 1, 4, np.pi/2))/2+0.01


out = fw.fit_sin(times, signals, errors, print_results=False)

if all([fit == expected for fit, expected in zip(out.val,
	[0.5626104290061619, 2.127634081268933, 3.8293300079383035, 1.393917602190907])]):
	print('Test fit sin value PASSED')
else:
	print('Test fit sin value FAILED')


if all([fit == expected for fit, expected in zip(out.err,
	[0.049135393665772215, 0.1332139043241083, 0.019244487872042058, 0.03284413398254605])]):
	print('Test fit sin error PASSED')
else:
	print('Test fit sin error FAILED')

### TEST GAUSS  #######################################################################################

def my_signal(x, off, amp, x0, sx):
    return  off + amp * np.exp(-(x-x0)**2 / (2*sx**2))

SIZE = 100
random_noise = np.ones(SIZE) * 2**22
for i in range(SIZE-1):
    random_noise[i+1] = (random_noise[i]*1664525 + 1013904223) % 2**23 # from numerical recepies
random_noise = random_noise/2**23

xx = np.linspace(0, 10, SIZE)
signals = my_signal(xx, 0.5, 8, 4, 1) + (random_noise-0.5)*2

out = fw.fit_gauss(xx, signals, print_results=False)

if all([fit == expected for fit, expected in zip(out.val,
	[0.2811854994469001, 8.172019362078377, 3.988140408772186, 1.0245154822192815])]):
	print('Test fit gauss value PASSED')
else:
	print('Test fit gauss value FAILED')


if all([fit == expected for fit, expected in zip(out.err,
	[0.08458826476316875, 0.1764090991808999, 0.02402102813958344, 0.028307724693133765])]):
	print('Test fit gauss error PASSED')
else:
	print('Test fit gauss error FAILED')