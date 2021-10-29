import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)

# pulse p(t) parameters
# pulse duration
Tb = 4
# fs: number of samples per time unit
fs = 50
# total number of samples of a pulse
npulse = Tb * fs

# noise parameters: mean and variance
nmean = 0
nvar = 0.4

# rectangular single pulse construction
# pulse: rectangular pulse of amplitude 1
p = np.ones(npulse)

# sequence of symbols
ak = np.array([0, 1, 0, 1, 1, 0, 0, 1, 0, 0])

# matching filter impulse response
h = p[::-1]

# pam signal construction
# number of symbols
nak = ak.shape[0]
# number of samples
nxr = nak * npulse
# samples index
n1 = np.arange(nxr)
# pam signal construction
xr = np.empty(nxr)
for i in range(nak):
    xr[i * npulse:(i + 1) * npulse] = p if ak[i] == 1 else -p

# recieved signal: recieved pulse + noise
yr = xr + np.random.normal(nmean, np.sqrt(nvar), (nxr,))

# detected signal construction
# yd = scipy.signal.lfilter(h, 1, yr, axis=-1)
# with convolution to get full signal
yd = scipy.signal.convolve(yr, h, mode='full')
yd = yd * Tb / npulse / 2 # normalization to simulate continuous time convolution
# number of samples of the output (yd.shape)
nyd = nxr + npulse - 1
n2 = np.arange(nyd)

# optimum sample instant for bits detection
opt_sample_instants = np.arange(1, nak+1)*npulse-1
# detected bits sequence
ak_det = np.empty(nak, dtype=int)
for i in np.arange(nak):
    ak_det[i] = 0 if yd[opt_sample_instants[i]] < 0 else 1

# PLOT

# axis parameters
# x and y axis limits
# time samples bounds
nmin = 0
nmax = nyd - 1

dx = 60
xmin = nmin - dx
xmax = nmax + dx
ymax = 3
ymin = -3

# vertical tick margin
vtm = 1
# horizontal tick margin
htm = 12
# y axis coordinate label margin
ym = 20

# font size
font_size = 12

plt.figure(1, figsize=(8, 5), frameon=False)

##########
# x_R(t) #
##########
ax = plt.subplot(3, 1, 1)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# axis arrows
plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, color=[0.5, 0.5, 0.5]))

# vertical lines of bit intervals
for i in np.arange(nak+1):
    plt.plot((i*npulse-1)*np.ones(2), [ymin, ymax], 'k--', dashes=(5, 3), linewidth=0.5)
# bits
for i in np.arange(nak):
    plt.text(((i+0.5)*npulse-1), ymax, r'$'+str(ak[i])+'$', fontsize=font_size, ha='center', va='center')


# signal
plt.plot(n1, xr, 'k', linewidth=1)

# xlabels
plt.text(xmax, -vtm, r'$t$', fontsize=font_size, ha='right', va='baseline')
# ylabels
plt.text(xmin-120, 0.6, r'$x_R(t)$', fontsize=font_size, ha='left', va='baseline')

plt.axis('off')

##########
# y_D(t) #
##########
plt.subplot(3, 1, 2)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# axis arrows
plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, color=[0.5, 0.5, 0.5]))

# pulse
plt.plot(n1, yr, 'k', linewidth=1)
# vertical lines of bit intervals
for i in np.arange(nak+1):
    plt.plot((i*npulse-1)*np.ones(2), [ymin, ymax], 'k--', dashes=(5, 3), linewidth=0.5)

# xlabels
plt.text(xmax, -vtm, r'$t$', fontsize=font_size, ha='right', va='baseline')
# ylabels
plt.text(xmin-120, 0.6, r'$y_R(t)$', fontsize=font_size, ha='left', va='baseline')

plt.axis('off')

##########
# y_D(t) #
##########
plt.subplot(3, 1, 3)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# axis arrows
plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, color=[0.5, 0.5, 0.5]))

# p(-t)
plt.plot(n2, yd, 'k', linewidth=1)
# vertical lines of bit intervals
for i in np.arange(nak+1):
    plt.plot((i*npulse-1)*np.ones(2), [ymin, ymax], 'k--', dashes=(5, 3), linewidth=0.5)

# xlabels
plt.text(xmax, -vtm, r'$t$', fontsize=font_size, ha='right', va='baseline')
# ylabels
plt.text(xmin-120, 0.6, r'$y_D(t)$', fontsize=font_size, ha='left', va='baseline')

plt.plot(opt_sample_instants, yd[opt_sample_instants], 'r.', markersize=8)
# bits
for i in np.arange(nak):
    plt.text(opt_sample_instants[i], ymin-0.4, r'${}$'.format(ak_det[i]), fontsize=font_size, ha='center', va='top')

plt.axis('off')

plt.savefig('matched_filter_operating.eps', format='eps', bbox_inches='tight')
plt.show()

