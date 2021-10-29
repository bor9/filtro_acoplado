import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)


# auxiliar function for plot ticks of equal length in x and y axis despite its scales.
def convert_display_to_data_coordinates(transData, length=10):
    # create a transform which will take from display to data coordinates
    inv = transData.inverted()
    # transform from display coordinates to data coordinates in x axis
    data_coords = inv.transform([(0, 0), (length, 0)])
    # get the length of the segment in data units
    x_coord_len = data_coords[1, 0] - data_coords[0, 0]
    # transform from display coordinates to data coordinates in y axis
    data_coords = inv.transform([(0, 0), (0, length)])
    # get the length of the segment in data units
    y_coord_len = data_coords[1, 1] - data_coords[0, 1]
    return x_coord_len, y_coord_len

###############################################
# plot of waveforms for matched filter scheme #
###############################################

# pulse p(t) parameters
# pulse duration
Tb = 4
# fs: number of samples per time unit
fs = 50
# total number of samples of a pulse
npulse = Tb * fs

# previous and posterior samples number
nprev = 150
npost = 150

# noise parameters: mean and variance
nmean = 0
nvar = 0.13

# rectangular single pulse construction
# pulse: rectangular pulse of amplitude 1
p1 = np.ones(npulse)

# x_R(t): received pulse construction. add zeros before and after
nxr = npulse + nprev + npost  # total number of samples with zero padding
xr1 = np.zeros(nxr)
xr1[nprev:nprev + npulse] = 1

# additive white gaussian noise signal
nr1 = np.random.normal(nmean, np.sqrt(nvar), (nxr,))

# signal plus noise
yr1 = xr1 + nr1

# matched filter impulse response
h1 = p1[::-1]

# filter implementation with convolution
yd1 = scipy.signal.convolve(yr1, h1, mode='full')
# normalization for amplitude 1.5 (arbitrary)
yd1 = 1.5 * yd1 / npulse

# pulse sample indexes
n1 = np.arange(nxr)
# number of samples of the filtered signal
nyd = nxr+npulse-1
# filtered signal sample indexes
n2 = np.arange(nyd)

axis = [0, nyd, -1, 2]
font_size = 25

fig = plt.figure(0, figsize=(4, 6), frameon=False)
plt.subplot(3, 1, 1)
plt.plot(n1, xr1, 'k', linewidth=2)
plt.text(nprev, -0.1, r'$t_0$', fontsize=font_size, ha='center', va='top')
plt.text(nprev-20, 1, r'$A_p$', fontsize=font_size, ha='right', va='center')
plt.axis(axis)
plt.axis('off')
plt.subplot(3, 1, 2)
plt.plot(n1, yr1, 'k', linewidth=1.5)
plt.axis(axis)
plt.axis('off')
plt.subplot(3, 1, 3)
plt.plot(n2, yd1, 'k', linewidth=2)
plt.plot([nprev+npulse, nprev+npulse], [0, yd1[nprev+npulse]], 'k--', dashes=(5, 3))
plt.text(nprev+npulse, -0.1, r'$t_0+t_d$', fontsize=font_size, ha='center', va='top')
plt.text(nprev+npulse-60, 1.5, r'$A$', fontsize=font_size, ha='right', va='center')
plt.axis(axis)
plt.axis('off')
plt.savefig('matched_filter_scheme_waveforms.eps', format='eps', bbox_inches='tight')
plt.show()

######################################################
# matched filter waveforms for operation explanation #
######################################################

# pulse parameters
# the pulse has the forma of a Rayleigh distribution with stdev rsigma
rsigma = 1
# pulse construction
t = np.linspace(0, Tb, npulse)
# pulse: Rayleigh distribution
p2 = (t / rsigma) * np.exp(-np.square(t) / (2*rsigma**2))
# subtract a rect for p[-1]=0
p2 -= np.arange(npulse)/(npulse-1)*p2[-1]
# normalization for unit maximum amplitude
p2 /= np.max(p2)

# recieved pulse delay
t0 = 1.6 * Tb  # in time units

# noise parameters: mean and sigma^2
nmean = 0
nvar = 0.1

# matching filter impulse response
h2 = p2[::-1]

# time samples bounds
nmin = int(-1.5 * npulse)
nmax = int(4 * npulse)
# number of sample of time origin (n=0)
ind_0 = -nmin
# total number of samples
nsamples = nmax - nmin
# samples indexes
n = np.arange(nmin, nmax)

# pulse p(t) construction
p = np.zeros(nsamples)
p[ind_0:ind_0 + npulse] = 1.5 * p2

# recieved pulse: p(t-t_0)
nt0 = int(t0 * fs)  # number of samples of the delay respect to the origin
ind_t0 = nt0 + ind_0  # number of samples of the delay respect to initial sample
# signal construction
xr = np.zeros(nsamples)
xr[ind_t0: ind_t0 + npulse] = p2

# impulse responses
# p(-t) construction
h = np.zeros(nsamples)
h[ind_0 - npulse: ind_0] = 1.5 * h2
# h(t)=p(t_d-t_0) construction (with t_d=T_b)
h_causal = np.zeros(nsamples)
h_causal[ind_0:ind_0 + npulse] = 1.5 * h2

# detected signal construction
xd = scipy.signal.lfilter(1.5 * h2, 1, xr, axis=-1)
xd = xd * Tb / npulse  # normalization to simulate continuous time convolution

# recieved signal: recieved pulse + noise (not used in the plot)
yr = xr + np.random.normal(nmean, np.sqrt(nvar), (nsamples,))

# detected signal: recieved pulse filtered (not used in the plot)
yd = scipy.signal.lfilter(1.5 * h2, 1, yr, axis=-1)
yd = yd * Tb * 1.5 / npulse

# PLOT

# axis parameters
# x and y axis limits
dx = 60
xmin = nmin - dx
xmax = nmax + dx
ymax = 2.5
ymin = -0.5

# vertical tick margin
vtm = 0.45
# horizontal tick margin
htm = 12
# y axis coordinate label margin
ym = 20

# font size
font_size = 12

plt.figure(1, figsize=(8, 7), frameon=False)

#################
# pulse p(t)
#################
ax = plt.subplot(4, 1, 1)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# length of the ticks for all subplot (5 pixels)
display_length = 5  # in pixels
htl, vtl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black'))
plt.annotate("", xytext=(0, ymin), xycoords='data', xy=(0, ymax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black'))

# pulse
plt.plot(n, p, 'k', linewidth=2)
# ticks
plt.plot([npulse, npulse], [0, vtl], 'k', linewidth=0.8)
# xlabels
plt.text(npulse, -vtm, r'$T_b$', fontsize=font_size, ha='center', va='baseline')
plt.text(-htm, -vtm, r'$0$', fontsize=font_size, ha='right', va='baseline')
plt.text(xmax, -vtm, r'$t$', fontsize=font_size, ha='right', va='baseline')
# ylabel
plt.text(ym, ymax, r'$p(t)$', fontsize=font_size, ha='left', va='center')

plt.plot([0, htl], [1.5, 1.5], 'k', linewidth=0.8)
plt.text(-htm, 1.5, r'$1$', fontsize=font_size, ha='right', va='center')

plt.axis('off')

#################
# received pulse
#################
plt.subplot(4, 1, 2)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# axis arrows
plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black'))
plt.annotate("", xytext=(0, ymin), xycoords='data', xy=(0, ymax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black'))

# pulse
plt.plot(n, xr, 'k', linewidth=2)
# ticks
plt.plot([nt0, nt0], [0, vtl], 'k', linewidth=0.8)
plt.plot([0, htl], [1, 1], 'k', linewidth=0.8)
# xlabels
plt.text(nt0, -vtm, r'$t_0$', fontsize=font_size, ha='center', va='baseline')
plt.text(-htm, -vtm, r'$0$', fontsize=font_size, ha='right', va='baseline')
plt.text(xmax, -vtm, r'$t$', fontsize=font_size, ha='right', va='baseline')
# ylabel
plt.text(ym, ymax, r'$x_R(t)=A_pp(t-t_0)$', fontsize=font_size, ha='left', va='center')
plt.text(-htm, 1, r'$A_p$', fontsize=font_size, ha='right', va='center')

plt.axis('off')

#################
# h(t)
#################
plt.subplot(4, 1, 3)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# axis arrows
plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black'))
plt.annotate("", xytext=(0, ymin), xycoords='data', xy=(0, ymax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black'))

# p(-t)
plt.plot(n, h, 'b', linewidth=1)
# p(t_d-t)
plt.plot(n, h_causal, 'k', linewidth=2)
# ticks
plt.plot([npulse, npulse], [0, vtl], 'k', linewidth=0.8)
plt.plot([-npulse, -npulse], [0, vtl], 'k', linewidth=0.8)
# xlabels
plt.text(npulse, -vtm, r'$T_b$', fontsize=font_size, ha='center', va='baseline')
plt.text(-npulse, -vtm, r'$-T_b$', fontsize=font_size, ha='center', va='baseline')
plt.text(-htm, -vtm, r'$0$', fontsize=font_size, ha='right', va='baseline')
plt.text(xmax, -vtm, r'$t$', fontsize=font_size, ha='right', va='baseline')
# ylabel
plt.text(ym, ymax, r'$h(t)=p(t_d-t)$', fontsize=font_size, ha='left', va='center')
plt.text(xmax, ymax, r'$t_d = T_b$', fontsize=font_size, ha='right', va='center')

plt.plot([0, htl], [1.5, 1.5], 'k', linewidth=0.8)
plt.text(-htm, 1.5, r'$1$', fontsize=font_size, ha='right', va='center')
# p(-t) annotate
plt.annotate(r'$p(-t)$', xytext=(-140-80, 0.8+h[ind_0 - 80]), xycoords='data', xy=(-80, h[ind_0 - 80]),
             textcoords='data', color='blue',
             arrowprops=dict(arrowstyle="->", relpos=(0, 1), color='blue'))
plt.axis('off')

#################
# detected pulse
#################
# ax = plt.subplot2grid((8, 1), (6, 0), rowspan=2)
plt.subplot(4, 1, 4)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# axis arrows
plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black'))
plt.annotate("", xytext=(0, ymin), xycoords='data', xy=(0, ymax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black'))

# pulse
plt.plot(n, xr, 'b', linewidth=1)
plt.plot(n, xd, 'k', linewidth=2)
# ticks
plt.plot([nt0, nt0], [0, vtl], 'k', linewidth=0.8)
plt.plot([nt0+npulse, nt0+npulse], [0, vtl], 'k', linewidth=0.8)
# xlabels
plt.text(nt0, -vtm, r'$t_0$', fontsize=font_size, ha='center', va='baseline')
plt.text(nt0 + npulse, -vtm, r'$t_0+t_d$', fontsize=font_size, ha='center', va='baseline')
plt.text(-htm, -vtm, r'$0$', fontsize=font_size, ha='right', va='baseline')
plt.text(xmax, -vtm, r'$t$', fontsize=font_size, ha='right', va='baseline')
# ylabel
plt.text(ym, ymax, r'$x_D(t)=(x_R*h)(t)$', fontsize=font_size, ha='left', va='center')
plt.axis('off')
# xr annotate
plt.annotate(r'$x_R(t)$', xytext=(-140 + nt0 + 30, 0.8 + xr[ind_t0 + 30]), xycoords='data', xy=(nt0+30, xr[ind_t0+30]),
             textcoords='data', color='blue',
             arrowprops=dict(arrowstyle="->", relpos=(0, 1), color='blue'))

plt.plot([0, htl], [xd[ind_t0 + npulse], xd[ind_t0 + npulse]], 'k', linewidth=0.8)
plt.text(-htm, xd[ind_t0 + npulse], r'$A$', fontsize=font_size, ha='right', va='center')

# plt.tight_layout()
# plt.savefig('matched_filter_waveforms.eps', format='eps')
plt.savefig('matched_filter_waveforms.eps', format='eps', bbox_inches='tight')
plt.show()

