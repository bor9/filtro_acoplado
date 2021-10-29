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
tau = 2
# fs: number of samples per time unit
fs = 100
# total number of samples of a pulse
npulse = tau * fs + 1  # +1 for odd number of samples
# number of samples of half pulse
nhpulse = int((npulse-1) / 2)
# sample indexes of centered pulse
center_indexes = np.arange(-nhpulse, nhpulse + 1)


# rectangular single pulse construction
# pulse: rectangular pulse of amplitude 1
ak = 1  # pulse amplitude
p = np.ones(npulse)

# compute tau eq = \int p^2(t)dt
tau_eq = scipy.integrate.trapz(np.square(p), x=None, dx=1/fs)

# impulses responses delays
t0 = 1.5 * tau  # in time units
t1 = -1.5 * tau  # in time units

# time samples bounds
nmin = int(-2.2 * npulse)
nmax = int(2.2 * npulse)

# index of the sample of the time origin (n=0)
ind_0 = -nmin
# total number of samples
nsamples = nmax - nmin
# samples indexes
n = np.arange(nmin, nmax)

# signal x(t): ak*p(t)
p0 = np.zeros(nsamples)
p0[center_indexes+ind_0] = p

# impulse responses: (1/tau)*(t-t0)
nt0 = int(t0 * fs)  # number of samples of the delay respect to the origin
ind_t0 = nt0 + ind_0  # number of samples of the delay respect to initial sample
# signal construction
h0 = np.zeros(nsamples)
h0[center_indexes+ind_t0] = p

nt1 = int(t1 * fs)  # number of samples of the delay respect to the origin
ind_t1 = nt1 + ind_0  # number of samples of the delay respect to initial sample
# signal construction
h1 = np.zeros(nsamples)
h1[center_indexes+ind_t1] = p

# PLOT

# axis parameters
# x and y axis limits
dx = 60
xmin = nmin - dx
xmax = nmax + dx
ymax = 2.2
ymin = -0.5

# vertical tick margin
vtm = 0.5
# horizontal tick margin
htm = 6
# y axis coordinate label margin
ym = 16

# font size
font_size = 12
font_size_frac = 14

plt.figure(0, figsize=(8, 3), frameon=False)

#################
# pulse p(t)
#################
ax = plt.subplot(2, 1, 1)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# length of the ticks for all subplot (5 pixels)
display_length = 6  # in pixels
htl, vtl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black'))
plt.annotate("", xytext=(0, ymin), xycoords='data', xy=(0, ymax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black'))

# pulse
plt.plot(n, p0, 'k', linewidth=2)
plt.plot(n, h1, 'b', linewidth=2)


# ticks
plt.plot([nt1, nt1], [0, vtl], 'b', linewidth=0.8)
# xlabels
plt.text(nhpulse, -vtm, r'$\frac{\tau}{2}$', fontsize=12, ha='center', va='baseline')
plt.text(-nhpulse, -vtm, r'$-\frac{\tau}{2}$', fontsize=12, ha='center', va='baseline')
plt.text(-htm, -vtm, r'$0$', fontsize=font_size, ha='right', va='baseline')
plt.text(xmax, -vtm, r'$u$', fontsize=font_size, ha='right', va='baseline')
plt.text(nt1, -vtm, r'$t-t_d$', fontsize=font_size, ha='center', va='baseline', color='blue')

plt.annotate(r'$p(u)$', xytext=(1.2*nhpulse, 1.8), xycoords='data',
             xy=(nhpulse-20, 1),
             textcoords='data', color='black', fontsize=font_size,
             arrowprops=dict(arrowstyle="->", relpos=(0, 0), color='black'),
             horizontalalignment='left', verticalalignment='baseline',
             )

plt.annotate(r'$t-t_d+\frac{\tau}{2}$', xytext=(nt1 + nhpulse + 10, -2.3 * vtm), xycoords='data', xy=(nt1 + nhpulse, 0),
             textcoords='data', color='blue',
             arrowprops=dict(width=0.1, headwidth=3, headlength=4, color='blue', shrink=0.1),
             horizontalalignment='center', verticalalignment='baseline')
plt.annotate(r'$t-t_d-\frac{\tau}{2}$', xytext=(nt1 - nhpulse - 10, -2.3 * vtm), xycoords='data', xy=(nt1 - nhpulse, 0),
             textcoords='data', color='blue',
             arrowprops=dict(width=0.1, headwidth=3, headlength=4, color='blue', shrink=0.1),
             horizontalalignment='center', verticalalignment='baseline')
plt.annotate(r'$p(u-[t-t_d])$', xytext=(1.2 * nhpulse + nt1 - 40, 1.8), xycoords='data',
             xy=(nhpulse + nt1 - 20 - 40, 1),
             textcoords='data', color='blue', fontsize=font_size,
             arrowprops=dict(arrowstyle="->", relpos=(0, 0), color='blue'),
             horizontalalignment='left', verticalalignment='baseline',
             )

plt.axis('off')


ax = plt.subplot(2, 1, 2)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# length of the ticks for all subplot (5 pixels)
display_length = 6  # in pixels
htl, vtl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black'))
plt.annotate("", xytext=(0, ymin), xycoords='data', xy=(0, ymax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black'))

# pulse
plt.plot(n, p0, 'k', linewidth=2)
plt.plot(n, h0, 'b', linewidth=2)

# ticks
plt.plot([nt0, nt0], [0, vtl], 'b', linewidth=0.8)
# xlabels
plt.text(nhpulse, -vtm, r'$\frac{\tau}{2}$', fontsize=12, ha='center', va='baseline')
plt.text(-nhpulse, -vtm, r'$-\frac{\tau}{2}$', fontsize=12, ha='center', va='baseline')
plt.text(-htm, -vtm, r'$0$', fontsize=font_size, ha='right', va='baseline')
plt.text(xmax, -vtm, r'$u$', fontsize=font_size, ha='right', va='baseline')
plt.text(nt0, -vtm, r'$t-t_d$', fontsize=font_size, ha='center', va='baseline', color='blue')

plt.annotate(r'$t-t_d+\frac{\tau}{2}$', xytext=(nt0 + nhpulse + 10, -2.3 * vtm), xycoords='data', xy=(nt0 + nhpulse, 0),
             textcoords='data', color='blue',
             arrowprops=dict(width=0.1, headwidth=3, headlength=4, color='blue', shrink=0.1),
             horizontalalignment='center', verticalalignment='baseline')
plt.annotate(r'$t-t_d-\frac{\tau}{2}$', xytext=(nt0 - nhpulse - 10, -2.3 * vtm), xycoords='data', xy=(nt0 - nhpulse, 0),
             textcoords='data', color='blue',
             arrowprops=dict(width=0.1, headwidth=3, headlength=4, color='blue', shrink=0.1),
             horizontalalignment='center', verticalalignment='baseline')

plt.annotate(r'$p(u)$', xytext=(1.2*nhpulse, 1.8), xycoords='data',
             xy=(nhpulse-20, 1),
             textcoords='data', color='black', fontsize=font_size,
             arrowprops=dict(arrowstyle="->", relpos=(0, 0), color='black'),
             horizontalalignment='left', verticalalignment='baseline',
             )

plt.annotate(r'$p(u-[t-t_d])$', xytext=(1.2 * nhpulse + nt0 - 70, 1.8), xycoords='data',
             xy=(nhpulse + nt0 - 20 - 70, 1),
             textcoords='data', color='blue', fontsize=font_size,
             arrowprops=dict(arrowstyle="->", relpos=(0, 0), color='blue'),
             horizontalalignment='left', verticalalignment='baseline',
             )

plt.axis('off')

plt.savefig('matched_filter_output_support.eps', format='eps', bbox_inches='tight')
plt.show()

###########################
###########################
##  Pulse output figure  ##
###########################
###########################

ak = 1.4

# impulse response: (1/tau)*p(t-t0)
t2 = tau/2
nt2 = int(t2 * fs)  # number of samples of the delay respect to the origin
ind_t2 = nt2 + ind_0  # number of samples of the delay respect to initial sample
# signal construction
h = np.zeros(nsamples)
h[center_indexes+ind_t2] = p/tau_eq

# detected signal construction
y = scipy.signal.lfilter(p/tau_eq, 1, ak*p0, axis=-1)
y = y * tau / npulse  # normalization to simulate continuous time convolution

plt.figure(0, figsize=(8, 6), frameon=False)

#################
# pulse p(t)
#################
ax = plt.subplot(4, 1, 1)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# length of the ticks for all subplot (5 pixels)
display_length = 6  # in pixels
htl, vtl = convert_display_to_data_coordinates(ax.transData, length=display_length)

plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black'))
plt.annotate("", xytext=(0, ymin), xycoords='data', xy=(0, ymax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black'))

# pulse
plt.plot(n, p0, 'k', linewidth=2)

# xlabels
plt.text(nhpulse, -vtm, r'$\frac{\tau}{2}$', fontsize=font_size_frac, ha='center', va='baseline')
plt.text(-nhpulse-1, -vtm, r'$-\frac{\tau}{2}$', fontsize=font_size_frac, ha='center', va='baseline')
plt.text(-htm, -vtm, r'$0$', fontsize=font_size, ha='right', va='baseline')
plt.text(xmax, -vtm, r'$t$', fontsize=font_size, ha='right', va='baseline')

# ylabels
plt.text(-htm, 1.05, r'$1$', fontsize=font_size, ha='right', va='bottom')

plt.text(ym, ymax, r'$p(t)$', fontsize=font_size, ha='left', va='center')

plt.axis('off')

#################
# input x(t)
#################
ax = plt.subplot(4, 1, 2)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black'))
plt.annotate("", xytext=(0, ymin), xycoords='data', xy=(0, ymax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black'))

# pulse
plt.plot(n, ak*p0, 'k', linewidth=2)

# xlabels
plt.text(nhpulse, -vtm, r'$\frac{\tau}{2}$', fontsize=font_size_frac, ha='center', va='baseline')
plt.text(-nhpulse-1, -vtm, r'$-\frac{\tau}{2}$', fontsize=font_size_frac, ha='center', va='baseline')
plt.text(-htm, -vtm, r'$0$', fontsize=font_size, ha='right', va='baseline')
plt.text(xmax, -vtm, r'$t$', fontsize=font_size, ha='right', va='baseline')
# ylabels
plt.text(-htm, ak+0.05, r'$a_k$', fontsize=font_size, ha='right', va='bottom')


plt.text(ym, ymax, r'$x(t)=a_kp(t)$', fontsize=font_size, ha='left', va='center')

plt.axis('off')

#########################
# impulse response h(t)
#########################
ax = plt.subplot(4, 1, 3)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black'))
plt.annotate("", xytext=(0, ymin), xycoords='data', xy=(0, ymax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black'))

# pulse
plt.plot(n, h, 'k', linewidth=2)

# xlabels
plt.text(nhpulse, -vtm, r'$t_d=\frac{\tau}{2}$', fontsize=12, ha='center', va='baseline')
plt.text(npulse, -vtm, r'$\tau$', fontsize=12, ha='center', va='baseline')
plt.text(-htm, -vtm, r'$0$', fontsize=font_size, ha='right', va='baseline')
plt.text(xmax, -vtm, r'$t$', fontsize=font_size, ha='right', va='baseline')
# ylabels
plt.text(-htm, 1/tau_eq, r'$\frac{1}{\tau_\mathrm{eq}}=\frac{1}{\tau}$', fontsize=font_size_frac, ha='right', va='center')
# ticks
plt.plot([nt2, nt2], [0, vtl], 'k', linewidth=0.8)

plt.text(ym, ymax-0.2, r'$h(t)=\frac{1}{\tau_\mathrm{eq}}p(t_d-t)$', fontsize=font_size, ha='left', va='center')

plt.axis('off')

#########################
# filter output y(t)
#########################
ax = plt.subplot(4, 1, 4)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black'))
plt.annotate("", xytext=(0, ymin), xycoords='data', xy=(0, ymax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black'))

# pulse
plt.plot(n, y, 'k', linewidth=2)

# xlabels
plt.text(nhpulse, -vtm, r'$t_d=\frac{\tau}{2}$', fontsize=12, ha='center', va='baseline')
plt.text(npulse+nhpulse, -vtm, r'$\frac{3\tau}{2}$', fontsize=font_size_frac, ha='center', va='baseline')
plt.text(-htm, -vtm, r'$0$', fontsize=font_size, ha='right', va='baseline')
plt.text(xmax, -vtm, r'$t$', fontsize=font_size, ha='right', va='baseline')
plt.text(-nhpulse-1, -vtm, r'$-\frac{\tau}{2}$', fontsize=font_size_frac, ha='center', va='baseline')
# ylabels
plt.text(-htm, ak, r'$a_k$', fontsize=font_size, ha='right', va='center')
# ticks
# plt.plot([nt2, nt2], [0, vtl], 'k', linewidth=0.8)
plt.plot((npulse+nhpulse)*np.ones((2,)), [0, vtl], 'k', linewidth=0.8)
plt.plot([-nt2-1, -nt2-1], [0, vtl], 'k', linewidth=0.8)
# plt.plot([0, htl], [ak, ak], 'k', linewidth=0.8)
# maximum
plt.plot([nt2, nt2], [0, ak], 'k--', linewidth=0.8, dashes=(5, 3))
plt.plot([0, nt2], [ak, ak], 'k--', linewidth=0.8, dashes=(5, 3))

plt.text(ym, ymax, r'$y(t)=(x*h)(t)$', fontsize=font_size, ha='left', va='center')

plt.axis('off')


plt.savefig('matched_filter_square_pulse_output.eps', format='eps', bbox_inches='tight')
plt.show()
