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


# pulse parameters
# pulse duration
Tb = 4
# number of samples of a singe pulse
npulse = Tb * 100
# the pulse is a Rayleigh distribution with stdev sigma
sigma = 1

# pulse construction
t = np.linspace(0, Tb, npulse)
# pulse: Rayleigh distribution
p = (t / sigma) * np.exp(-np.square(t) / (2*sigma**2))
# subtract a rect for p[-1]=0
p -= np.arange(npulse)/(npulse-1)*p[-1]
# normalization for unit maximum amplitude
p /= np.max(p)

# matching filter impulse response
tau_eq = 0.8
h = p[::-1] / tau_eq

# sequence parameters
# sequence of amplitudes
ak = np.array([1, 0, 0, 1, 1, 0, 1])
# first pulse index
fi = 0
# alpha
alpha = 0
# Amplitude
A = 1.8

# sequence construction
# number of symbols
nak = ak.shape[0]
n = np.arange(-fi*npulse+alpha, (nak-fi)*npulse+alpha)
y_u = np.empty(nak*npulse)
y_p = np.empty(nak*npulse)
for i in range(nak):
    y_u[i * npulse:(i + 1) * npulse] = A * ak[i] * p
    y_p[i * npulse:(i + 1) * npulse] = (A/2 if ak[i] == 1 else -A/2) * p


# filter implementation with convolution
y_ud = scipy.signal.convolve(y_u, h, mode='full')
y_pd = scipy.signal.convolve(y_p, h, mode='full')
# normalization for amplitude 1.5 (arbitrary)
y_ud /= np.sum(p ** 2) / tau_eq
y_pd /= np.sum(p ** 2) / tau_eq


# PLOT

# axis parameters
dx = 2 * 80
xmin = n[0] - dx
xmax = n[-1] + dx
ymax = 2.4
ymin = -2

# vertical tick margin
vtm = 0.5
# horizontal tick margin
htm = 15
# font size
font_size = 12

# length of the ticks for all subplot (5 pixels)
display_length = 7  # in pixels

fig = plt.figure(1, figsize=(8, 6), frameon=False)

ax = plt.subplot2grid((6, 1), (0, 0), rowspan=2)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# horizontal and vertical ticks length
htl, vtl = convert_display_to_data_coordinates(ax.transData, length=display_length)

#############
# p(t) plot #
#############

# axis arrows
plt.annotate("", xytext=(-dx, 0), xycoords='data', xy=(npulse+3*dx, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, -0.5), xycoords='data', xy=(0, ymax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# pulse
plt.plot(np.arange(npulse), p, 'k', linewidth=2)
# xticks
plt.plot([npulse, npulse], [0, vtl], 'k', linewidth=0.8)
# xlabels
plt.text(npulse, -vtm, r'$D$', fontsize=font_size, ha='center', va='baseline')
plt.text(-15, -vtm, r'$0$', fontsize=font_size, ha='right', va='baseline')
plt.text(npulse+3*dx, -vtm, r'$t$', fontsize=font_size, ha='right', va='baseline')
# ylabels
plt.text(60, ymax, r'$p(t)$', fontsize=font_size, ha='left', va='center')
plt.text(-htm, 1, r'$1$', fontsize=font_size, ha='right', va='center')
# yticks
plt.plot([0, htl], [1, 1], 'k', linewidth=0.8)

plt.axis('off')

#############
# h(t) plot #
#############

# axis arrows
dp = 4
plt.annotate("", xytext=(dp * npulse - dx, 0), xycoords='data', xy=((dp + 1) * npulse+3*dx, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(dp * npulse, -0.5), xycoords='data', xy=(dp * npulse, ymax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# h(t)
plt.plot(np.arange(npulse) + dp * npulse, h, 'k', linewidth=2)

# xticks
plt.plot([(dp + 1) * npulse, (dp + 1) * npulse], [0, vtl], 'k', linewidth=0.8)
# xlabels
plt.text((dp + 1) * npulse, -vtm, r'$D$', fontsize=font_size, ha='center', va='baseline')
plt.text(dp * npulse - 15, -vtm, r'$0$', fontsize=font_size, ha='right', va='baseline')
plt.text((dp + 1) * npulse + 3 * dx, -vtm, r'$t$', fontsize=font_size, ha='right', va='baseline')
# ylabels
plt.text(dp * npulse + 60, ymax, r'$h(t)=\frac{1}{\tau_\mathrm{eq}}p(t_d-t),$', fontsize=font_size, ha='left',
         va='center')
plt.text(dp * npulse + 800, ymax, r'$\mathrm{con}\; t_d=D$', fontsize=font_size, ha='left', va='center')
plt.text(dp * npulse - htm, 1/tau_eq, r'$\frac{1}{\tau_\mathrm{eq}}$', fontsize=font_size, ha='right', va='center')
# yticks
plt.plot([dp * npulse, dp * npulse + htl], [1/tau_eq, 1/tau_eq], 'k', linewidth=0.8)

plt.axis('off')

###################
# pam signal plot #
###################

ax = plt.subplot2grid((6, 1), (2, 0), rowspan=2)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# axis arrows
plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin), xycoords='data', xy=(0, ymax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# pulse sequence
ax_unipolar, = plt.plot(n, y_u, 'k', linewidth=2)
ax_polar, = plt.plot(n, y_p, 'b', linewidth=2)

# xlabels
plt.text(xmax, -vtm, r'$t$', fontsize=font_size, ha='right', va='baseline')
# ylabels
plt.text(60, ymax, r'$x(t)$', fontsize=font_size, ha='left', va='center')
plt.text(-htm, A, r'$A$', fontsize=font_size, ha='right', va='center')
plt.text(-htm, A/2, r'$\frac{A}{2}$', fontsize=font_size, ha='right', va='center', color='b')
plt.text(-htm, -A/2, r'$-\frac{A}{2}$', fontsize=font_size, ha='right', va='center', color='b')
# yticks
plt.plot([0, htl], [A, A], 'k', linewidth=0.8)
plt.plot([0, htl], [A/2, A/2], 'b', linewidth=0.8)
plt.plot([0, htl], [-A/2, -A/2], 'b', linewidth=0.8)

# plt.legend([ax_unipolar, ax_polar], ["Unipolar", "Polar"], loc=4, prop={'size': 10}, frameon=False)
plt.legend([ax_unipolar, ax_polar], ["Unipolar", "Polar"], bbox_to_anchor=(1.02, 0.3),
           prop={'size': 10}, frameon=False)


# vertical lines of pulses starts
for i in np.arange(nak):
    plt.plot(i * npulse * np.ones(2), [ymin, ymax], 'k--', dashes=(5, 3), linewidth=0.5)

plt.axis('off')

########################
# detected signal plot #
########################
ax = plt.subplot2grid((6, 1), (4, 0), rowspan=2)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# axis arrows
plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin), xycoords='data', xy=(0, ymax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

n_end = y_ud.shape[0] - 320
# pulse sequence
plt.plot(np.arange(n_end), y_ud[:n_end], 'k', linewidth=2)
plt.plot(np.arange(n_end), y_pd[:n_end], 'b', linewidth=2)

# ticks
plt.plot([alpha]*2, [0, vtl], 'k', linewidth=0.8)
# xlabels
plt.text(xmax, -vtm, r'$t$', fontsize=font_size, ha='right', va='baseline')
# ylabels
plt.text(60, ymax, r'$y(t)$', fontsize=font_size, ha='left', va='center')
plt.text(-htm, A, r'$A$', fontsize=font_size, ha='right', va='center')
plt.text(-htm, A/2, r'$\frac{A}{2}$', fontsize=font_size, ha='right', va='center', color='b')
plt.text(-htm, -A/2, r'$-\frac{A}{2}$', fontsize=font_size, ha='right', va='center', color='b')
# yticks
plt.plot([0, htl], [A, A], 'k', linewidth=0.8)
plt.plot([0, htl], [A/2, A/2], 'b', linewidth=0.8)
plt.plot([0, htl], [-A/2, -A/2], 'b', linewidth=0.8)

# vertical lines of pulses starts
for i in np.arange(1, nak + 1):
    plt.plot(i * npulse * np.ones(2), [ymin, ymax], 'k--', dashes=(5, 3), linewidth=0.5)

# sample values
sample_instants = np.arange(1, nak + 1) * npulse
plt.plot(sample_instants, y_ud[sample_instants], 'ks', markersize=5)
plt.plot(sample_instants, y_pd[sample_instants], 'bs', markeredgecolor='b', markersize=5)

plt.axis('off')

plt.savefig('matched_filter_pam_output.eps', format='eps', bbox_inches='tight')
plt.show()
