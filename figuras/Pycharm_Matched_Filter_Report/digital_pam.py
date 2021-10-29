import matplotlib.pyplot as plt
import numpy as np

from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)

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

# sequence parameters
# sequence of amplitude
ak = np.array([0.3, -0.8, 0.7, 0.4, -0.5, 1, 0.3])
# first pulse index
fi = 3
# alpha
alpha = np.round(npulse/4)

# sequence construction
# number of symbols
nak = ak.shape[0]
n = np.arange(-fi*npulse+alpha, (nak-fi)*npulse+alpha)
y = np.empty(nak*npulse)
for i in range(nak):
    y[i*npulse:(i+1)*npulse] = ak[i] * p

# PLOT

# axis parameters
dx = 80
xmin = n[0]- dx
xmax = n[-1] + dx
ymax = 1.2
ymin = -1

# vertical ticks length
vtl = (ymax-ymin)/20
# vertical tick margin
vtm = 0.25
# font size
font_size = 12

fig = plt.figure(1, figsize=(8, 4), frameon=False)

ax = plt.subplot2grid((4, 1), (0, 0), rowspan=2)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# axis arrows
plt.annotate("", xytext=(-3*dx, 0), xycoords='data', xy=(npulse+3*dx, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, -0.5), xycoords='data', xy=(0, ymax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# pulse
plt.plot(np.arange(npulse), 0.5*p, 'k', linewidth=2)
# ticks
plt.plot([npulse, npulse], [0, vtl], 'k', linewidth=0.8)
# xlabels
plt.text(npulse, -vtm, r'$T_b$', fontsize=font_size, ha='center', va='baseline')
plt.text(-15, -vtm, r'$0$', fontsize=font_size, ha='right', va='baseline')
plt.text(npulse+3*dx, -vtm, r'$t$', fontsize=font_size, ha='right', va='baseline')
# ylabel
plt.text(60, ymax, r'$p(t)$', fontsize=font_size, ha='left', va='center')
plt.axis('off')


ax = plt.subplot2grid((4, 1), (2, 0), rowspan=2)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# axis arrows
plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, ymin), xycoords='data', xy=(0, ymax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

# pulse sequence
plt.plot(n, y, 'k', linewidth=2)

# ticks
plt.plot([alpha]*2, [0, vtl], 'k', linewidth=0.8)
# xlabels
plt.text(alpha, -vtm, r'$\alpha$', fontsize=font_size, ha='center', va='baseline')
plt.text(-15, -vtm, r'$0$', fontsize=font_size, ha='right', va='baseline')
plt.text(xmax, -vtm, r'$t$', fontsize=font_size, ha='right', va='baseline')
# ylabel
plt.text(60, ymax, r'$y(t)$', fontsize=font_size, ha='left', va='center')

plt.axis('off')

plt.savefig('digital_pam.eps', format='eps', bbox_inches='tight')
plt.show()
