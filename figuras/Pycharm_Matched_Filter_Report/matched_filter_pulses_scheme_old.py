import matplotlib.pyplot as plt
import numpy as np
import scipy.signal

from matplotlib import rc

__author__ = 'ernesto'

# if use latex or mathtext
rc('text', usetex=False)

# pulse parameters
# pulse duration
Tb = 4
# number of samples of a singe pulse
npulse = Tb * 100
# the pulse is a Rayleigh distribution with stdev rsigma
rsigma = 1

# previous and posterior samples number
sprev = 200
spost = 200

# noise parameters: mean and sigma^2
nmean = 0
nsigma = 0.1

# pulse construction
t = np.linspace(0, Tb, npulse)
# pulse: Rayleigh distribution
p1 = (t / rsigma) * np.exp(-np.square(t) / (2*rsigma**2))
# subtract a rect for p[-1]=0
p1 -= np.arange(npulse)/(npulse-1)*p1[-1]
# normalization for unit maximum amplitude
p1 /= np.max(p1)

# matched filter impulse response
h1 = p1[::-1]

# x_R(t)
nxr = npulse+sprev+spost
xr1 = np.zeros(nxr)
xr1[sprev:sprev+npulse] = p1

# noise
nr1 = np.random.normal(nmean, np.sqrt(nsigma), (nxr, ))

# signal plus noise
yr1 = xr1 + nr1


# detected signal: filtered pulse
xd, zf = scipy.signal.lfilter(h1, 1, p1, axis=-1, zi=np.zeros(npulse-1))
xd = np.concatenate((xd, zf))

# filter implementation with convolution
xd_alt = scipy.signal.convolve(p1, h1, mode='full')

print(xd.shape)
print(xd_alt.shape)

sss = xd - xd_alt
print(np.amax(sss))

plt.subplot(3, 1, 1)
plt.plot(np.arange(nxr), xr1+nr1)
plt.subplot(3, 1, 2)
plt.plot(np.arange(npulse), h1)
plt.subplot(3, 1, 3)
plt.plot(xd)
plt.plot(zf, 'k')
plt.show()


###########################
###########################
###########################

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
    y[i*npulse:(i+1)*npulse] = ak[i] * p1

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
plt.plot(np.arange(npulse), 0.5*p1, 'k', linewidth=2)
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

# plt.savefig('digital_pam.eps', format='eps', bbox_inches='tight')
plt.show()
