import matplotlib.pyplot as plt
import matplotlib.mlab as mlab
import numpy as np
import math

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

#####################################
# PARAMETERS - This can be modified #
#####################################

# gaussian noise standard deviation
sigma = 1
# levels separation
A = 3.5 * sigma
# maximum deviation from the mean where to plot each gaussian
max_mean_dev = 3 * sigma
# number of symbols
M = 4  # even number

#####################
# END OF PARAMETERS #
#####################

# maximum value of the pdf - value in the mean
pdf_max = mlab.normpdf(0, 0, sigma)
# value of the pdf in the threshold
pdf_thres = mlab.normpdf(A/2, 0, sigma)

# means vector
mus = np.linspace(-(M-1)*A/2, (M-1)*A/2, M, endpoint=True)
# thresholds vector
thres = np.linspace(-(M-2)*A/2, (M-2)*A/2, M-1, endpoint=True)

# axis parameters
dx = A / 4
xmin = mus[0] - max_mean_dev - dx
xmax = mus[-1] + max_mean_dev + dx
ymax = pdf_max * 1.4
ymin = -pdf_max * 0.1

# vertical tick margin
vtm = -0.06
# horizontal tick margin
htm = 15
# font size
font_size = 12

# length of the ticks for all subplot (5 pixels)
display_length = 5  # in pixels

fig = plt.figure(1, figsize=(8, 2.5), frameon=False)

plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

# horizontal and vertical ticks length
ax = plt.gca()
htl, vtl = convert_display_to_data_coordinates(ax.transData, length=display_length)

# axis arrows
plt.annotate("", xytext=(xmin, 0), xycoords='data', xy=(xmax, 0), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))
plt.annotate("", xytext=(0, 0), xycoords='data', xy=(0, ymax), textcoords='data',
             arrowprops=dict(width=0.1, headwidth=6, headlength=8, facecolor='black', shrink=0.002))

c = -(M-1)
for i in range(M):
    # pdfs plot
    x = np.linspace(mus[i]-max_mean_dev, mus[i]+max_mean_dev, 200)
    plt.plot(x, mlab.normpdf(x, mus[i], sigma), 'k', linewidth=2)
    # xticks
    plt.plot(mus[i]*np.ones((2,)), [0, vtl], 'k', linewidth=0.8)
    # xlabels
    xlabel = r'${sign}{value}A/2$'.format(sign="-" if c < 0 else "",
                                          value="" if math.fabs(c) == 1 else int(math.fabs(c)))
    plt.text(mus[i], vtm, xlabel, fontsize=font_size, ha='center', va='baseline')
    pdf_label = r'$P_Y(y\mid H_{})$'.format(i)
    plt.text(mus[i], (pdf_max+ymax)/2, pdf_label, fontsize=font_size, ha='center', va='baseline')
    c += 2

c = -(M-2)/2
for i in range(M-1):
    # threshold lines
    plt.plot(thres[i]*np.ones((2,)), [0, pdf_thres], 'k', linewidth=1)
    # threshold xlabels
    if c != 0:
        xlabel = r'${sign}{value}A$'.format(sign="-" if c < 0 else "",
                                            value="" if math.fabs(c) == 1 else int(math.fabs(c)))
        plt.text(thres[i], vtm, xlabel, fontsize=font_size, ha='center', va='baseline')
    else:
        plt.text(thres[i], vtm, r'$0$', fontsize=font_size, ha='center', va='baseline')
    c += 1

# fill regions
nsamples = 50
col_step = 1 / (M-1)
for i in range(M):
    if i == 0:
        x = np.linspace(mus[i] + A/2, mus[i] + max_mean_dev, nsamples)
        plt.fill_between(x, 0, mlab.normpdf(x, mus[i], sigma), edgecolor="none", facecolor=(1-i*col_step, 0, 0),
                         label=r'$P_{{e_{}}}$'.format(i))
    elif i == M-1:
        x = np.linspace(mus[i] - max_mean_dev, mus[i] - A/2, nsamples)
        plt.fill_between(x, 0, mlab.normpdf(x, mus[i], sigma), edgecolor="none", facecolor=(1-i*col_step, 0, 0),
                         label=r'$P_{{e_{}}}$'.format(i))
    else:
        x = np.linspace(mus[i]-max_mean_dev, mus[i]-A/2, nsamples)
        plt.fill_between(x, 0, mlab.normpdf(x, mus[i], sigma), edgecolor="none", facecolor=(1-i*col_step, 0, 0),
                         label=r'$P_{{e_{}}}$'.format(i))
        x = np.linspace(mus[i] + A/2, mus[i] + max_mean_dev, nsamples)
        plt.fill_between(x, 0, mlab.normpdf(x, mus[i], sigma), edgecolor="none", facecolor=(1-i*col_step, 0, 0))

#plt.legend(loc='center left', frameon=False, prop={'size': font_size})
plt.legend(bbox_to_anchor=(0.11, 0.8), prop={'size': font_size}, frameon=False)

# xlegend
plt.text(xmax, vtm, r'$y$', fontsize=font_size, ha='right', va='baseline')

plt.axis('off')
plt.savefig('quaternary_gaussian_noise.eps', format='eps', bbox_inches='tight')
plt.show()
