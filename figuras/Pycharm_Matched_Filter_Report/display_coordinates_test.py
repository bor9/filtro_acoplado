import matplotlib.pyplot as plt
import numpy as np


def coord_equal_unity(transData, axis='x', length=1):
    if axis == 'x':
        a_coords = (0, length)
        b_coords = np.array([1, 0])
        a_col = 1
        b_col = 0
    else:
        a_coords = (length, 0)
        b_coords = np.array([0, 1])
        a_col = 0
        b_col = 1

    # transform data coordinates of a segment with a given length in axis a to display coordinates in axis a
    display_unity = transData.transform([(0, 0), a_coords])
    # get the length of the segment in display units
    a_display_unity = display_unity[1, a_col] - display_unity[0, a_col]
    # create a transform which will take from display to data coordinates
    inv = transData.inverted()
    # transform from display coordinates to data coordinates in axis b
    coords_unity = inv.transform([(0, 0), a_display_unity * b_coords])
    # get the length of the segment in data units
    b_coords_unity = coords_unity[1, b_col] - coords_unity[0, b_col]
    return b_coords_unity


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


# arbitrary values
xmin = -120
xmax = 400
ymin = -0.5
ymax = 2.5

plt.figure(0, figsize=(8, 7), frameon=False)
ax = plt.subplot(2, 1, 1)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

y_coords_unity = 1
display_unity = ax.transData.transform([(0, 0), (0, y_coords_unity)])
y_display_unity = display_unity[1, 1] - display_unity[0, 1]
print(display_unity)

inv = ax.transData.inverted()
coords_unity = inv.transform([(0,  0), (y_display_unity, 0)])
x_coords_unity = coords_unity[1, 0] - coords_unity[0, 0]

# plot a square
plt.plot([0, x_coords_unity], [0, 0], 'k')
plt.plot([0, x_coords_unity], [y_coords_unity, y_coords_unity], 'k')

plt.plot([0, 0], [0, y_coords_unity], 'k')
plt.plot([x_coords_unity, x_coords_unity], [0, y_coords_unity], 'k')

plt.show()

####################
# test function 1 y

plt.figure(1, figsize=(8, 7), frameon=False)


ax = plt.subplot(2, 1, 1)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

y_coords_unity = 3
x_coords_unity = coord_equal_unity(ax.transData, axis='x', length=y_coords_unity)

# plot a square
plt.plot([0, x_coords_unity], [0, 0], 'k')
plt.plot([0, x_coords_unity], [y_coords_unity, y_coords_unity], 'k')

plt.plot([0, 0], [0, y_coords_unity], 'k')
plt.plot([x_coords_unity, x_coords_unity], [0, y_coords_unity], 'k')

plt.show()

####################
# test function 1 x

plt.figure(2, figsize=(8, 7), frameon=False)


ax = plt.subplot(2, 1, 1)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

x_coords_unity = 10
y_coords_unity = coord_equal_unity(ax.transData, axis='y', length=x_coords_unity)

# plot a square
plt.plot([0, x_coords_unity], [0, 0], 'k')
plt.plot([0, x_coords_unity], [y_coords_unity, y_coords_unity], 'k')

plt.plot([0, 0], [0, y_coords_unity], 'k')
plt.plot([x_coords_unity, x_coords_unity], [0, y_coords_unity], 'k')

plt.show()

####################
# test function 2

plt.figure(2, figsize=(8, 7), frameon=False)


ax = plt.subplot(2, 1, 1)
plt.xlim(xmin, xmax)
plt.ylim(ymin, ymax)

display_length = 10
x_coords, y_coords = convert_display_to_data_coordinates(ax.transData, length=display_length)

# plot a square
plt.plot([0, x_coords], [0, 0], 'k')
plt.plot([0, x_coords], [y_coords, y_coords], 'k')

plt.plot([0, 0], [0, y_coords], 'k')
plt.plot([x_coords, x_coords], [0, y_coords], 'k')

plt.show()



