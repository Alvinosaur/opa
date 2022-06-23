# Adapted from the "Loss Landscape" paper
# https://github.com/tomgoldstein/loss-landscape

from mpl_toolkits.mplot3d import Axes3D
from matplotlib import pyplot as plt
from matplotlib import cm
import h5py
import numpy as np
from os.path import exists
import seaborn as sns


def plot_2d_contour(x, y, Z, title, vmin=0.1, vmax=10, vlevel=0.5, show=False,
                    xlabel='', ylabel='', zlabel=''):
    """
        x: x range coordinates
        y: y range coordinates
        Z: 2D array of values
    """
    """Plot 2D contour map and 3D surface."""
    X, Y = np.meshgrid(x, y)

    print('------------------------------------------------------------------')
    print('plot_2d_contour')
    print('------------------------------------------------------------------')
    print('len(xcoordinates): %d   len(ycoordinates): %d' % (len(x), len(y)))
    print('max = %f \t min = %f' %
          (np.max(Z), np.min(Z)))

    if (len(x) <= 1 or len(y) <= 1):
        print('The length of coordinates is not enough for plotting contours')
        return

    # # --------------------------------------------------------------------
    # # Plot 2D contours
    # # --------------------------------------------------------------------
    # fig = plt.figure()
    # CS = plt.contour(X, Y, Z, cmap='summer',
    #                  levels=np.arange(vmin, vmax, vlevel))
    # plt.xlabel(xlabel)
    # plt.ylabel(ylabel)
    # plt.clabel(CS, inline=1, fontsize=8)
    # fig.savefig(title + '_contour.jpg', dpi=300, bbox_inches='tight')

    # # --------------------------------------------------------------------
    # # Plot 2D heatmaps
    # # --------------------------------------------------------------------
    # fig = plt.figure()
    # sns_plot = sns.heatmap(Z, cmap='viridis', cbar=True, vmin=vmin, vmax=vmax,
    #                        xticklabels=False, yticklabels=False)
    # sns_plot.invert_yaxis()
    # sns_plot.get_figure().savefig(title + '_2dheat.jpg', dpi=300, bbox_inches='tight')

    # --------------------------------------------------------------------
    # Plot 3D surface
    # --------------------------------------------------------------------
    fig = plt.figure()
    ax = Axes3D(fig)
    surf = ax.plot_surface(X, Y, Z, cmap=cm.coolwarm,
                           linewidth=0, antialiased=False)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    ax.set_zlabel('loss')
    plt.title(title)
    fig.colorbar(surf, shrink=0.5, aspect=5)

    for i, ang in enumerate(range(0, 360, 45)):
        ax.view_init(30, ang)
        fig.savefig(title + f'_3dsurface{i}.jpg', dpi=300,
                    bbox_inches='tight')

    if show:
        plt.show()
