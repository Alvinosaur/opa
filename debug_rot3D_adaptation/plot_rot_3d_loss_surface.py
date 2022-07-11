import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from matplotlib import cm
from scipy.spatial.transform import Rotation as R

from sklearn.cluster import KMeans


def plot_2d_contour(x, y, Z, title, vmin=0.1, vmax=10, vlevel=0.5, show=False,
                    xlabel='', ylabel='', zlabel='', alpha=1.0):
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
                           linewidth=0, antialiased=False,
                           rstride=1, cstride=1, edgecolor='none', alpha=alpha)

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


def plot_against_clusters(rot_feats, other_feats, losses, feat_is,
                          xlabel, ylabel, title, show=False):
    n_clusters = 10
    kmeans = KMeans(n_clusters=n_clusters, n_init=10, max_iter=300).fit(
        other_feats.reshape(-1, 1))
    clusters = kmeans.cluster_centers_.flatten()
    sorted_idxs = np.argsort(clusters)
    sorted_clusters = clusters[sorted_idxs]
    labels = kmeans.labels_
    # average loss over similar rot_diffs for each rot feat
    Z = np.zeros((len(rot_feats), n_clusters))
    counts = np.zeros((n_rot_feats, n_clusters))
    for i in range(len(feat_is)):
        feat_i = feat_is[i]
        label = labels[i]
        loss = losses[i]
        Z[feat_i, label] += loss
        counts[feat_i, label] += 1

    assert np.all(counts > 0)
    Z = Z / counts

    # plot loss surface over rot_feat vs rot_diff
    plot_2d_contour(sorted_clusters, rot_feats, Z[:, sorted_idxs], title, vmin=0.1,
                    vmax=10, vlevel=0.5, show=show,
                    xlabel=xlabel, ylabel=ylabel)


rot_feat_traj = np.array(rot_feat_traj)
rot_offset_xyz_traj = np.array(rot_offset_xyz_traj)
import ipdb
ipdb.set_trace()

# Get loss surface data
# [rot_feat, rot_offset_xyz, true_rot_offset_xyz, rot_diff, rot_loss]
results = np.load("rot_3D_loss_results.npy", allow_pickle=True)
n_rot_feats = len(results)
rot_feats = []
rot_diffs = []
rot_losses = []
pred_offsets_xyz = []
true_offsets_xyz = []
feat_is = []
for feat_i, row in enumerate(results):
    rot_feats.append(row[0][0])
    for (rot_feat, rot_offset_xyz, true_rot_offset_xyz, rot_diff, rot_loss) in row:
        rot_diffs.append(rot_diff)
        rot_losses.append(rot_loss)
        feat_is.append(feat_i)
        pred_offsets_xyz.append(rot_offset_xyz)
        true_offsets_xyz.append(true_rot_offset_xyz)

rot_feats = np.array(rot_feats)
rot_diffs = np.array(rot_diffs)
rot_losses = np.array(rot_losses)
feat_is = np.array(feat_is)
pred_offsets_xyz = np.array(pred_offsets_xyz)
true_offsets_xyz = np.array(true_offsets_xyz)
true_offset_quat = R.from_euler("xyz", true_offsets_xyz[0]).as_quat()

# Get data on a specific optimization trajectory
opt_traj = np.load("debug_rot_adaptation_results.npy", allow_pickle=True)
rot_feat_traj = []
rot_offset_xyz_traj = []
rot_error_traj = []
offset_diff_traj = []
loss_traj = []
for rot_feat, rot_offset_quat, loss in opt_traj:
    rot_feat_traj.append(rot_feat)
    rot_offset_quat = rot_offset_quat / np.linalg.norm(rot_offset_quat)
    rot_offset_xyz = R.from_quat(rot_offset_quat).as_euler('xyz')
    rot_offset_xyz_traj.append(rot_offset_xyz)
    rot_error_traj.append(
        np.arccos(np.abs(true_offset_quat @ rot_offset_quat)))
    offset_diff_traj.append(
        (true_offsets_xyz[0] - rot_offset_xyz) % (2 * np.pi))
    loss_traj.append(loss)


plot_against_clusters(rot_feats, rot_diffs, rot_losses, feat_is,
                      xlabel="Error in rot offset", ylabel="Rot Feats", title="rot_loss_vs_rot_error", show=False)
T = len(loss_traj)
ax = plt.gca()
for t in range(T):
    ax.plot(rot_error_traj, rot_feat_traj, loss_traj, 'o', alpha=0.9,
            color=cm.jet(t / T))
plt.show()

# plot loss surface over rot_feat vs rot_offset_x - true_rot_offset_x
offset_diffs = true_offsets_xyz - pred_offsets_xyz
offset_diffs = offset_diffs % (2 * np.pi)  # handle wrap around

x_rots = pred_offsets_xyz[:, 0]
y_rots = pred_offsets_xyz[:, 1]
z_rots = pred_offsets_xyz[:, 2]

plot_against_clusters(rot_feats, x_rots, rot_losses, feat_is,
                      xlabel="X Rot", ylabel="Rot Feats", title="rot_loss_vs_X", show=False)
plot_against_clusters(rot_feats, y_rots, rot_losses, feat_is,
                      xlabel="Y Rot", ylabel="Rot Feats", title="rot_loss_vs_Y", show=False)
plot_against_clusters(rot_feats, z_rots, rot_losses, feat_is,
                      xlabel="Z Rot", ylabel="Rot Feats", title="rot_loss_vs_Z", show=False)


x_diffs = offset_diffs[:, 0]
y_diffs = offset_diffs[:, 1]
z_diffs = offset_diffs[:, 2]

plot_against_clusters(rot_feats, rot_diffs, rot_losses, feat_is,
                      xlabel="X Diff", ylabel="Rot Feats", title="rot_loss_vs_X_diff", show=False)
plot_against_clusters(rot_feats, y_diffs, rot_losses, feat_is,
                      xlabel="Y Diff", ylabel="Rot Feats", title="rot_loss_vs_Y_diff", show=False)
plot_against_clusters(rot_feats, z_diffs, rot_losses, feat_is,
                      xlabel="Z Diff", ylabel="Rot Feats", title="rot_loss_vs_Z_diff", show=False)
