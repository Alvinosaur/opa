import matplotlib.pyplot as plt
import numpy as np
import os
from sklearn.decomposition import PCA
from tqdm import tqdm

from data_params import Params
from viz_2D import draw
from elastic_band import Object

from data_generation import generate_traj_helper

# Adapted from: https://github.com/Pmlsa/Discrete-Fourier-Transform/blob/master/main.py
# Modified to handle variable timesteps based on the discussion: https://math.stackexchange.com/questions/27452/fft-of-waveform-with-non-constant-timestep


def magnitude(real: float, imaginary: complex):
    """
    Magnitude of a Vector
    """
    return np.sqrt(real ** 2 + imaginary.real ** 2)


def DFT(x, y):
    # normalize x/time to [0, 1]
    # x: [1.5, 2, 3.5] ->
    # y: [1, 2, 3]

    x = x / x.max()
    N = len(x)
    assert np.isclose(x[0], 0.0)
    assert np.isclose(x[-1], 1.0)

    freqBins = []
    for k in range(N):
        Σ = 0

        for n in range(N - 1):
            # midpoint formula
            tn = (x[n + 1] + x[n]) / 2
            yn = (y[n + 1] + y[n]) / 2
            tmp = (0 - 1j) * (2 * np.pi * k * tn)
            Σ += (x[n + 1] - x[n]) * yn * np.exp(tmp)

        freqBins.append(2 * magnitude(Σ.real, Σ.imag))

    return freqBins


def fftPlot(sig, dt=None, ts=None, plot=True, title='Analytic FFT plot'):
    """
    Taken from https://stackoverflow.com/a/53925342
    """
    # Here it's assumes analytic signal (real signal...) - so only half of the axis is required

    if ts is None:
        if dt is None:
            dt = 1
            ts = np.arange(0, sig.shape[-1])
            xLabel = 'samples'
        else:
            ts = np.arange(0, sig.shape[-1]) * dt
            xLabel = 'freq [Hz]'
    else:
        # need to ensure samples are evenly spaced in time
        # basic idea is to fit Gaussian Process to data, then uniformly interpolate

        dt = 0.2
        new_sig = generate_traj_helper(
            waypoints=np.hstack([ts.reshape(-1, 1), sig.reshape(-1, 1)]), dstep=dt)
        import ipdb
        ipdb.set_trace()
        ts = np.arange(0, sig.shape[-1]) * dt

    if sig.shape[0] % 2 != 0:
        # warnings.warn("signal preferred to be even in size, autoFixing it...")
        ts = ts[0:-1]
        sig = sig[0:-1]

    # Divided by size t for coherent magnitude
    sigFFT = np.fft.fft(sig) / ts.shape[0]

    freq = np.fft.fftfreq(ts.shape[0], d=dt)

    # Plot analytic signal - right half of frequence axis needed only...
    firstNegInd = np.argmax(freq < 0)
    freqAxisPos = freq[0:firstNegInd]
    # *2 because of magnitude of analytic signal
    sigFFTPos = 2 * sigFFT[0:firstNegInd]

    if plot:
        plt.figure()
        plt.plot(freqAxisPos, np.abs(sigFFTPos))
        plt.xlabel(xLabel)
        plt.ylabel('mag')
        plt.title(title)
        plt.show()

    return np.abs(sigFFTPos), freqAxisPos


def test():
    dt = 1 / 100

    # Build a signal within Nyquist - the result will be the positive FFT with actual magnitude
    f0 = 200  # [Hz]
    ts = np.arange(0, 1 + dt, dt)
    sig = 1 * np.sin(2 * np.pi * f0 * ts) + \
        10 * np.sin(2 * np.pi * f0 / 2 * ts) + \
        3 * np.sin(2 * np.pi * f0 / 4 * ts) +\
        7.5 * np.sin(2 * np.pi * f0 / 5 * ts)

    # # Plot the signal
    # plt.figure()
    # plt.plot(t, sig)
    # plt.xlabel('time [s]')
    # plt.ylabel('sig')
    # plt.title('Signal')
    # plt.show()

    # freq_bins = DFT(x=t, y=sig)

    # nufft = NUFFT()
    # nufft.plan(t, len())
    # import ipdb
    # ipdb.set_trace()

    # Result in frequencies
    fftPlot(sig)
    # Result in samples (if the frequencies axis is unknown)
    fftPlot(sig)


if __name__ == '__main__':
    fig, ax = plt.subplots()
    # Frequency Analysis
    case = 0
    train_rot = False
    samples_files = [
        "sampled_file_idxs_pos.npy",
        "sampled_file_idxs_rot.npy",
        "sampled_file_idxs_rot_ignore.npy",
    ]
    datasets = ["pos_2D_test", "rot_2D_test", "rot_2D_test"]
    sampled_file_idxs = np.load(
        f"eval_adaptation_results/{samples_files[case]}")
    max_mag_per_sample = []
    auc_per_sample = []
    variance_per_sample = []
    for sample_i, file_idx in enumerate(sampled_file_idxs):
        data = np.load(os.path.join("data", datasets[case], "traj_%d.npz" %
                                    file_idx), allow_pickle=True)
        object_types = data["object_types"]
        expert_traj = data["states"]

        ######### Debug Plot #########
        # goal_radius = data["goal_radius"].item()
        # object_poses = data["object_poses"]
        # object_radii = data["object_radii"]
        # # NOTE: this is purely for viz, model should NOT know this!
        # theta_offsets = Params.ori_offsets_2D[object_types]
        # start = expert_traj[0]
        # goal = expert_traj[-1]
        # num_objects = len(object_types)
        # # NOTE: rather, object indices should be actual indices, not types
        # object_idxs = np.arange(num_objects)
        # objects = [
        #     Object(pos=object_poses[i][0:2], radius=object_radii[i],
        #            ori=object_poses[i][-1] + theta_offsets[i]) for i in range(len(object_types))
        # ]
        # draw(ax, start_pose=start, goal_pose=goal,
        #      goal_radius=goal_radius, agent_radius=Params.agent_radius,
        #      object_types=object_types, offset_objects=objects,
        #      pred_traj=expert_traj, expert_traj=expert_traj,
        #      show_rot=train_rot, hold=True)

        ######## FFT Analysis #########
        expert_traj_xy = expert_traj[:, :2]
        # Use PCA to extract principal axis, align trajectory to principal axis
        pca = PCA(n_components=1)
        pca.fit(expert_traj_xy)
        major_axis = pca.components_[0]
        midpoint = np.mean(expert_traj_xy, axis=0)
        ang = np.arctan2(major_axis[1], major_axis[0])
        R = np.array([[np.cos(ang), -np.sin(ang)],
                      [np.sin(ang), np.cos(ang)]])

        major_axis_start = midpoint + major_axis * \
            np.linalg.norm(expert_traj_xy[0] - midpoint)
        major_axis_end = midpoint - major_axis * \
            np.linalg.norm(expert_traj_xy[-1] - midpoint)

        # normalize to be zero-centered and undo rotation of major axis
        traj_aligned = np.dot(expert_traj_xy - midpoint, R)

        # import ipdb
        # ipdb.set_trace()
        # ang_traj = np.arctan2(traj_aligned[:, 1], traj_aligned[:, 0])
        # ang_traj[np.where(ang_traj < 0)] += 2 * np.pi
        # mag, freq = fftPlot(ang_traj,
        #                     title="Overall FFT", plot=False)
        mag, freq = fftPlot(traj_aligned[:, 1],
                            dt=1, title="Overall FFT", plot=False)
        max_mag_per_sample.append(np.max(mag))
        auc_per_sample.append(np.trapz(mag, dx=freq[1] - freq[0]))
        variance_per_sample.append(np.var(traj_aligned[:, 1]))

    # Save results per sample
    np.save(os.path.join(
        "eval_adaptation_results", f"max_mag_per_sample_{datasets[0]}"),
        max_mag_per_sample)
    np.save(os.path.join(
        "eval_adaptation_results", f"auc_per_sample_{datasets[0]}"),
        auc_per_sample)
    np.save(os.path.join(
        "eval_adaptation_results", f"variance_per_sample_{datasets[0]}"),
        variance_per_sample)

    max_mag_sample_idx = sampled_file_idxs[np.argmax(max_mag_per_sample)]
    max_auc_sample_idx = sampled_file_idxs[np.argmax(auc_per_sample)]
    max_variance_sample_idx = sampled_file_idxs[np.argmax(variance_per_sample)]

    ######### Max Mag sample #########
    print("Max Mag sample")
    data = np.load(os.path.join("data", datasets[case], "traj_%d.npz" %
                                max_mag_sample_idx), allow_pickle=True)
    object_types = data["object_types"]
    expert_traj = data["states"]
    goal_radius = data["goal_radius"].item()
    object_poses = data["object_poses"]
    object_radii = data["object_radii"]
    # NOTE: this is purely for viz, model should NOT know this!
    theta_offsets = Params.ori_offsets_2D[object_types]
    start = expert_traj[0]
    goal = expert_traj[-1]
    num_objects = len(object_types)
    # NOTE: rather, object indices should be actual indices, not types
    object_idxs = np.arange(num_objects)
    objects = [
        Object(pos=object_poses[i][0:2], radius=object_radii[i],
               ori=object_poses[i][-1] + theta_offsets[i]) for i in range(len(object_types))
    ]

    expert_traj_xy = expert_traj[:, :2]
    # Use PCA to extract principal axis, align trajectory to principal axis
    pca = PCA(n_components=1)
    pca.fit(expert_traj_xy)
    major_axis = pca.components_[0]
    midpoint = np.mean(expert_traj_xy, axis=0)
    ang = np.arctan2(major_axis[1], major_axis[0])
    R = np.array([[np.cos(ang), -np.sin(ang)],
                  [np.sin(ang), np.cos(ang)]])

    major_axis_start = midpoint + major_axis * \
        np.linalg.norm(expert_traj_xy[0] - midpoint)
    major_axis_end = midpoint - major_axis * \
        np.linalg.norm(expert_traj_xy[-1] - midpoint)
    draw(ax, start_pose=start, goal_pose=goal,
         goal_radius=goal_radius, agent_radius=Params.agent_radius,
         object_types=object_types, offset_objects=objects,
         pred_traj=np.vstack([major_axis_start[np.newaxis],
                              major_axis_end[np.newaxis]
                              ]), expert_traj=expert_traj,
         show_rot=train_rot, hold=True)

    # normalize to be zero-centered and undo rotation of major axis
    plt.clf()
    traj_aligned = np.dot(expert_traj_xy - midpoint, R)
    plt.plot(traj_aligned[:, 0], traj_aligned[:, 1])
    plt.show()

    mag, freq = fftPlot(traj_aligned[:, 1],
                        dt=1, title="Overall FFT")

    ######### Max AUC sample" #########
    fig, ax = plt.subplots()
    print("Max AUC sample")
    data = np.load(os.path.join("data", datasets[case], "traj_%d.npz" %
                                max_auc_sample_idx), allow_pickle=True)
    object_types = data["object_types"]
    expert_traj = data["states"]
    goal_radius = data["goal_radius"].item()
    object_poses = data["object_poses"]
    object_radii = data["object_radii"]
    # NOTE: this is purely for viz, model should NOT know this!
    theta_offsets = Params.ori_offsets_2D[object_types]
    start = expert_traj[0]
    goal = expert_traj[-1]
    num_objects = len(object_types)
    # NOTE: rather, object indices should be actual indices, not types
    object_idxs = np.arange(num_objects)
    objects = [
        Object(pos=object_poses[i][0:2], radius=object_radii[i],
               ori=object_poses[i][-1] + theta_offsets[i]) for i in range(len(object_types))
    ]
    expert_traj_xy = expert_traj[:, :2]
    # Use PCA to extract principal axis, align trajectory to principal axis
    pca = PCA(n_components=1)
    pca.fit(expert_traj_xy)
    major_axis = pca.components_[0]
    midpoint = np.mean(expert_traj_xy, axis=0)
    ang = np.arctan2(major_axis[1], major_axis[0])
    R = np.array([[np.cos(ang), -np.sin(ang)],
                  [np.sin(ang), np.cos(ang)]])

    major_axis_start = midpoint + major_axis * \
        np.linalg.norm(expert_traj_xy[0] - midpoint)
    major_axis_end = midpoint - major_axis * \
        np.linalg.norm(expert_traj_xy[-1] - midpoint)
    draw(ax, start_pose=start, goal_pose=goal,
         goal_radius=goal_radius, agent_radius=Params.agent_radius,
         object_types=object_types, offset_objects=objects,
         pred_traj=np.vstack([major_axis_start[np.newaxis],
                              major_axis_end[np.newaxis]
                              ]), expert_traj=expert_traj,
         show_rot=train_rot, hold=True)

    # normalize to be zero-centered and undo rotation of major axis
    plt.clf()
    traj_aligned = np.dot(expert_traj_xy - midpoint, R)
    plt.plot(traj_aligned[:, 0], traj_aligned[:, 1])
    plt.show()

    mag, freq = fftPlot(traj_aligned[:, 1],
                        dt=1, title="Overall FFT")

    ######### Max Var sample #########
    fig, ax = plt.subplots()
    print("Max Var sample")
    data = np.load(os.path.join("data", datasets[case], "traj_%d.npz" %
                                max_auc_sample_idx), allow_pickle=True)
    object_types = data["object_types"]
    expert_traj = data["states"]
    goal_radius = data["goal_radius"].item()
    object_poses = data["object_poses"]
    object_radii = data["object_radii"]
    # NOTE: this is purely for viz, model should NOT know this!
    theta_offsets = Params.ori_offsets_2D[object_types]
    start = expert_traj[0]
    goal = expert_traj[-1]
    num_objects = len(object_types)
    # NOTE: rather, object indices should be actual indices, not types
    object_idxs = np.arange(num_objects)
    objects = [
        Object(pos=object_poses[i][0:2], radius=object_radii[i],
               ori=object_poses[i][-1] + theta_offsets[i]) for i in range(len(object_types))
    ]
    expert_traj_xy = expert_traj[:, :2]
    # Use PCA to extract principal axis, align trajectory to principal axis
    pca = PCA(n_components=1)
    pca.fit(expert_traj_xy)
    major_axis = pca.components_[0]
    midpoint = np.mean(expert_traj_xy, axis=0)
    ang = np.arctan2(major_axis[1], major_axis[0])
    R = np.array([[np.cos(ang), -np.sin(ang)],
                  [np.sin(ang), np.cos(ang)]])

    major_axis_start = midpoint + major_axis * \
        np.linalg.norm(expert_traj_xy[0] - midpoint)
    major_axis_end = midpoint - major_axis * \
        np.linalg.norm(expert_traj_xy[-1] - midpoint)
    draw(ax, start_pose=start, goal_pose=goal,
         goal_radius=goal_radius, agent_radius=Params.agent_radius,
         object_types=object_types, offset_objects=objects,
         pred_traj=np.vstack([major_axis_start[np.newaxis],
                              major_axis_end[np.newaxis]
                              ]), expert_traj=expert_traj,
         show_rot=train_rot, hold=True)

    # normalize to be zero-centered and undo rotation of major axis
    plt.clf()
    traj_aligned = np.dot(expert_traj_xy - midpoint, R)
    plt.plot(traj_aligned[:, 0], traj_aligned[:, 1])
    plt.show()

    mag, freq = fftPlot(traj_aligned[:, 1],
                        dt=1, title="Overall FFT")
    max_mag_per_sample.append(np.max(mag))
    auc_per_sample.append(np.trapz(mag, dx=freq[1] - freq[0]))
