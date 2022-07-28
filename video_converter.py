import numpy as np
import cv2
import os
from tqdm import tqdm

frames_folder = "hardware_demo_videos/trial1"

import ipdb
ipdb.set_trace()
data = np.load(os.path.join(frames_folder, "video2.npz"))
frames = data["frames"]
timesteps = data["timesteps"]
video_writer = cv2.VideoWriter(os.path.join(frames_folder, "video.mp4"),
                               cv2.VideoWriter_fourcc(*'FMP4'), fps=20, dsize=(960, 540))
for fi in tqdm(range(len(frames))):
    video_writer.write(frames[fi])

video_writer.release()