#!/home/ruic/Documents/meta_cobot_wksp/src/meta_cobot_learning/hri_tasks/venv/bin/python

import cv2
from cv_bridge import CvBridge
from sensor_msgs.msg import Image
import rospy

import os
import numpy as np

# NOTE: Run using the top python executable!!!!

class DemoVideoLogger(object):
    def __init__(self, out_path, fps=33, dsize=None):
        self.out_path = out_path
        self.fps = fps
        self.dsize = dsize  # width x height
        self.frames = []
        self.timesteps = []
        self.bridge = CvBridge()
        # self.sub = rospy.Subscriber("/kinect/hd/image_color", Image, self.callback, queue_size=10)
        self.sub = rospy.Subscriber("/cam_rs2/color/image_raw", Image, self.callback, queue_size=10)
        self.out_path = out_path
        self.it = 0

    def callback(self, img_msg):
        self.it += 1
        if self.it % 5 == 0: 
            print("received frame!")
        img = self.bridge.imgmsg_to_cv2(img_msg, desired_encoding='passthrough')
        if self.dsize is not None:
            img = cv2.resize(img, dsize=self.dsize)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        delta_time = rospy.Time.now().secs + rospy.Time.now().nsecs / float(1e9)
        self.frames.append(img)
        self.timesteps.append(delta_time)

    def save(self):
        video_writer = cv2.VideoWriter(self.out_path + ".mp4",
                                       cv2.VideoWriter_fourcc(*'FMP4'), self.fps, self.dsize)
        for fi in range(len(self.frames)):
            video_writer.write(self.frames[fi])
        video_writer.release()

        # np.save(self.out_path + ".npy", self.frames)
        print("Saved {} frames to {}".format(len(self.frames), self.out_path))


if __name__ == "__main__":

    rospy.init_node("demo_video_logger")

    frame_height = 540
    frame_width = 960
    exp_name = "trial7"
    out_root = "/home/ruic/Documents/opa/hardware_demo_videos"
    out_path = os.path.join(out_root, exp_name)
    video_path = os.path.join(out_path, "video")
    if not os.path.exists(out_path):
        os.mkdir(out_path)
    elif os.path.exists(video_path):
        rospy.logerr("Video already exists!")
        exit(-1)

    print("Starting video collection!")
    video_logger = DemoVideoLogger(dsize=(frame_width, frame_height), out_path=video_path)

    rospy.spin()

    video_logger.save()
