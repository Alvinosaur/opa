#!/home/ruic/Documents/meta_cobot_wksp/src/meta_cobot_learning/venv3/bin/python
import numpy as np
import time

import rospy
# from hri_tasks_msgs.msg import PoseSync
from geometry_msgs.msg import PoseStamped

from scipy.spatial.transform import Rotation as R
from scipy.spatial.transform import Slerp

from viz_3D import pose_to_msg

robot_pose = None


def robot_pose_cb(msg):
    global robot_pose
    robot_pose = msg.pose


if __name__ == '__main__':
    rospy.init_node('test_kinova_interface')
    rospy.Subscriber('/kinova/pose_tool_in_base_fk',
                     PoseStamped, robot_pose_cb, queue_size=1)
    pose_pub = rospy.Publisher("/kinova_demo/pose_cmd", PoseStamped, queue_size=10)

    """
    Actual initial pose:
        float init_x_ = 0.406
        float init_y_ = -0.158
        float init_z_ = 0.348
    """

    target_pos = np.array([0.406, -0.158, 0.348]) + np.array([0.1, 0.2, 0.15])
    target_ori = target_pose = None

    start_t = time.time()
    while not rospy.is_shutdown():
        if robot_pose is not None:
            # print(robot_pose)
            # print(robot_pose.position)
            if target_ori is None:
                target_ori = np.array([robot_pose.orientation.x,
                                        robot_pose.orientation.y,
                                        robot_pose.orientation.z,
                                        robot_pose.orientation.w])
                target_pose = np.concatenate(
                    [target_pos, target_ori])

            if time.time() - start_t > 1.0:
                pose_pub.publish(pose_to_msg(target_pose))

        else:
            print('No robot pose received...')

        rospy.sleep(0.1)
