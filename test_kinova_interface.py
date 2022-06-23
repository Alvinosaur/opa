import numpy as np
import time

import rospy
from hri_tasks_msgs.msg import PoseSync
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
    rospy.Subscriber('/robot_cartesian_pose', PoseSync, robot_pose_cb)
    pose_pub = rospy.Publisher("/kinova_pose_ref", PoseStamped, queue_size=10)

    """
    Actual initial pose:
        float init_x_ = 0.406
        float init_y_ = -0.158
        float init_z_ = 0.348
    """

    target_pos = np.array([0.406, -0.158, 0.348]) + np.array([0.05, 0.0, 0.0])
    target_ori = target_pose = None

    start_t = time.time()
    while not rospy.is_shutdown():
        if robot_pose is not None:
            print(robot_pose)
            print(robot_pose.pose.position)
            if target_ori is None:
                target_ori = R.from_euler('zyx',
                                          [robot_pose.pose.orientation.z,
                                           robot_pose.pose.orientation.y,
                                           robot_pose.pose.orientation.x])
                target_pose = np.concatenate(
                    [target_pos, target_ori.as_quat()])

            # if time.time() - start_t > 3.0:
            #     pose_pub.publish(pose_to_msg(target_pos))

        else:
            print('No robot pose received...')

        rospy.sleep(0.1)
