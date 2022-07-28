import numpy as np
from pynput import keyboard
from scipy.spatial.transform import Rotation as R

import rospy
from geometry_msgs.msg import PoseStamped
from std_msgs.msg import Bool, Float64MultiArray

# Global info updated by callbacks
cur_joints = None
cur_pos_world, cur_ori_quat = None, None
target_pos_world, target_ori_quat = None, None
is_perturb_pos = True
ROBOT_FRAME = "kinova_base"

def pose_to_msg(pose: np.ndarray, frame) -> PoseStamped:
    """
    Convert a pose to a geometry_msgs.msg.PoseStamped() message

    :param pose: Pose (pos_dim + rot_dim)
    :return: geometry_msgs.msg.PoseStamped() message
    """
    msg = PoseStamped()
    msg.header.frame_id = frame
    msg.pose.position.x = pose[0]
    msg.pose.position.y = pose[1]
    msg.pose.position.z = pose[2]

    msg.pose.orientation.x = pose[3]
    msg.pose.orientation.y = pose[4]
    msg.pose.orientation.z = pose[5]
    msg.pose.orientation.w = pose[6]

    return msg

def msg_to_pose(msg):
    pose = np.array([
        msg.pose.position.x,
        msg.pose.position.y,
        msg.pose.position.z,
        msg.pose.orientation.x,
        msg.pose.orientation.y,
        msg.pose.orientation.z,
        msg.pose.orientation.w,
    ])
    return pose

def normalize_pi_neg_pi(ang):
    while ang > np.pi:
        ang -= 2 * np.pi
    while ang <= -np.pi:
        ang += 2 * np.pi
    return ang


def on_press(key):
    """
    Callback for keyboard events. Modifies global variables directly.

    :param key: pressed key
    :return:
    """
    global cur_pos_world, cur_ori_quat
    global target_pos_world, target_ori_quat
    global is_perturb_pos
    ROT_SPEED = np.deg2rad(10)
    POS_SPEED = 0.075
    if key == keyboard.Key.shift:
        is_perturb_pos = not is_perturb_pos

    if is_perturb_pos:
        try:
            if key.char == "w":
                target_pos_world[1] -= POS_SPEED
            elif key.char == "s":
                target_pos_world[1] += POS_SPEED

            # flipped for y-axis due to rotated camera view
            elif key.char == "a":
                target_pos_world[0] += POS_SPEED
            elif key.char == "d":
                target_pos_world[0] -= POS_SPEED

        except Exception as e:
            if key == keyboard.Key.up:
                target_pos_world[2] += POS_SPEED
            elif key == keyboard.Key.down:
                target_pos_world[2] -= POS_SPEED

    else:
        # Apply rotation perturbations wrt world frame -> left multiply the offset
        try:
            if key.char == "w":
                target_ori_quat = (
                        R.from_euler("z", ROT_SPEED, degrees=False) * R.from_quat(target_ori_quat)).as_quat()
            elif key.char == "s":
                target_ori_quat = (
                        R.from_euler("z", -ROT_SPEED, degrees=False) * R.from_quat(target_ori_quat)).as_quat()

            # flipped for y-axis due to rotated camera view
            elif key.char == "d":
                target_ori_quat = (
                        R.from_euler("x", ROT_SPEED, degrees=False) * R.from_quat(target_ori_quat)).as_quat()
            elif key.char == "a":
                target_ori_quat = (
                        R.from_euler("x", -ROT_SPEED, degrees=False) * R.from_quat(target_ori_quat)).as_quat()

        except:
            if key == keyboard.Key.up:
                target_ori_quat = (
                        R.from_euler("y", -ROT_SPEED, degrees=False) * R.from_quat(target_ori_quat)).as_quat()
            elif key == keyboard.Key.down:
                target_ori_quat = (
                        R.from_euler("y", ROT_SPEED, degrees=False) * R.from_quat(target_ori_quat)).as_quat()

def robot_pose_cb(msg):
    global cur_pos_world, cur_ori_quat
    pose = msg_to_pose(msg)
    cur_pos_world = pose[0:3]
    cur_ori_quat = pose[3:]


def robot_joints_cb(msg):
    global cur_joints
    cur_joints = np.deg2rad(msg.data)
    for i in range(len(cur_joints)):
        cur_joints[i] = normalize_pi_neg_pi(cur_joints[i])


if __name__ == "__main__":
    rospy.init_node('cartesian_keyboard')

    # Robot EE pose
    rospy.Subscriber('/kinova/pose_tool_in_base_fk',
                     PoseStamped, robot_pose_cb, queue_size=1)

    rospy.Subscriber('/kinova/current_joint_state',
                     Float64MultiArray, robot_joints_cb, queue_size=1)

    # Target pose topic
    pose_pub = rospy.Publisher(
        "/kinova_demo/pose_cmd", PoseStamped, queue_size=10)
    is_intervene_pub = rospy.Publisher("/is_intervene", Bool, queue_size=10)
    gripper_pub = rospy.Publisher("/siemens_demo/gripper_cmd", Bool, queue_size=10)

    # Listen for keypresses marking start/stop of human intervention
    listener = keyboard.Listener(
        on_press=on_press)
    listener.start()

    while cur_pos_world is None and cur_joints is None:
        rospy.sleep(0.1)

    target_pos_world = np.copy(cur_pos_world)
    target_ori_quat = np.copy(cur_ori_quat)
    is_intervene = False
    it = 0
    while not rospy.is_shutdown():
    
        # command_kinova_gripper(gripper_pub, cmd_open=True)

        is_intervene_pub.publish(Bool(is_intervene))

        target_pose = np.concatenate(
            [target_pos_world, target_ori_quat])
        pose_pub.publish(pose_to_msg(target_pose, frame=ROBOT_FRAME))
        
        if it % 3 == 0:
            print(f"pos: {np.array2string(cur_pos_world, precision=3, separator=', ')}, ori: {np.array2string(cur_ori_quat, precision=3, separator=', ')}", )
            print(f"joints: {np.array2string(cur_joints, precision=3, separator=', ')}")
        rospy.sleep(0.3)
        it += 1


