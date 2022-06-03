import numpy as np
from functools import reduce

from scipy.spatial.transform import Rotation as R


class KinovaRobot(object):
    def __init__(self):
        # define joint transformations in zero configuration wrt base frame
        self.g0 = np.zeros((4, 4))
        self.g0[:3, :3] = np.eye(3)
        self.g0[:3, 3] = np.array([0.61, 0.72, 2.376])
        self.g0[-1, -1] = 1

        # define axis of rotations
        self.w1 = np.array([0, 0, 1])[:, np.newaxis]
        self.w2 = np.array([-1, 0, 0])[:, np.newaxis]
        self.w3 = np.array([0, 0, 1])[:, np.newaxis]
        self.w4 = np.array([-1, 0, 0])[:, np.newaxis]
        self.w5 = np.array([0, 0, 1])[:, np.newaxis]
        self.w6 = np.array([-1, 0, 0])[:, np.newaxis]
        self.w7 = np.array([0, 0, 1])[:, np.newaxis]
        self.ws = [self.w1, self.w2, self.w3,
                   self.w4, self.w5, self.w6, self.w7]

        # define joint origins
        # NOTE: the joint origins are defined in the base frame
        # NOTE: q1 and q3 z offset could be defined as 0 since q can be placed
        #   anywhere along the axis of rotation (in their case z)
        #   same applies for joint 5 and 7
        self.q1 = self.q2 = self.q3 = np.array(
            [0.61, 0.72, 1.346])[:, np.newaxis]
        self.q4 = np.array([0.61, 0.72 + 0.045, 1.346 + 0.55])[:, np.newaxis]
        self.q5 = self.q6 = self.q7 = np.array(
            [0.61, 0.72, 1.346 + 0.55 + 0.3])[:, np.newaxis]
        self.qs = [self.q1, self.q2, self.q3,
                   self.q4, self.q5, self.q6, self.q7]

    @staticmethod
    def as_symm_matrix(a):
        a = a.flatten()
        assert a.shape[0] == 3

        return np.array([
            [0, -a[2], a[1]],
            [a[2], 0, -a[0]],
            [-a[1], a[0], 0]
        ])

    @staticmethod
    def twist_transform(theta, w, q):
        """
        Computes the twist transformation for a given joint angle and joint axis.

        :param theta: The joint angle in radians.
        :param w: The joint axis.
        :param q: The joint origin.
        :return: The twist transformation.
        """
        w_hat = Robot.as_symm_matrix(w)
        v = -w_hat @ q  # -w x q
        rot = exp_w_theta = (np.eye(3) + np.sin(theta) * w_hat +
                             (1 - np.cos(theta)) * np.dot(w_hat, w_hat))
        trans = (np.eye(3) - exp_w_theta) @ (w_hat @ v) + w @ w.T @ v * theta

        g = np.zeros((4, 4))
        g[:3, :3] = rot
        g[:3, 3] = trans.flatten()
        g[-1, -1] = 1
        return g

    @staticmethod
    def adjoint_transform(g):
        """
        Computes the adjoint transformation for a given jraoint axis and joint origin.

        :param g: Forward transforms
        """
        rot = g[:3, :3]
        trans = g[:3, 3]
        adj_g = np.zeros((6, 6))
        adj_g[0:3, 0:3] = rot
        adj_g[0:3, 3:] = Robot.as_symm_matrix(trans) @ rot
        adj_g[3:, 3:] = rot
        return adj_g

    def fk(self, thetas):
        """
        Forward kinematics for the robot arm.

        g_i = [e^{\psi_i * theta_i}, -w x q]
              [0                 , 0]
        e^{\psi_i * theta_i} = [1 + sin(theta)w_hat + (1-cos(theta)w_hat^2]


        :param thetas: joint angles in radians.
        :return: A tuple of the form (x, y, z) representing the position of the end effector.
        """

        joint_transforms = [self.twist_transform(
            thetas[i], self.ws[i], self.qs[i]) for i in range(7)]
        return reduce((lambda x, y: x @ y), joint_transforms) @ self.g0

    def calc_jacobian(self, thetas):
        joint_transforms = [self.twist_transform(
            thetas[i], self.ws[i], self.qs[i]) for i in range(7)]
        w_hats = [Robot.as_symm_matrix(self.ws[i]) for i in range(7)]
        vs = [-w_hats[i] @ self.qs[i] for i in range(7)]

        # 1st joint's g transform is the identity
        # 2nd joint's g transform is the first joint's g transform
        # 3rd joint's g transform is the first joint's g transform * 2nd joint's g transform
        jacobian = np.zeros((6, 7))
        g = np.eye(4)
        for i in range(7):
            psi_vec = np.concatenate([vs[i], self.ws[i]])
            psi_vec = self.adjoint_transform(g) @ psi_vec
            g = g @ joint_transforms[i]

            jacobian[:, i] = psi_vec.flatten()

        return jacobian

    @staticmethod
    def calc_quat_error(q1, q2):
        q1_0 = q1[-1]
        quat2_0 = q2[-1]
        q1_vec = q1[:-1]
        quat2_vec = q2[:-1]

        return q1_0 * quat2_vec - quat2_0 * q1_vec - np.cross(quat2_vec, q1_vec)

    def jacobian_transpose_method(self, jacobian, EE_error):
        return jacobian.T @ EE_error

    def jacobian_pseudoinv_method(self, jacobian, EE_error):
        return jacobian.T @ np.linalg.inv(jacobian @ jacobian.T) @ EE_error

    def damped_least_squares_method(self, jacobian, EE_error, damping_factor=0.01):
        return jacobian.T @ np.linalg.inv(jacobian @ jacobian.T + damping_factor * np.eye(jacobian.shape[0])) @ EE_error

    def calc_EE_error(self, xs, xd):
        pos_error = xd[:3] - xs[:3]

        quat1 = xs[3:7]
        quat2 = xd[3:7]
        quat1 = quat1 / np.linalg.norm(quat1)
        quat2 = quat2 / np.linalg.norm(quat2)
        quat_error = self.calc_quat_error(quat1, quat2)

        EE_error = np.concatenate([pos_error, quat_error])
        return EE_error

    def iterative_IK(self, thetas, xs, xd, alpha=0.1, max_iter=100, eps=1e-6, method='damped_LS'):
        if method == 'damped_LS':
            ik_method = self.damped_least_squares_method
        elif method == 'jacobian_transpose':
            ik_method = self.jacobian_transpose_method
        elif method == 'jacobian_pseudoinv':
            ik_method = self.jacobian_pseudoinv_method
        else:
            raise ValueError('Invalid method')

        thetas = np.array(thetas)
        theta_traj = []
        error_traj = []
        it = 0
        error = eps
        while error >= eps and it < max_iter:
            it += 1

            # calculate change in jacobian, error, and delta joint angles
            EE_error = self.calc_EE_error(xs, xd)
            jacobian = self.calc_jacobian(thetas)
            delta_joints = ik_method(jacobian, EE_error)

            # store current thetajacobian_pseudoinvs and error
            theta_traj.append(thetas.copy())
            error_traj.append(np.linalg.norm(EE_error))

            # update joint angles
            thetas = thetas + alpha * delta_joints

            # update current state
            EE_pose_mat = self.fk(thetas)
            EE_pos = EE_pose_mat[:3, 3]
            EE_quat = R.from_matrix(EE_pose_mat[:3, :3]).as_quat()
            xs = np.concatenate([EE_pos, EE_quat])

        theta_traj = [thetas % (2 * np.pi) for thetas in theta_traj]
        return theta_traj, np.array(error_traj)
