"""
Copyright (c) 2022 Alvin Shek
This work is licensed under the terms of the MIT license.
For a copy, see <https://opensource.org/licenses/MIT>.
"""
import numpy as np
import matplotlib.pyplot as plt
from typing import *


class Bubble(object):
    def __init__(self, pos, radius):
        self.radius = radius
        self.pos = pos


class Object(object):
    def __init__(self, pos, radius, ori):
        self.pos = pos
        self.radius = radius
        self.ori = ori  # theta in radians or quaternion


class ElasticBand(object):
    def __init__(self, start: np.ndarray, end: np.ndarray,
                 agent_radius: float, f_internal: float, f_alpha: float,
                 objects: List[Object], object_alphas: np.ndarray,
                 buffer=0.0):
        """
        Elastic Band trajectory generator.

        :param start: start position
        :param end: end position
        :param agent_radius: radius of agent
        :param f_internal: internal force coefficient
        :param f_alpha: external force coefficient
        :param objects: list of objects that exert forces on the agent
        :param object_alphas: determines direction and strength of object forces
        :param buffer: extra buffer added to object radii
        """
        self.start = start
        self.end = end
        self.agent_radius = agent_radius
        self.f_internal = f_internal
        self.objects = objects
        self.f_alpha = f_alpha
        self.object_poses = np.array([obj.pos for obj in self.objects])
        self.object_radii = np.array([obj.radius for obj in self.objects])
        self.object_alphas = object_alphas
        self.buffer = buffer

        self.bubbles = self.fill_gaps_helper(start, end)
        self.bubbles.append(Bubble(pos=self.end, radius=self.agent_radius))

    def calc_f_internal(self, i):
        """
        Given the index of a target bubble, calculate internal forces from its 
        neighboring bubbles.
        """
        target = self.bubbles[i]
        f = np.zeros_like(target.pos)
        if 0 <= i - 1:
            vec = (self.bubbles[i - 1].pos - target.pos)
            f += self.f_internal * vec  # force proportional to distance, so don't normalize

        if i + 1 < len(self.bubbles):
            vec = (self.bubbles[i + 1].pos - target.pos)
            f += self.f_internal * vec  # force proportional to distance

        return f

    def calc_f_external(self, bubble_idx):
        """
        Calculate external forces on a bubble.
        An object should only have influence if it is close enough
        to a bubble/node. This also depends on the object and the node radii
        since a larger object has a larger radius of influence. There is
        also some buffer so that even if a node does not overlap with
        an object, it still can be influenced if within the extra buffer zone.

        :param bubble_idx: index of bubble to calculate external forces for
        :return: sum of external forces on bubble: np.ndarray
        """
        target = self.bubbles[bubble_idx]
        f = np.zeros_like(target.pos)
        # vector from target to object, attraction(object_alpha > 0) or repulsion(object_alpha < 0)
        vecs = self.object_poses - target.pos
        is_attract = self.object_alphas > 0
        is_repel = self.object_alphas < 0

        center_dists = np.linalg.norm(vecs, axis=-1)
        # active attract: within radius of influence of object and object is an attractor
        active_attract = (center_dists < target.radius + self.object_radii + self.buffer) * is_attract
        active_repel = (center_dists < target.radius + self.object_radii + self.buffer) * is_repel
        active = np.where(np.bitwise_or(active_attract, active_repel))[0]

        for obj_i in active:
            # external force inverse proportional to distance
            # as distance increases, force decreases
            f += self.object_alphas[obj_i] * vecs[obj_i] / center_dists[obj_i]

        return f

    def update(self):
        """
        Update the positions of the bubbles, calculating forces on each bubble.
        Updates happen in-place, so current bubble's update affects subsequent bubbles.
        :return:
        """
        # Do not update the first(start) and last(goal) bubbles since those are
        # constraints for the trajectory
        for bi in range(1, len(self.bubbles) - 1):
            f_ext = self.calc_f_external(bi)
            f_int = self.calc_f_internal(bi)
            self.bubbles[bi].pos += self.f_alpha * (f_ext + f_int)

        self.fill_gaps_remove_overlaps()

    def fill_gaps_helper(self, p1, p2):
        """
        Helper to fill gaps between two bubbles.
        :param p1: bubble 1 position
        :param p2: bubble 2 position
        :return: List of bubbles between p1 and p2, including p1 itself
        """
        vec = p2 - p1
        dist = np.linalg.norm(vec)
        # number of bubbles needed to fill gap
        num_bubbles = int(np.ceil(dist / (2 * self.agent_radius)))

        # linearly interpolate between p1 and p2 adding the necessary bubbles
        fill = []
        for i in range(num_bubbles):
            pos = p1 + (i / num_bubbles) * vec
            fill.append(Bubble(pos=pos, radius=self.agent_radius))

        return fill

    def fill_gaps_remove_overlaps(self):
        """
        Any two bubbles that are not touching need to be filled with bubble(s).
        For three bubbles i-1, i, i+1 where i-1 and i+1 are touching, the middle
        bubble i needs to be removed.
        :return:
        """
        # add start bubble
        bi = 1
        new_bubbles = [self.bubbles[0]]

        while bi < len(self.bubbles) - 1:
            left = new_bubbles[-1]
            center = self.bubbles[bi]
            right = self.bubbles[bi + 1]
            if self.is_intersect(left, right):
                # don't add center bubble
                pass
            else:
                # try to fill gaps between (left, center) and (center, right)
                if self.is_intersect(left, center):
                    new_bubbles.append(center)
                else:
                    fill = self.fill_gaps_helper(left.pos, center.pos)
                    if len(fill) == 1:
                        new_bubbles += fill
                    else:
                        new_bubbles += fill[1:]
                    new_bubbles.append(center)

                if not self.is_intersect(center, right):
                    fill = self.fill_gaps_helper(center.pos, right.pos)
                    if len(fill) == 1:
                        new_bubbles += fill
                    else:
                        new_bubbles += fill[1:]
            bi += 1

        # add goal bubble
        new_bubbles.append(self.bubbles[-1])
        self.bubbles = new_bubbles

    def is_intersect(self, b1, b2):
        return np.linalg.norm(b1.pos - b2.pos) < (b1.radius + b2.radius)

    def visualize(self, title="", traj=None):
        plt.clf()
        for oi, obj in enumerate(self.objects):
            if self.object_alphas[oi] < 0:
                color = 'r'
                label = "Repel"
            else:
                color = 'g'
                label = "Attract"
            circle2 = plt.Circle(obj.pos, obj.radius + self.buffer, color=color, alpha=0.3)
            plt.gca().add_patch(circle2)

            circle1 = plt.Circle(obj.pos, obj.radius, color=color, label=label)
            plt.gca().add_patch(circle1)

        for bi, bubble in enumerate(self.bubbles):
            if bi == 0:
                color = 'c'
                label = "Start"
            elif bi == len(self.bubbles) - 1:
                label = "Goal"
                color = 'y'
            else:
                color = 'b'
                if bi == 1:
                    label = "Waypoints"
                else:
                    label = None
            circle1 = plt.Circle((bubble.pos[0], bubble.pos[1]),
                                 bubble.radius, color=color, alpha=0.5, label=label)
            plt.gca().add_patch(circle1)

        if traj is not None:
            plt.plot(traj[:, 0], traj[:, 1], '-', color="r", linewidth=2, label="GP Interpolation")

        plt.xlabel("X", fontsize=15)
        plt.ylabel("Y", fontsize=15)
        plt.xlim(0, 5)
        plt.ylim(0, 5)
        plt.xticks(fontsize=15)
        plt.yticks(fontsize=15)
        plt.title(title, fontsize=20)
        plt.gca().set_aspect('equal')
        plt.legend(bbox_to_anchor=(1.05, 1.0), loc='upper left', fontsize=15)
        plt.show()
