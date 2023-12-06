import random
from typing import Dict, Generator, List, Optional

import numpy as np
import rospy


from task_generator.constants import Constants
from task_generator.shared import PositionOrientation
from task_generator.tasks.task_factory import TaskFactory
from task_generator.tasks.base_task import BaseTask
from task_generator.tasks.utils import ITF_Random, RandomList

import geometry_msgs.msg as geometry_msgs

from tf.transformations import euler_from_quaternion


@TaskFactory.register(Constants.TaskMode.EXPLORE)
class ExploreTask(BaseTask):
    """
        The random task spawns static and dynamic
        obstacles on every reset and will create
        a new robot start and goal position for
        each task.
    """

    itf_random: ITF_Random

    _gen_static: Generator[int, None, None]
    _gen_interactive: Generator[int, None, None]
    _gen_dynamic: Generator[int, None, None]

    _static_obstacles: RandomList
    _interactive_obstacles: RandomList
    _dynamic_obstacles: RandomList

    _is_done: bool

    TOPIC_SET_POSITION = "/initialpose"
    TOPIC_SET_GOAL = "/goalpose"
    TOPIC_NEW_SCENARIO = "/clicked_point"
    PARAM_WAYPOINTS = "guided_waypoints"

    def __init__(self, **kwargs):
        BaseTask.__init__(self, **kwargs)

        self.itf_random = ITF_Random(self)

        obstacle_ranges = self.itf_random.load_obstacle_ranges()
        self._gen_static = ITF_Random.randrange_generator(
            obstacle_ranges.static)
        self._gen_interactive = ITF_Random.randrange_generator(
            obstacle_ranges.interactive)
        self._gen_dynamic = ITF_Random.randrange_generator(
            obstacle_ranges.dynamic)

        self._static_obstacles, \
            self._interactive_obstacles, \
            self._dynamic_obstacles = self.itf_random.load_obstacle_list()

        self.iters = 0
        self._is_done = False

        self._new_scenario()

        rospy.Subscriber(self.TOPIC_SET_POSITION,
                         geometry_msgs.PoseWithCovarianceStamped, self._set_position)
        rospy.Subscriber(self.TOPIC_SET_GOAL,
                         geometry_msgs.PoseStamped, self._set_goal)
        rospy.Subscriber(self.TOPIC_NEW_SCENARIO,
                         geometry_msgs.PointStamped, self._new_scenario)

    @BaseTask.reset_helper(parent=BaseTask)
    def reset(
            self,
            n_static_obstacles: Optional[int] = None,
            n_interactive_obstacles: Optional[int] = None,
            n_dynamic_obstacles: Optional[int] = None,
            static_obstacles: Optional[RandomList] = None,
            interactive_obstacles: Optional[RandomList] = None,
            dynamic_obstacles: Optional[RandomList] = None,
            **kwargs):

        if n_static_obstacles is None:
            n_static_obstacles = next(self._gen_static)

        if n_interactive_obstacles is None:
            n_interactive_obstacles = next(self._gen_interactive)

        if n_dynamic_obstacles is None:
            n_dynamic_obstacles = next(self._gen_dynamic)

        if static_obstacles is None:
            static_obstacles = self._static_obstacles

        if interactive_obstacles is None:
            interactive_obstacles = self._interactive_obstacles

        if dynamic_obstacles is None:
            dynamic_obstacles = self._dynamic_obstacles

        def callback():

            robot_positions = (
                PositionOrientation(position.x, position.y,
                                    random.random() * 2 * np.pi)
                for position in (
                    self.world_manager.get_positions_on_map(
                        n=len(self.robot_managers),
                        safe_dist=max(
                            robot.safe_distance for robot in self.robot_managers)
                    )
                )
            )

            self.obstacle_manager.respawn(callback=lambda: self.itf_random.setup_random(
                n_static_obstacles=n_static_obstacles,
                n_interactive_obstacles=n_interactive_obstacles,
                n_dynamic_obstacles=n_dynamic_obstacles,
                static_obstacles=static_obstacles,
                interactive_obstacles=interactive_obstacles,
                dynamic_obstacles=dynamic_obstacles,
                robot_positions=[(pos, pos) for pos in robot_positions]
            ))

            self.iters += 1

            return False

        return {}, callback

    @property
    def is_done(self) -> bool:

        biggest_robot = max(robot.safe_distance for robot in self.robot_managers)

        for robot in self.robot_managers:
            if robot.is_done:
                waypoint = self.world_manager.get_position_on_map(safe_dist=2*biggest_robot, forbid=False)
                robot.reset(start_pos=None, goal_pos=PositionOrientation(*waypoint, random.random()*2*np.pi))

        if self._is_done:
            self._is_done = False
            return True
        return False

    def _set_position(self, pos: geometry_msgs.PoseWithCovarianceStamped):
        poso = PositionOrientation(
            pos.pose.pose.position.x,
            pos.pose.pose.position.y,
            euler_from_quaternion(
                [
                    pos.pose.pose.orientation.x,
                    pos.pose.pose.orientation.y,
                    pos.pose.pose.orientation.z,
                    pos.pose.pose.orientation.w
                ]
            )[2]
        )

        for robot in self.robot_managers:
            robot.reset(poso, None)

    def _set_goal(self, pos: geometry_msgs.PoseStamped):
        poso = PositionOrientation(
            pos.pose.position.x,
            pos.pose.position.y,
            euler_from_quaternion(
                [
                    pos.pose.orientation.x,
                    pos.pose.orientation.y,
                    pos.pose.orientation.z,
                    pos.pose.orientation.w
                ]
            )[2]
        )

        for robot in self.robot_managers:
            robot.reset(None, poso)

    def _new_scenario(self, *args, **kwargs):
        self.reset(lambda: None)
