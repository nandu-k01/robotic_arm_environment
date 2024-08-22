import os
import sys
import time
import rclpy
import random
import numpy as np
import message_filters
from rclpy.node import Node
from sensor_msgs.msg import JointState
from gazebo_msgs.msg import ModelStates
from gazebo_msgs.srv import SetEntityState

import tf2_ros
from tf2_ros import TransformException

from rclpy.action import ActionClient
from trajectory_msgs.msg import JointTrajectoryPoint
from control_msgs.action import FollowJointTrajectory

from rclpy.duration import Duration
import torch.nn as nn

class MyRLEnvironmentNode(Node):

    def __init__(self):
        super().__init__('node_main_rl_environment')

        # End-effector transformation
        self.tf_buffer = tf2_ros.Buffer()
        self.tf_listener = tf2_ros.TransformListener(self.tf_buffer, self)

        # Client for resetting the sphere position
        self.client_reset_sphere = self.create_client(SetEntityState, '/gazebo/set_entity_state')
        while not self.client_reset_sphere.wait_for_service(timeout_sec=1.0):
            self.get_logger().info('sphere reset-service not available, waiting...')
        self.request_sphere_reset = SetEntityState.Request()

        # Action client to change joints position
        self.trajectory_action_client = ActionClient(self, FollowJointTrajectory, '/joint_trajectory_controller/follow_joint_trajectory')

        # Subscribers
        self.joint_state_subscription = self.create_subscription(
            JointState,
            '/joint_states',
            self.joint_state_callback,
            10
        )
        self.target_point_subscription = self.create_subscription(
            ModelStates,
            '/gazebo/model_states',
            self.target_point_callback,
            10
        )

        # Variables to store latest messages
        self.joint_state_msg = None
        self.target_point_msg = None

    def joint_state_callback(self, msg):
        self.joint_state_msg = msg
        self.process_data()

    def target_point_callback(self, msg):
        self.target_point_msg = msg
        self.process_data()

    def process_data(self):
        if self.joint_state_msg is not None and self.target_point_msg is not None:
            # Process the synchronized data here
            # self.get_logger().info('Processing synchronized data')

            # Position of each joint:
            self.joint_1_pos = self.joint_state_msg.position[2]
            self.joint_2_pos = self.joint_state_msg.position[0]
            self.joint_3_pos = self.joint_state_msg.position[1]
            self.joint_4_pos = self.joint_state_msg.position[3]
            self.joint_5_pos = self.joint_state_msg.position[4]
            self.joint_6_pos = self.joint_state_msg.position[5]

            # Velocity of each joint:
            self.joint_1_vel = self.joint_state_msg.velocity[2]
            self.joint_2_vel = self.joint_state_msg.velocity[0]
            self.joint_3_vel = self.joint_state_msg.velocity[1]
            self.joint_4_vel = self.joint_state_msg.velocity[3]
            self.joint_5_vel = self.joint_state_msg.velocity[4]
            self.joint_6_vel = self.joint_state_msg.velocity[5]
            # self.get_logger().info(f'Joint positions: {self.joint_1_pos}, {self.joint_2_pos}, {self.joint_3_pos}, {self.joint_4_pos}, {self.joint_5_pos}, {self.joint_6_pos}')

            # Determine the sphere position in Gazebo wrt world frame
            sphere_index = self.target_point_msg.name.index('my_sphere')  # Get the correct index for the sphere
            self.pos_sphere_x = self.target_point_msg.pose[sphere_index].position.x
            self.pos_sphere_y = self.target_point_msg.pose[sphere_index].position.y
            self.pos_sphere_z = self.target_point_msg.pose[sphere_index].position.z

            # Determine the pose(position and location) of the end-effector w.r.t. world frame
            self.robot_x, self.robot_y, self.robot_z = self.get_end_effector_transformation()

    def get_end_effector_transformation(self):
        # Determine the pose(position and location) of the end effector w.r.t. world frame
        try:
            now = rclpy.time.Time()
            self.reference_frame = 'world'
            self.child_frame = 'link6'
            trans = self.tf_buffer.lookup_transform(self.reference_frame, self.child_frame, now)  # This calculates the position of link6 w.r.t. world frame
        except TransformException as ex:
            self.get_logger().info(f'Could not transform {self.reference_frame} to {self.child_frame}: {ex}')
            return
        else:
            # Translation
            ef_robot_x = trans.transform.translation.x
            ef_robot_y = trans.transform.translation.y
            ef_robot_z = trans.transform.translation.z
            # Rotation
            ef_qx = trans.transform.rotation.x
            ef_qy = trans.transform.rotation.y
            ef_qz = trans.transform.rotation.z
            ef_qw = trans.transform.rotation.w

            return round(ef_robot_x, 3), round(ef_robot_y, 3), round(ef_robot_z, 3)

    def reset_environment_request(self):
        print("reset called")
        # Every time this function is called, a request to reset the environment is sent 
        # i.e. Move the robot to home position and change the
        # sphere location and wait until a response/confirmation

        # -------------------- Reset sphere position ------------------#
        sphere_position_x = random.uniform(0.05, 1.05)
        sphere_position_y = random.uniform(-0.5, 0.5)
        sphere_position_z = random.uniform(0.05, 1.05)

        self.request_sphere_reset.state.name = 'my_sphere'
        self.request_sphere_reset.state.reference_frame = 'world'
        self.request_sphere_reset.state.pose.position.x = sphere_position_x
        self.request_sphere_reset.state.pose.position.y = sphere_position_y
        self.request_sphere_reset.state.pose.position.z = sphere_position_z

        self.future_sphere_reset = self.client_reset_sphere.call_async(self.request_sphere_reset)
        # self.get_logger().info('Resetting sphere to new position...')
        rclpy.spin_until_future_complete(self, self.future_sphere_reset)

        sphere_service_response = self.future_sphere_reset.result()

        # --------------------- Reset robot position -------------------#
        home_point_msg = JointTrajectoryPoint()
        home_point_msg.positions = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        home_point_msg.velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        home_point_msg.accelerations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        home_point_msg.time_from_start = Duration(seconds=2).to_msg()

        joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
        home_goal_msg = FollowJointTrajectory.Goal()
        home_goal_msg.goal_time_tolerance = Duration(seconds=1).to_msg()
        home_goal_msg.trajectory.joint_names = joint_names
        home_goal_msg.trajectory.points = [home_point_msg]

        self.trajectory_action_client.wait_for_server()  # Waits for the action server to be available
        send_home_goal_future = self.trajectory_action_client.send_goal_async(home_goal_msg)  # Sending home-position request
        rclpy.spin_until_future_complete(self, send_home_goal_future)  # Wait for goal status
        goal_reset_handle = send_home_goal_future.result()

        get_reset_result = goal_reset_handle.get_result_async()
        rclpy.spin_until_future_complete(self, get_reset_result)  # Wait for response
        return self.state_space_funct()

    def action_step_service(self, action_values):
        
        # Every time this function is called, it passes the action vector (desire position of each joint) 
        # to the action-client to execute the trajectory
        
        points = []

        point_msg = JointTrajectoryPoint()
        point_msg.positions     = action_values
        point_msg.velocities    = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        point_msg.accelerations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
        point_msg.time_from_start = Duration(seconds=2.0).to_msg() # be careful about this time 
        points.append(point_msg) 

        joint_names = ['joint1','joint2','joint3','joint4','joint5','joint6']
        goal_msg    = FollowJointTrajectory.Goal()
        goal_msg.goal_time_tolerance = Duration(seconds=1).to_msg() # goal_time_tolerance allows some freedom in time, so that the trajectory goal can still
                                                                    # succeed even if the joints reach the goal some time after the precise end time of the trajectory.
                                                            
        goal_msg.trajectory.joint_names = joint_names
        goal_msg.trajectory.points      = points

        # self.get_logger().info('Waiting for action server to move the robot...')
        self.trajectory_action_client.wait_for_server() # waits for the action server to be available

        # self.get_logger().info('Sending goal-action request...')
        self.send_goal_future = self.trajectory_action_client.send_goal_async(goal_msg) 

        # self.get_logger().info('Checking if the goal is accepted...')
        rclpy.spin_until_future_complete(self, self.send_goal_future ) # Wait for goal status

        goal_handle = self.send_goal_future.result()

        # if not goal_handle.accepted:
            # self.get_logger().info(' Action-Goal rejected ')
            # return
        # self.get_logger().info('Action-Goal accepted')

        # self.get_logger().info('Checking the response from action-service...')
        self.get_result = goal_handle.get_result_async()
        rclpy.spin_until_future_complete(self, self.get_result ) # Wait for response

        # if self.get_result.result().result.error_code == 0:
            # self.get_logger().info('Action Completed without problem')
        # else:
            # self.get_logger().info('There was a problem with the accion')

    # def action_step_service(self, action_values):
    #     # Every time this function is called, it passes the action vector (desired position of each joint) 
    #     # to the action-client to execute the trajectory
    #     points = []

    #     point_msg = JointTrajectoryPoint()
    #     point_msg.positions = action_values
    #     point_msg.velocities = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     point_msg.accelerations = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    #     point_msg.time_from_start = Duration(seconds=2.0).to_msg()
    #     points.append(point_msg)

    #     joint_names = ['joint1', 'joint2', 'joint3', 'joint4', 'joint5', 'joint6']
    #     goal_msg = FollowJointTrajectory.Goal()
    #     goal_msg.goal_time_tolerance = Duration(seconds=1).to_msg()
    #     goal_msg.trajectory.joint_names = joint_names
    #     goal_msg.trajectory.points = points

    #     self.trajectory_action_client.wait_for_server()  # Waits for the action server to be available
    #     self.send_goal_future = self.trajectory_action_client.send_goal_async(goal_msg)
    #     rclpy.spin_until_future_complete(self, self.send_goal_future)  # Wait for goal status
    #     goal_handle = self.send_goal_future.result()

    #     if not goal_handle.accepted:
    #         self.get_logger().info('Goal rejected')
    #         return

    #     self.get_logger().info('Goal accepted')
    #     get_result_future = goal_handle.get_result_async()
    #     rclpy.spin_until_future_complete(self, get_result_future)  # Wait for result
    #     result = get_result_future.result().result
    #     self.get_logger().info(f'Action result: {result}')

    def generate_action_funct(self):
        # Generates a random action (desired position for each joint within specified range)
        action_values = [
            random.uniform(-1.5, 1.5),
            random.uniform(-1.5, 1.5),
            random.uniform(-1.5, 1.5),
            random.uniform(-1.5, 1.5),
            random.uniform(-1.5, 1.5),
            random.uniform(-1.5, 1.5),
        ]
        return action_values


    def calculate_reward_funct(self):

        # I aim with this function to get the reward value. For now, the reward is based on the distance
        # i.e. Calculate the euclidean distance between the link6 (end effector) and sphere (target point)
        # and each timestep the robot receives -1 but if it reaches the goal (distance < 0.05) receives +10


        try:
            robot_end_position    = np.array((self.robot_x, self.robot_y, self.robot_z))
            target_point_position = np.array((self.pos_sphere_x, self.pos_sphere_y, self.pos_sphere_z))
            # print(robot_end_position, 'robot end position')
            # print(np.abs(robot_end_position), 'robot end position')
            # print(target_point_position, 'target point position')
            # print(np.abs(target_point_position), 'target point position')



        except: 
            self.get_logger().info('could not calculate the distance yet, trying again...')
            return 

        else:
            # distance = np.linalg.norm(robot_end_position - target_point_position)
            # loss = nn.MSELoss()
            distance = np.mean((robot_end_position - target_point_position) ** 2)

            
            if distance <= 0.05:
                self.get_logger().info('Goal Reached')
                done = True
                reward_d = 10
            else:
                done = False
                reward_d = -distance
                # self.get_logger().info('Goal Not Reached')


            return reward_d, done

    def state_space_funct(self):
        # Returns state vector
        state = [
            self.robot_x,
            self.robot_y,
            self.robot_z,
            self.joint_1_pos,
            self.joint_2_pos,
            self.joint_3_pos,
            self.joint_4_pos,
            self.joint_5_pos,
            self.joint_6_pos,
            self.pos_sphere_x,
            self.pos_sphere_y,
            self.pos_sphere_z
        ]
        return np.array(state)


# def main(args=None):
#     rclpy.init(args=args)
#     node = MyRLEnvironmentNode()

#     while rclpy.ok():
#         rclpy.spin_once(node, timeout_sec=0.1)
#         # Example usage:
#         action_values = node.generate_action_funct()
#         node.action_step_service(action_values)
#         reward = node.calculate_reward_funct()
#         state = node.state_space_funct()
#         node.get_logger().info(f'State: {state}, Reward: {reward}')

#     node.destroy_node()
#     rclpy.shutdown()


# if __name__ == '__main__':
#     main()
