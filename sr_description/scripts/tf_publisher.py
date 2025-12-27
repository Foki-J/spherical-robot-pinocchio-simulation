#!/usr/bin/env python3

import rospy
import tf2_ros
import geometry_msgs.msg
import sensor_msgs.msg
from nav_msgs.msg import Odometry
from std_msgs.msg import Float64MultiArray
from communication_msgs.msg import motors
from numpy import pi
class TfAndJointStatePublisher:
    def __init__(self):
        # 初始化节点
        rospy.init_node('tf_and_joint_state_publisher')

        # 创建一个tf2广播器
        self.br = tf2_ros.TransformBroadcaster()

        # 创建一个JointState发布者
        self.joint_state_pub = rospy.Publisher('/joint_states', sensor_msgs.msg.JointState, queue_size=10)

        # 订阅/odom话题
        self.odom_sub = rospy.Subscriber('/odom', Odometry, self.odom_callback)

        # 订阅/motors_state话题
        self.motors_state_sub = rospy.Subscriber('/motors_state', motors, self.motors_state_callback)

        # 初始化JointState消息
        self.joint_state = sensor_msgs.msg.JointState()
        self.joint_state.name = ['base_to_shell_joint', 'base_to_pendulum_joint']
        self.joint_state.position = [0.0, 0.0]
        self.joint_state.velocity = []
        self.joint_state.effort = []

    def odom_callback(self, msg):
        # 获取当前时间
        current_time = rospy.Time.now()

        # 创建TransformStamped消息
        t = geometry_msgs.msg.TransformStamped()
        t.header.stamp = current_time
        t.header.frame_id = "map"
        t.child_frame_id = "base"

        # 设置平移部分
        t.transform.translation.x = msg.pose.pose.position.x
        t.transform.translation.y = msg.pose.pose.position.y
        t.transform.translation.z = msg.pose.pose.position.z
        # 设置旋转部分
        t.transform.rotation = msg.pose.pose.orientation

        # 广播变换
        self.br.sendTransform(t)

    def motors_state_callback(self, msg):
        # 更新JointState消息的position字段
        
        self.joint_state.position = [ (msg.first.position/ 18000.0 * pi)%(2*pi)-pi, msg.second.position]

        # 设置时间戳
        self.joint_state.header.stamp = rospy.Time.now()

        # 发布JointState消息
        self.joint_state_pub.publish(self.joint_state)

if __name__ == '__main__':
    try:
        node = TfAndJointStatePublisher()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass