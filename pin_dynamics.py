# pin_dynamics.py

import pinocchio as pin
import numpy as np
import matplotlib.pyplot as plt

def quat_to_euler_xyz(quat):
    """
    Convert a quaternion [x, y, z, w] to Euler angles in ZYX order (roll, pitch, yaw).
    
    Parameters:
        quat (array-like): Quaternion in [x, y, z, w] format.
    
    Returns:
        np.ndarray: Euler angles [roll, pitch, yaw] in radians.
    """
    x, y, z, w = quat

    # Roll (x-axis rotation)
    sinr_cosp = 2.0 * (w * x + y * z)
    cosr_cosp = 1.0 - 2.0 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)

    # Pitch (y-axis rotation)
    sinp = 2.0 * (w * y - z * x)
    if abs(sinp) >= 1:
        pitch = np.sign(sinp) * np.pi / 2  # Use 90 degrees if out of range (clamping)
    else:
        pitch = np.arcsin(sinp)

    # Yaw (z-axis rotation)
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)

    return np.array([roll, pitch, yaw])
class RobotDynamics:
    def __init__(self, urdf_path, shell_radius, dt):
        """
        初始化机器人动力学仿真器。
        """
        self.model = pin.buildModelFromUrdf(urdf_path, pin.JointModelFreeFlyer())
        self.data = self.model.createData()
        self.dt = dt
        self.shell_radius = shell_radius
        self.shell_frame_id = self.model.getFrameId("shell")
        self.pendulum_frame_id = self.model.getFrameId("pendulum")
        self.flywheel_frame_id = self.model.getFrameId("flywheel")
        print(f"shell frame id: {self.shell_frame_id}, pendulum frame id: {self.pendulum_frame_id}, flywheel frame id: {self.flywheel_frame_id}")
        self.q = pin.neutral(self.model)
        self.v = np.zeros(self.model.nv)
        self.q[2] = self.shell_radius
        self.r = np.array([0., 0., self.shell_radius])
        self.max_spin_friction = 15.0  # 最大自旋摩擦力矩
        self.coef_spin_friction = 10   # 自旋粘性
        self.coef_rolling_friction = 10 # 滚动阻力系数
        self.max_joint_friction = np.array([1, 0.5, 0.5])  # 最大关节摩擦力矩
        self.coef_joint_friction = np.array([10, 5, 10])   # 关节粘性
        for i, name in enumerate(self.model.names):
            joint = self.model.joints[i]
            print(f"Joint {i}: {name}, nq={joint.nq}, nv={joint.nv}")

    def get_state(self):
        motors_velocity = self.v[6:].copy()
        motors_position = self.q[7:].copy()

        rpy = quat_to_euler_xyz(self.q[3:7])

        angular_velocity = self.v[3:6].copy()
        return motors_velocity, motors_position, rpy, angular_velocity

    def set_state(self, q, v):
        self.q = q.copy()
        self.v = v.copy()

    def calculate_friction(self, local_r, tau):
        local_r_normal = local_r / np.linalg.norm(local_r) # 地面法向量方向
        w_shell = self.v[3:6] + np.array([0, self.v[6], 0]).T
        w_r_local = np.dot(w_shell, local_r_normal) * local_r_normal  # 在地面法向量方向上的角速度，这部分旋转会受到地面的阻尼
        w_v_local = w_shell - w_r_local # 在地面切线方向上的角速度，这部分旋转受滚动阻尼
        # 自旋部分
        tau_r_local = np.dot(tau[3:6], local_r_normal) * local_r_normal
        spin_friction = np.zeros(3)
        if(np.linalg.norm(tau_r_local) < self.max_spin_friction and np.linalg.norm(w_r_local) < 1e-4):
            spin_friction = -tau_r_local
        else:
            spin_friction = -np.sign(w_r_local) * self.max_spin_friction - self.coef_spin_friction * w_r_local
        # 滚动阻力
        rolling_friction = -np.sign(w_v_local) * self.coef_rolling_friction
        # 关节摩擦
        joint_friction = np.zeros(3)
        for i in range(6, 9):
            if(abs(self.v[i]) < 1e-4 and abs(tau[i]) < self.max_joint_friction[i-6]):
                joint_friction[i-6] = -tau[i]
            else:
                joint_friction[i-6] = -np.sign(self.v[i]) * self.max_joint_friction[i-6] - self.coef_joint_friction[i-6] * self.v[i]
        return spin_friction, rolling_friction, joint_friction
    def step_forward_dynamics_manual(self, tau_actuators):
        q = self.q
        v = self.v
        model = self.model
        data = self.data

        # 1. 计算关节雅可比矩阵（在世界坐标系）
        pin.computeJointJacobians(model, data, q)
        pin.forwardKinematics(model, data, q, v)
        
        # 2. 计算非线性项
        nle = pin.rnea(model, data, q, v, np.zeros(model.nv))
        
        # 3. 构建总力矩
        tau = np.zeros(model.nv)
        tau[6:] = tau_actuators
        
        # 4. 约束雅可比 - 确保在世界坐标系
        # 获取基座在世界坐标系中的位置和姿态
        pin.updateFramePlacements(model, data)
        base_pose = data.oMi[1]  # 浮动基座的位姿
        
        local_r = base_pose.rotation.T @ self.r
        
        Jc = np.zeros((3, model.nv))
        Jc[:, :3] = np.eye(3)  # 位置部分
        Jc[:, 3:6] = pin.skew(local_r)  # 姿态部分
        Jc[:, 6] = pin.skew(local_r)[:, 1]
        # 5. 计算质量矩阵
        pin.crba(model, data, q)
        M = data.M.copy()
        # 6. 计算摩擦力
        spin_friction, rolling_friction, joint_friction = self.calculate_friction(local_r, tau)
        tau[3:6] += spin_friction + rolling_friction
        tau[6:] += joint_friction
        # 7. 构建并求解系统
        n_constraints = Jc.shape[0]
        system_matrix = np.zeros((model.nv + n_constraints, model.nv + n_constraints))
        system_matrix[:model.nv, :model.nv] = M
        system_matrix[:model.nv, model.nv:] = Jc.T
        system_matrix[model.nv:, :model.nv] = Jc
        

        rhs = np.zeros(model.nv + n_constraints)
        rhs[:model.nv] = tau - nle
        rhs[model.nv:] = 0  
        
        solution = np.linalg.solve(system_matrix, rhs)
        
        dv = solution[:model.nv]
        lambda_c = -solution[model.nv:]
        
        # 7. 积分更新
        v_next = v + dv * self.dt
        w_shell_next = v_next[3:6] + np.array([0, v_next[6], 0]).T
        v_next[:3] = np.cross(w_shell_next, local_r)  # 更新线速度
        q_next = pin.integrate(model, q, v_next * self.dt)
        if(q_next[8]>0.523):
            q_next[8]=0.523
            v_next[7]=0
        elif(q_next[8]<-0.523):
            q_next[8]=-0.523
            v_next[7]=0
        self.q = q_next
        self.v = v_next
        return dv, lambda_c

if __name__ == "__main__":
    urdf_path = "sr_description/urdf/ball.urdf"
    mesh_dir = "sr_description"
    
    sim = RobotDynamics(
        urdf_path=urdf_path,
        shell_radius=0.4,
        dt=0.001
    )
    q0 = pin.neutral(sim.model)
    v0 = np.zeros(sim.model.nv)

    sim.set_state()
    motors_velocity, motors_position, euler_angles, angular_velocity = sim.get_state()
    print(motors_velocity, motors_position, euler_angles, angular_velocity)