import numpy as np
import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import sys
from pathlib import Path
import time
import math  # 导入math库用于sin函数

def main():
    # 1. 加载URDF模型
    urdf_path = "sr_description/urdf_/ball.urdf"
    mesh_dir ="sr_description"  # STL文件所在目录
    
    # 构建模型和几何
    model, collision_model, visual_model = pin.buildModelsFromUrdf(
        urdf_path, 
        mesh_dir, 
        pin.JointModelFreeFlyer()  # 注意：如果您的模型有浮动基座才需要这个
    )

    # 2. 初始化可视化器
    try:
        viz = MeshcatVisualizer(model, collision_model, visual_model)
        viz.initViewer(open=True)
        viz.loadViewerModel()
    except ImportError as err:
        print("错误: 需要安装Meshcat才能进行可视化")
        print("请运行: pip install meshcat")
        print(err)
        return
    
    # 3. 设置初始状态
    q0 = pin.neutral(model)
    viz.display(q0)
    
    # 4. 显示网格和碰撞体
    viz.displayVisuals(True)  # 显示视觉几何
    
    # 获取关节索引 (base2flywheel)
    joint_name = "base2flywheel"
    joint_id = model.getJointId(joint_name)
    
    # 确保找到关节
    if joint_id >= len(model.joints):
        print(f"错误: 未找到关节 '{joint_name}'")
        return
    
    # 5. 动画循环 - 使关节转动
    t = 0.0
    while True:
        # 更新关节角度 (使用正弦函数创建摆动效果)
        # 根据实际模型调整索引
        angle = 0.5 * t  # ±0.5弧度(≈±28.6度)的摆动
        
        # 复制当前状态
        q = q0.copy()
        
        # 方法2: 使用更精确的关节配置空间索引
        joint = model.joints[joint_id]
        if joint.idx_q >= 0:  # 确保关节有配置变量
            q[joint.idx_q] = angle
        
        # 更新模型显示
        viz.display(q)
        
        # 更新时间并暂停
        t += 0.1
        time.sleep(0.1)

if __name__ == "__main__":
    main()