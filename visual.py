import pinocchio as pin
from pinocchio.visualize import MeshcatVisualizer
import numpy as np
import meshcat.geometry as g
import meshcat.transformations as tf
from typing import Optional

class RobotVisualizer:
    def __init__(self, urdf_path: str, mesh_dir: str):
        """
        增强版机器人可视化器，支持显示力矩条
        """
        self.urdf_path = urdf_path
        self.mesh_dir = mesh_dir
        
        # 加载模型
        self.model, self.collision_model, self.visual_model = pin.buildModelsFromUrdf(
            urdf_path, 
            mesh_dir, 
            pin.JointModelFreeFlyer()
        )
        
        # 初始化可视化器
        self.viz = None
        self.initialize_visualizer()
    
    def initialize_visualizer(self):
        """初始化Meshcat可视化器"""
        try:
            self.viz = MeshcatVisualizer(
                self.model, 
                self.collision_model, 
                self.visual_model
            )
            self.viz.initViewer(open=True)
            self.viz.loadViewerModel()
            self.viz.displayVisuals(True)
            
            # 初始显示中性位置
            q0 = pin.neutral(self.model)
            self.display(q0)
        except ImportError as err:
            print("错误: 需要安装Meshcat才能进行可视化")
            print("请运行: pip install meshcat")
            raise err
    def display(self, q: np.ndarray):
        """显示给定的配置向量q"""
        self.viz.display(q)

    def reset(self):
        """重置为中性配置"""
        q0 = pin.neutral(self.model)
        self.display(q0)