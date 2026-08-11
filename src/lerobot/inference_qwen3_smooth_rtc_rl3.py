import time
import torch
import math
from dataclasses import dataclass, field
from threading import Lock, Thread
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig 
from lerobot.datasets.utils import build_dataset_frame, hw_to_dataset_features
from lerobot.policies.factory import get_policy_class, make_pre_post_processors
from lerobot.policies.rtc.action_queue import ActionQueue
from lerobot.policies.rtc.configuration_rtc import RTCConfig
from lerobot.policies.rtc.latency_tracker import LatencyTrackerInt
from lerobot.processor.factory import (
    make_default_robot_action_processor,
    make_default_robot_observation_processor,
)
from lerobot.robots import (
    Robot,
    RobotConfig,
    make_robot_from_config,
    single_piper,
)
import numpy as np

# 导入必要的库
import numpy as np  # 数值计算
import torch  # PyTorch深度学习框架
import time  # 时间操作
from collections import deque  # 双端队列，用于存储历史动作
import requests  # HTTP请求库
import threading  # 多线程支持
from PIL import Image  # 图像处理
from transformers import AutoProcessor, AutoModelForImageTextToText  # HuggingFace transformers

# 导入LeRobot相关配置和模块
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig
from lerobot.datasets.lerobot_dataset import LeRobotDatasetMetadata
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.utils import build_inference_frame, make_robot_action

# 导入机器人模型相关模块
from lerobot.robots import (  # noqa: F401
    Robot,  # 机器人基类
    RobotConfig,  # 机器人配置类
    make_robot_from_config,  # 从配置创建机器人实例的工厂函数
    single_piper,  # 单臂Piper机器人
    # moving_dual_piper,  # 双臂移动机器人(暂未使用)
)

# 导入相机相关模块
# from lerobot.cameras import (  # noqa: F401
#     CameraConfig,  # 相机配置类
# )
from dataclasses import asdict, dataclass  # 数据类装饰器
from lerobot.utils.robot_utils import busy_wait  # 忙等待函数
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401

# 导入策略模型类
from lerobot.policies.act.modeling_act import ACTPolicy  # ACT策略
from lerobot.policies.diffusion.modeling_diffusion import DiffusionPolicy  # Diffusion策略
from lerobot.policies.smolvla.modeling_smolvla import SmolVLAPolicy  # SmolVLA策略
from lerobot.policies.groot.modeling_groot import GrootPolicy  # Groot策略

# from lerobot.policies.xvla.modeling_xvla import XVLAPolicy  # XVLA策略(暂未使用)
from lerobot.policies.pi05.modeling_pi05 import PI05Policy  # PI05策略
from lerobot.configs.policies import PreTrainedConfig  # 预训练配置
from lerobot.configs import parser  # 命令行参数解析器

# 导入Piper机械臂SDK
from piper_sdk import *
from scipy.spatial.transform import Rotation as R  # 旋转矩阵处理

import casadi                     # CasADi 符号优化库，用于 IK 数值求解
import meshcat.geometry as mg     # Meshcat 可视化几何对象
import numpy as np               # 数值计算
import pinocchio as pin          # Pinocchio 动力学 / 运动学库
import time                       # 时间相关（未使用）

# 终端键盘相关（Linux / Windows 兼容）
try:
    import termios
    import tty
except ImportError:
    import msvcrt

# ROS 相关（当前被注释）
# import rospy
from pinocchio import casadi as cpin   # Pinocchio 的 CasADi 后端
from pinocchio.robot_wrapper import RobotWrapper
from pinocchio.visualize import MeshcatVisualizer

# ROS tf 转换（已替换为 scipy）
# from tf.transformations import quaternion_from_euler, euler_from_quaternion

import os
import sys
import threading

# Piper 控制与消息（当前未启用）
# from piper_control import PIPER
# from piper_msgs.msg import PosCmd

# 使用 scipy 的 Rotation 做欧拉角 ↔ 四元数
from scipy.spatial.transform import Rotation as R

# 后处理功能：
# #             处理                        位置                                                        作用
# 1、动作块滑窗平滑ArmSmoother；    类：234-327，调用：587；                                            对策略一次预测出的整段chunk(horizonx7)沿时间维做均值卷积（reflect padding）；角度维先 _unwarp_angle 解缠绕、平滑后再_wrap_angle, 避免±180°跳变。去抖动
# 2、RTC实时分块拼接                ：575-592+ ActionQueue.merge                                       用推理延迟和上一块剩余动作做跨chunk边界软衔接，消除“换块”瞬间的速度突变
# 3、夹爪非线性拼接                 :632-641，另有幂函数版 grapper_attenuation :497-514                 分段：ratio≥0.4 线性、0.04~0.4 用三次曲线、<0.04 直接归零；抑制小噪声误触，让开合更贴近物理。
# 4、关节硬件零位偏置               :626 JointCtrl(j0-1670, j1-104, j2+408, j3, j4+24348, j5+1432)     补偿从臂偏移（lerobot版本由0.4.4-->0.5.x导致的）
# 5、关节限位（可选）               :610-615 + clamp() :330                                            把每个关节夹到安全范围。
# 6、推理/执行解耦 + 定频执行       _collect_and_infer / _execute_action，busy_wait :646-649            推理与执行稳定 ~30Hz，队列缓冲。


task_dict = {
    "1": "Grab the package and place it on the pallet.",
    "2": "Flip the package if the barcode is not facing up.",
    "3": "Grab the package and place it into the box.",
}




class RobotWrapper:
    def __init__(self, robot: Robot):
        self.robot = robot
        self.lock = Lock()

    def get_observation(self) -> dict[str, torch.Tensor]:
        with self.lock:
            return self.robot.get_observation()

    def send_action(self, action: torch.Tensor):
        with self.lock:
            self.robot.send_action(action)

    def observation_features(self) -> list[str]:
        with self.lock:
            return self.robot.observation_features

    def action_features(self) -> list[str]:
        with self.lock:
            return self.robot.action_features


@dataclass
class InferenceConfig:
    robot: RobotConfig | None = None
    policy: PreTrainedConfig | None = None

    rtc: RTCConfig = field(
        default_factory=lambda: RTCConfig(
            enabled=True,
            execution_horizon=25,       # 最大引导区间长度
        )
    )

    duration: float = 36000             #10小时
    fps: float = 30.0                   # 控制频率

    get_actions_threshold: int = 32
    ckpt_path: str = None  # 模型检查点路径
    qwen3_vl_path: str = None  # 子任务规划模型路径
    subtask_planning_period: float = 1.0  # 子任务规划周期(秒)

    task: str = field(default="", metadata={"help": "Task to execute"})

    use_torch_compile: bool = field(
        default=False,
        metadata={"help": "Use torch.compile for faster inference (PyTorch 2.0+)"},
    )

    # def __post_init__(self):
    #     """初始化后处理，从命令行参数获取预训练模型路径"""
    #     policy_path = parser.get_path_arg("policy")
    #     if policy_path:
    #         cli_overrides = parser.get_cli_overrides("policy")
    #         self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
    #         self.policy.pretrained_path = policy_path
    #     # else:
    #     #     raise ValueError("Policy path is required")

    #     if self.robot is None:
    #         raise ValueError("Robot configuration must be provided")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """返回路径字段列表，使解析器能够通过--policy.path=local/dir加载配置"""
        return ["policy"]

def enable_fun(piper: C_PiperInterface):
    """
    使能机械臂并检测使能状态，尝试5秒，如果使能超时则退出程序

    参数:
        piper: Piper机械臂接口对象

    流程:
        1. 循环尝试使能机械臂
        2. 检查所有6个关节电机的驱动使能状态
        3. 如果5秒内未使能成功则退出程序
    """
    enable_flag = False  # 使能状态标志
    timeout = 5  # 超时时间(秒)
    start_time = time.time()  # 记录开始时间
    elapsed_time_flag = False  # 超时标志

    # 等待机械臂使能成功
    while not (enable_flag):
        elapsed_time = time.time() - start_time  # 计算已用时间
        # 检查6个关节电机是否全部使能成功
        enable_flag = (
            piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status
            and piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status
            and piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status
            and piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status
            and piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status
            and piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        )
        print("使能状态:", enable_flag)
        piper.EnableArm(7)  # 发送使能命令(参数7表示使能所有关节)
        if elapsed_time > timeout:  # 超时检查
            print("超时....")
            elapsed_time_flag = True  # 设置超时标志
            enable_flag = True  # 退出循环
            break
        time.sleep(1)  # 每秒检查一次

    if elapsed_time_flag:
        print("程序自动使能超时,退出程序")
        exit(0)  # 超时则退出程序




class ArmSmoother:
    def __init__(self, smooth_window=11, angle_indices=[]):
        self.smooth_window = smooth_window if smooth_window % 2 == 1 else smooth_window + 1
        self.pad = self.smooth_window // 2
        self.window = np.ones(self.smooth_window) / self.smooth_window
        self.angle_indices = angle_indices
        self.max_angle = 180000
        self.min_angle = -self.max_angle
        self.full_circle = self.max_angle - self.min_angle
        self.angle_clamp_range = (-180, 180)

    def _unwrap_angle(self, values):
        """角度解缠绕"""
        unwrapped = np.copy(values).astype(np.float64)
        for i in range(1, len(unwrapped)):
            diff = unwrapped[i] - unwrapped[i - 1]
            if diff > self.max_angle:
                unwrapped[i:] -= self.full_circle
            elif diff < -self.max_angle:
                unwrapped[i:] += self.full_circle
        return unwrapped

    def _wrap_angle(self, values):
        """角度重缠绕"""
        wrapped = np.copy(values)
        wrapped = wrapped % self.full_circle
        wrapped[wrapped >= self.max_angle] -= self.full_circle
        return wrapped.astype(np.float64)

    def _angle_weighted_average(self, angle1, angle2, weight1, weight2):
        """计算两个角度的加权平均, 兼容-180~180°环形"""
        scale = self.max_angle / 180
        norm_angle1 = angle1 / scale
        norm_angle2 = angle2 / scale

        total_weight = weight1 + weight2
        w1 = weight1 / total_weight
        w2 = weight2 / total_weight

        rad1 = np.radians(norm_angle1)
        rad2 = np.radians(norm_angle2)
        avg_cos = w1 * np.cos(rad1) + w2 * np.cos(rad2)
        avg_sin = w1 * np.sin(rad1) + w2 * np.sin(rad2)

        avg_rad = np.arctan2(avg_sin, avg_cos)
        avg_angle_norm = np.degrees(avg_rad)

        clamp_min, clamp_max = self.angle_clamp_range
        if avg_angle_norm > clamp_max:
            avg_angle_norm -= 360
        elif avg_angle_norm < clamp_min:
            avg_angle_norm += 360
        return avg_angle_norm * scale

    def smooth_column(self, col_data, need_angle_wrap = False):
        """
        单列平滑
        :param need_angle_wrap: 是否需要角度解缠绕/重缠绕
        """
        if len(col_data) < self.smooth_window:
            return col_data.astype(np.float64)

        col_processed = self._unwrap_angle(col_data) if need_angle_wrap else col_data.astype(np.float64)
        col_padded = np.pad(col_processed, pad_width=(self.pad, self.pad), mode='reflect')
        smoothed_col = np.convolve(col_padded, self.window, mode='valid')

        if need_angle_wrap:
            smoothed_col = self._wrap_angle(smoothed_col)
        return smoothed_col

    def smooth(self, data):
        """
        多列数据平滑 兼容 torch.Tensor / numpy array
        :param data: 2D数组
        """
        is_tensor = isinstance(data, torch.Tensor)
        if is_tensor:
            device = data.device
            data = data.detach().cpu().numpy()

        if len(data) < self.smooth_window:
            result = data.astype(np.float64)
        else:
            smoothed_batch = np.copy(data).astype(np.float64)
            for dim in range(data.shape[-1]):
                need_wrap = dim in self.angle_indices
                smoothed_col = self.smooth_column(data[:, dim], need_wrap)
                smoothed_batch[:, dim] = smoothed_col
            result = smoothed_batch

        if is_tensor:
            result = torch.from_numpy(result).to(device)

        return result

# 定义限位函数
def clamp(value, min_val, max_val):
    """将值限制在最小最大值之间"""
    return max(min(value, max_val), min_val)

class RobotInferenceSystem:
    """机器人推理系统，封装线程化的采集+推理、动作执行逻辑

    设计思路:
        1. 双线程架构: 一个线程负责图像采集和策略推理
                      另一个线程负责动作执行(保证实时性)
        2. 队列缓冲: 使用deque作为动作队列，解耦推理和执行
        3. 动作平滑: 对策略输出的动作进行滑动窗口平滑处理

    线程分工:
        - _collect_and_infer: 采集相机图像 -> 策略推理 -> 动作平滑 -> 放入队列
        - _execute_action: 从队列取动作 -> 发送给机器人执行
    """
    def __init__(self, cfg: InferenceConfig):
        self.cfg = cfg
        self.robot = None
        self.robot_wrapper = None
        self.policy = None
        self.robot_observation_processor = None
        self.robot_action_processor = None
        self.device = "cuda"
        
        self.running = False                        # 线程控制标志
        self.infer_collect_thread = None            # 采集+推理合并线程
        self.execute_thread = None                  # 动作执行线程

        self.subtask_planning_thread = None  # 子任务规划线程

        self.SMOOTH_WINDOW = 9  # 平滑窗口大小
        self.INFERENCE_PERIOD = 2  # 采集+推理周期(秒)
        self.EXECUTION_PERIOD = 0.0333  # 执行周期(秒)，约25Hz
        self.SUBTASK_PLANNING_PERIOD = cfg.subtask_planning_period  # 子任务规划周期(秒)

        self.smoother = ArmSmoother(self.SMOOTH_WINDOW)  # 动作平滑器

        # 动作数据缩放因子(将归一化动作转换为实际电机指令)
        self.pos_factor = 1000000  # 位置缩放因子
        self.angel_factor = 1000  # 角度缩放因子(0.001度)
        # self.grapper_factor = 70000  # 夹爪缩放因子
        self.grapper_factor = 1000  # 夹爪缩放因子
        self.factor = 1000  # 统一缩放因子


        # 子任务规划相关
        self.subtask_model = None  # 子任务规划模型
        self.subtask_processor = None  # 子任务规划处理器
        self.current_subtask = None  # 当前子任务
        self.subtask_lock = threading.Lock()  # 子任务线程锁
        
        # 子任务稳定检测
        self.candidate_subtask = None
        self.candidate_count = 0
        self.required_confirmations = 4  # 需要连续4次确认
        self.log_path= "action_log.txt"  # 日志文件路径

        # 关节限位
        # self.joint_limits = [(-3, 3)] * 6
        # self.joint_limits[0] = (-2.687*180/3.14, 2.687*180/3.14)     # 关节0限位
        # self.joint_limits[1] = (0.0, 3.403*180/3.14)        # 关节1限位
        # self.joint_limits[2] = (-3.0541012*180/3.14, 0.0)   # 关节2限位
        # self.joint_limits[3] = (-1.5499*180/3.14, 1.5499*180/3.14)   # 关节3限位
        # self.joint_limits[4] = (-1.22*180/3.14, 1.22*180/3.14)       # 关节4限位
        # self.joint_limits[5] = (-1.7452*180/3.14, 1.7452*180/3.14)   # 关节5限位


    def init_robot(self):
        """初始化机器人并完成使能

        流程:
            1. 创建机器人实例并连接
            2. 初始化策略模型
            3. 机器人使能
            4. 等待使能完成
        """
        # 创建机器人实例并连接
        self.robot = make_robot_from_config(self.cfg.robot)
        if not self.robot.is_connected:
            self.robot.connect()

        # 初始化策略模型
        self._init_policy()

        # 机器人使能
        self._enable_robot()

        # self.robot.piper.MotorAngleLimitMaxSpdSet(1, 1500, -420)   #限位关节1避免打到相机支架
        print(self.robot.piper.GetAllMotorAngleLimitMaxSpd())
        # 等待使能完成
        time.sleep(2.0)

    def _init_policy(self):
        if self.cfg.task == None:
            raise ValueError("You need to provide a task name.")
        policy_class = get_policy_class(self.cfg.policy.type)
        ckpt_path = self.cfg.ckpt_path

        # config = PreTrainedConfig.from_pretrained(self.cfg.policy.pretrained_path)
        config = PreTrainedConfig.from_pretrained(ckpt_path)

        if self.cfg.policy.type == "pi05" or self.cfg.policy.type == "pi0":
            config.compile_model = self.cfg.use_torch_compile
    
        self.policy = policy_class.from_pretrained(ckpt_path, config=config)
        self.policy.config.rtc_config = self.cfg.rtc
        self.policy.init_rtc_processor()
    
        assert self.policy.name in ["smolvla", "pi05", "pi0"], "Only smolvla, pi05, and pi0 are supported for RTC"
    
        self.policy = self.policy.to(self.device)
        self.policy.eval()

        self.robot_observation_processor = make_default_robot_observation_processor()
        self.robot_action_processor = make_default_robot_action_processor()

        self.action_queue = ActionQueue(self.cfg.rtc) 
        # 初始化子任务规划模型
        self._init_subtask_planning_model()

    def _init_subtask_planning_model(self):
        """初始化子任务规划多模态大模型
        
        加载Qwen3-VL-2B子任务规划模型，用于根据当前图像确定下一个子任务
        """
        qwen3_vl_path = self.cfg.qwen3_vl_path
        if qwen3_vl_path is None:
            print("未配置子任务规划模型路径，跳过加载")
            return
            
        print(f"加载子任务规划模型: {qwen3_vl_path}")
        
        # 加载多模态大模型
        self.subtask_model = AutoModelForImageTextToText.from_pretrained(
            qwen3_vl_path, dtype="auto", device_map="auto"
        )
        self.subtask_processor = AutoProcessor.from_pretrained(qwen3_vl_path)
        
        # 设置为推理模式
        self.subtask_model.eval()
        
        # 初始化当前子任务
        self.current_subtask = self.cfg.task
        print(f"初始子任务: {self.current_subtask}")

    def _enable_robot(self):
        """根据机器人类型执行使能操作

        使能机械臂的各个关节电机，使其可以响应控制命令
        """
        self.robot = make_robot_from_config(self.cfg.robot)
        self.robot.connect()
        
        robot_type = self.cfg.robot.type
        if robot_type == "single_piper":
            self.robot.piper.EnableArm(7)
            self.robot.enable_fun()
            self.robot.piper.MotionCtrl_2(0x01, 0x01, 60, 0x00)  #关节控制
            self.robot.piper.GripperCtrl(round(1.0 * 70 * 1000), 2000, 0x01, 0)
            self.robot._enable_flag = True
        else:
            raise ValueError(f"Unsupported robot type: {robot_type}")

        self.robot_wrapper = RobotWrapper(self.robot)

    def grapper_attenuation(self, raw_value, full_open, mid=0.6, steepness=4):
        """夹爪开合度非线性衰减

        将线性输入转换为非线性输出，使夹爪开合更符合实际物理特性

        参数:
            raw_value: 原始输入值
            full_open: 夹爪全开最大值
            mid: 非线性转折点
            steepness: 陡度(控制曲线形状)

        返回:
            衰减后的夹爪控制值
        """
        x = np.clip(raw_value, 0, full_open) / full_open  # 归一化到[0,1]
        # 使用分段函数: x较小时使用幂函数，x较大时使用线性
        y = (x / mid) ** steepness * mid if x <= mid else x
        return int(y * full_open)
    
    def _collect_and_infer(self):
        """采集+推理线程函数

        循环执行:
            1. 获取机器人观测(相机图像+关节状态)
            2. 策略推理得到动作
            3. 动作平滑处理
            4. 放入动作队列
            5. 控制推理频率
        """
        latency_tracker = LatencyTrackerInt()
        is_first_infer = True
        
        dataset_features = hw_to_dataset_features(self.robot_wrapper.observation_features(), "observation")

        preprocessor, postprocessor = make_pre_post_processors(
            policy_cfg=self.cfg.policy,
            pretrained_path=self.cfg.ckpt_path,
            dataset_stats=None,
            preprocessor_overrides={
                "device_processor": {"device": self.device},
            },
        )

        while self.running:
            if self.action_queue.qsize() <= self.cfg.get_actions_threshold:
                index_before = self.action_queue.get_action_index()
                prev_left_actions = self.action_queue.get_left_over()
                
                obs = self.robot_wrapper.get_observation()
                obs["joint_1.pos"] = obs["joint_1.pos"]*57.3
                obs["joint_2.pos"] = obs["joint_2.pos"]*57.3
                obs["joint_3.pos"] = obs["joint_3.pos"]*57.3
                obs["joint_4.pos"] = obs["joint_4.pos"]*57.3
                obs["joint_5.pos"] = obs["joint_5.pos"]*57.3
                obs["joint_6.pos"] = obs["joint_6.pos"]*57.3
                obs["gripper.pos"] = abs(obs["gripper.pos"]*70)
                # print(obs["gripper.pos"] )

                obs_processed = self.robot_observation_processor(obs)

                obs_frame = build_dataset_frame(dataset_features, obs_processed, prefix="observation")
                for name in obs_frame:
                    obs_frame[name] = torch.from_numpy(obs_frame[name])
                    if "image" in name:
                        obs_frame[name] = (obs_frame[name].type(torch.float32) / 255)
                        obs_frame[name] = (obs_frame[name].permute(2, 0, 1).contiguous())
                    obs_frame[name] = obs_frame[name].unsqueeze(0)
                    obs_frame[name] = obs_frame[name].to(self.device)
                # obs_frame["task"] = [self.cfg.task]
                # 获取当前子任务(线程安全)
                with self.subtask_lock:
                    current_task = self.current_subtask
                obs_frame["task"] = [current_task]  # 添加任务描述    

                obs_frame["robot_type"] = (self.robot_wrapper.robot.name if hasattr(self.robot_wrapper.robot, "name") else "")
                
                preproceseded_obs = preprocessor(obs_frame)

                actions = self.policy.predict_action_chunk(
                    preproceseded_obs,
                    inference_delay=latency_tracker.max(),
                    prev_chunk_left_over=prev_left_actions,
                )  

                origin_actions = actions.squeeze(0).clone()


                posted_actions = postprocessor(actions)
                posted_actions = posted_actions.squeeze(0)

                posted_actions = self.smoother.smooth(posted_actions)
                # print("new ", end='')

                infer_steps = self.action_queue.get_action_index() - index_before
                
                self.action_queue.merge(origin_actions, posted_actions, infer_steps, index_before)

                if not is_first_infer:
                    latency_tracker.add(infer_steps)
                is_first_infer = False
            else:
                time.sleep(0.1)

    def _execute_action(self):
        action_interval = 1.0 / self.cfg.fps
        # i = 0
        while self.running:
            start_time = time.perf_counter()
            action = self.action_queue.get()

            if action is not None:
                action = action.cpu().numpy()
                # 单臂控制
                # joint_0 = round(clamp(action[0], *self.joint_limits[0]) * self.factor)
                # joint_1 = round(clamp(action[1], *self.joint_limits[1]) * self.factor)
                # joint_2 = round(clamp(action[2], *self.joint_limits[2]) * self.factor)
                # joint_3 = round(clamp(action[3], *self.joint_limits[3]) * self.factor)
                # joint_4 = round(clamp(action[4], *self.joint_limits[4]) * self.factor)
                # joint_5 = round(clamp(action[5], *self.joint_limits[5]) * self.factor)
                joint_0 = int(action[0] * self.factor)
                joint_1 = int(action[1] * self.factor)
                joint_2 = int(action[2] * self.factor)
                joint_3 = int(action[3] * self.factor)
                joint_4 = int(action[4] * self.factor)
                joint_5 = int(action[5] * self.factor)
                # 控制末端位姿(x, y, z, roll, pitch, yaw)
                # print(joint_0 - 1670, joint_1 - 104, joint_2 + 408, joint_3, joint_4 + 24348, joint_5 + 1432)
                # if i<50:
                # self.robot.piper.JointCtrl(joint_0 , joint_1 , joint_2 , joint_3, joint_4 , joint_5 )
                self.robot.piper.JointCtrl(joint_0 - 1670, joint_1 - 104, joint_2 + 408, joint_3, joint_4 + 24348, joint_5 + 1432)

                # i = i+1
                # 控制夹爪开合(应用非线性衰减)
                # gripper = self.grapper_attenuation(joint_6, self.grapper_factor*100)
               
                gripper_pos_raw = round(action[6]*self.grapper_factor)
                gripper_ratio = np.clip(gripper_pos_raw, 0, 101100) / 101100 
                if gripper_ratio >= 0.4:
                    gripper_attenuation = int(gripper_ratio * 101100)
                elif gripper_ratio >= 0.04:
                    ratio_in_range = (gripper_ratio - 0.04) / 0.36
                    gripper_attenuation = int((ratio_in_range ** 3) * 0.4 * 101100)
                else:
                    gripper_attenuation = 0
                self.robot.piper.GripperCtrl(gripper_attenuation, 1000, 0x01, 0)
                # print(joint_6)
            # dt_s = time.perf_counter() - start_time
            # time.sleep(max(0, (action_interval - dt_s) - 0.001))  # 避免sleep精度导致的睡过
            # 控制执行频率
            dt = time.perf_counter() - start_time
            wait_time = self.EXECUTION_PERIOD - dt
            if wait_time > 0:
                busy_wait(wait_time)

    def _plan_subtask(self):
        """子任务规划线程函数
        
        定期调用多模态大模型，根据当前相机图像确定下一个子任务
        
        循环执行:
            1. 获取俯视相机图像
            2. 调用Qwen3-VL模型进行子任务规划
            3. 更新当前子任务
            4. 控制规划频率
        """
        if self.subtask_model is None or self.subtask_processor is None:
            return
            
        # 系统提示词，定义三个子任务
        SYSTEM_PROMPT = "You are a robot action planner. Given images from the top and wrist cameras in a robotic workspace,output the subtask ID for the robot arm. Now there are three subtasks, Subtask 1:Grab the package and place it on the pallet. Subtask 2:Flip the package if the barcode is not facing up. Subtask 3:Grab the package and place it into the box. You must strictly output only the plan ID (1, 2 or 3) based on the current images. Do not output any other words or explanations."
        Required_confirmations = self.required_confirmations  # 需要连续确认的次数
        while self.running:
            start_t = time.perf_counter()
            try:
                # 获取俯视相机图像
                observation = self.robot.get_observation()
                image_top = None
                image_wrist = None
                for key, value in observation.items():
                    if isinstance(value, np.ndarray) and value.ndim == 3:
                        # 假设top相机包含'top'关键字
                        if 'top' in key.lower():
                            image_top = value
                        if 'wrist' in key.lower():
                            image_wrist = value

                
                # 构建对话消息
                messages = [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": SYSTEM_PROMPT}
                        ],
                    },
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": "top view:"},
                            {"type": "image", "image": image_top},
                            {"type": "text", "text": ", wrist view:"},
                            {"type": "image", "image": image_wrist},
                            {"type": "text", "text": ". Output the subtask ID (1, 2 or 3)."},
                        ],
                    }
                ]
                
                # 准备推理输入
                inputs = self.subtask_processor.apply_chat_template(
                    messages,
                    tokenize=True,
                    add_generation_prompt=True,
                    return_dict=True,
                    return_tensors="pt"
                )
                inputs = inputs.to(self.subtask_model.device)
                
                # 推理生成子任务
                with torch.inference_mode():
                    # generated_ids = self.subtask_model.generate(**inputs, max_new_tokens=128,do_sample=False)
                    generated_ids = self.subtask_model.generate(**inputs, max_new_tokens=128)

                
                # 解析输出
                generated_ids_trimmed = [
                    out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
                ]
                output_text = self.subtask_processor.batch_decode(
                    generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
                )
                
                # 更新当前子任务(线程安全)
                task_id = output_text[0]
                # print(task_id)
                if task_id in ["1","2","3"]:
                    new_subtask =  task_dict[output_text[0]]
                else:
                    new_subtask = self.current_subtask
                # print(new_subtask)
                with self.subtask_lock:
                    # 如果预测任务和当前任务一样，清空候选
                    if new_subtask == self.current_subtask:
                        self.candidate_subtask = None
                        self.candidate_count = 0

                    else:
                        # 如果和候选任务一致
                        if new_subtask == self.candidate_subtask:
                            self.candidate_count += 1
                            self.SUBTASK_PLANNING_PERIOD = 0.5
                        else:
                            # 新候选任务
                            self.candidate_subtask = new_subtask
                            self.candidate_count = 1
                            self.SUBTASK_PLANNING_PERIOD = 0.5

                        print(
                            f"候选子任务: {self.candidate_subtask}, "
                            f"连续次数: {self.candidate_count}"
                        )
                        
                        if (self.candidate_subtask == task_dict["2"] and self.current_subtask == task_dict["3"]):
                            print("检测到从子任务3切换到子任务2,增加确认次数翻倍")
                            Required_confirmations = self.required_confirmations * 2  # 对于3->2的切换需要更多确认次数
                        else:
                            Required_confirmations = self.required_confirmations
                        # 达到确认次数才切换
                        if self.candidate_count >= Required_confirmations:
                            print(
                                f"子任务确认切换: "
                                f"{self.current_subtask} -> {self.candidate_subtask}"
                            )
                            # if (self.candidate_subtask == task_dict["2"] and self.current_subtask == task_dict["1"]):
                            #     time.sleep(1.0)
                            #     print("sleep 1 s")
                            self.current_subtask = self.candidate_subtask
                            Required_confirmations = self.required_confirmations # 切换后重置确认次数要求
                            # self.current_subtask = task_dict["2"]

                            # 重置候选
                            self.candidate_subtask = None
                            self.candidate_count = 0
                            self.SUBTASK_PLANNING_PERIOD = self.cfg.subtask_planning_period

                        
                        # 子任务变化时重置策略
                        self.policy.reset()
                
                # 控制规划频率
                dt = time.perf_counter() - start_t
                # print(f"子任务规划完成. 耗时: {dt:.2f}s")
                wait_time = self.SUBTASK_PLANNING_PERIOD - dt
                if wait_time > 0:
                    busy_wait(wait_time)
                    
            except Exception as e:
                print(f"子任务规划线程异常: {e}")
                time.sleep(0.1)

    def start(self):
        if self.running:
            return
        self.running = True

        self.init_robot()

        self.infer_collect_thread = Thread(target=self._collect_and_infer, daemon=True)
        self.execute_thread = Thread(target=self._execute_action, daemon=True)
        # 只有当子任务规划模型可用时才创建该线程
        if self.subtask_model is not None:
            self.subtask_planning_thread = threading.Thread(target=self._plan_subtask, daemon=True)

        self.infer_collect_thread.start()
        self.execute_thread.start()
        if self.subtask_planning_thread is not None:
            self.subtask_planning_thread.start()

        print("所有线程已启动，开始机器人控制循环...")

    def stop(self):
        self.running = False

        if self.infer_collect_thread:
            self.infer_collect_thread.join(timeout=1.0)
        if self.execute_thread:
            self.execute_thread.join(timeout=1.0)
        print("所有线程已停止")

        if self.robot:
            self.robot.disconnect()


@parser.wrap()
def inference(cfg: InferenceConfig):
    inference_system = RobotInferenceSystem(cfg)
    try:
        inference_system.start()
        start_time = time.time()
        while time.time() - start_time < cfg.duration:
            if not inference_system.running:
                break
            time.sleep(1)
    except KeyboardInterrupt:
        print("\n收到停止信号，正在关闭系统...")
    finally:
        inference_system.stop()
        print("推理系统已正常关闭")


if __name__ == "__main__":
    inference()

