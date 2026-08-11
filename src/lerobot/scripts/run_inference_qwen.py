import time
from pathlib import Path
from typing import Any
import threading

from lerobot.configs import parser
from lerobot.utils.import_utils import register_third_party_plugins
import logging
from dataclasses import asdict, dataclass, field
from pprint import pformat

from lerobot.cameras import (  # noqa: F401
    CameraConfig,  # noqa: F401
)
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.cameras.reachy2_camera.configuration_reachy2_camera import Reachy2CameraConfig  # noqa: F401
from lerobot.cameras.realsense.configuration_realsense import RealSenseCameraConfig  # noqa: F401
from lerobot.cameras.zmq.configuration_zmq import ZMQCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.configs.policies import PreTrainedConfig
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.pipeline_features import aggregate_pipeline_dataset_features, create_initial_features
from lerobot.datasets.utils import combine_feature_dicts
from lerobot.datasets.video_utils import VideoEncodingManager
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.processor import make_default_processors
from lerobot.processor.rename_processor import rename_stats
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    piper_follower,
)
from lerobot.scripts.recording_hil import (
    ACPInferenceConfig,
    _capture_policy_runtime_state,  # noqa: F401
    _predict_policy_action_with_acp_inference,  # noqa: F401
)
from lerobot.utils.constants import ACTION
from lerobot.utils.utils import (
    init_logging,
)
from collections.abc import Callable
from typing import Any, TypeVar
from lerobot.datasets.image_writer import safe_stop_image_writer
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import build_dataset_frame
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.policies.utils import make_robot_action
from lerobot.processor import (
    PolicyAction,
    PolicyProcessorPipeline,
    RobotAction,
    RobotObservation,
    RobotProcessorPipeline,
)
from lerobot.robots import Robot
from lerobot.scripts.recording_hil import (
    INTERVENTION_STATE_POLICY,
    INTERVENTION_STATE_RELEASE,
    ACPInferenceConfig,
    _capture_policy_runtime_state,
    _predict_policy_action_with_acp_inference,
)
from lerobot.utils.constants import ACTION, OBS_STR
from lerobot.utils.robot_utils import precise_sleep
from lerobot.utils.utils import get_safe_torch_device
from concurrent.futures import ThreadPoolExecutor

# 导入必要的库
import numpy as np  # 数值计算
import torch  # PyTorch深度学习框架
import time  # 时间操作
from collections import deque  # 双端队列，用于存储历史动作
import requests  # HTTP请求库
import threading  # 多线程支持
from PIL import Image  # 图像处理
from transformers import AutoProcessor, AutoModelForImageTextToText  # HuggingFace transformers



task_dict = {
    "1": "Grab the package and place it on the pallet.",
    "2": "Flip the package if the barcode is not facing up.",
    "3": "Grab the package and place it into the box.",
}


class ArmExecutor:

    def __init__(self, robot: Robot, parallel_dispatch: bool = True):
        self.robot = robot
        self.parallel_dispatch = parallel_dispatch
        self._pool = ThreadPoolExecutor(max_workers=2) if parallel_dispatch else None

    def send_action(self, action: RobotAction) -> RobotAction:
        if self._pool is None:
            sent_action = self.robot.send_action(action)
            return sent_action

        robot_future = self._pool.submit(self.robot.send_action, action)
        sent_action = robot_future.result()
        return sent_action

    def shutdown(self) -> None:
        if self._pool is not None:
            self._pool.shutdown(wait=True)


@dataclass
class DatasetRecordConfig:
    repo_id: str
    root: str | Path | None = None
    episode_time_s: int | float = 60
    reset_time_s: int | float = 60
    num_episodes: int = 50
    video: bool = True
    private: bool = False
    tags: list[str] | None = None
    num_image_writer_processes: int = 0
    num_image_writer_threads_per_camera: int = 4
    video_encoding_batch_size: int = 1
    vcodec: str = "libsvtav1"
    rename_map: dict[str, str] = field(default_factory=dict)


@dataclass
class InferenceConfig:
    robot: RobotConfig
    dataset: DatasetRecordConfig
    single_task: str
    policy: PreTrainedConfig | None = None
    acp_inference: ACPInferenceConfig = field(default_factory=ACPInferenceConfig)
    communication_retry_timeout_s: float = 2.0
    communication_retry_interval_s: float = 0.1
    ckpt_path: str = None  # 模型检查点路径
    task: str = None  # 任务名称
    qwen3_vl_path: str = None  # 子任务规划模型路径
    subtask_planning_period: float = 1.0  # 子任务规划周期(秒)

    def __post_init__(self):
        policy_path = parser.get_path_arg("policy")

        if policy_path:
            cli_overrides = parser.get_cli_overrides("policy")

            self.policy = PreTrainedConfig.from_pretrained(policy_path, cli_overrides=cli_overrides)
            self.policy.pretrained_path = policy_path

        if self.acp_inference.use_cfg and not self.acp_inference.enable:
            raise ValueError("`acp_inference.use_cfg=true` requires `acp_inference.enable=true`.")
        if self.acp_inference.cfg_beta < 0:
            raise ValueError("`acp_inference.cfg_beta` must be >= 0.")
        if self.communication_retry_timeout_s < 0:
            raise ValueError("`communication_retry_timeout_s` must be >= 0.")
        if self.communication_retry_interval_s <= 0:
            raise ValueError("`communication_retry_interval_s` must be > 0.")

    @classmethod
    def __get_path_fields__(cls) -> list[str]:
        """This enables the parser to load config from the policy using `--policy.path=local/dir`"""
        return ["policy"]


T = TypeVar("T")




# @parser.wrap()
# def inference(cfg: InferenceConfig):
#     init_logging(log_file="inloop_record.log", file_level="INFO")
#     logging.info(pformat(asdict(cfg)))

#     robot = make_robot_from_config(cfg.robot)
#     teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

#     dataset_features = combine_feature_dicts(
#         aggregate_pipeline_dataset_features(
#             pipeline=teleop_action_processor,
#             initial_features=create_initial_features(
#                 action=robot.action_features
#             ),  # TODO(steven, pepijn): in future this should be come from teleop or policy
#             use_videos=cfg.dataset.video,
#         ),
#         aggregate_pipeline_dataset_features(
#             pipeline=robot_observation_processor,
#             initial_features=create_initial_features(observation=robot.observation_features),
#             use_videos=cfg.dataset.video,
#         ),
#     )
#     logging.info(f"dataset_features: {dataset_features}")

#     dataset = None
#     policy_sync_executor = None

#     try:
#         dataset = LeRobotDataset.create(
#             cfg.dataset.repo_id,
#             30,
#             root=cfg.dataset.root,
#             robot_type=robot.name,
#             features=dataset_features,
#             use_videos=cfg.dataset.video,
#             image_writer_processes=cfg.dataset.num_image_writer_processes,
#             image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
#             batch_encoding_size=cfg.dataset.video_encoding_batch_size,
#             vcodec=cfg.dataset.vcodec,
#         )

#         policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)
#         preprocessor = None
#         postprocessor = None
#         if cfg.acp_inference.enable and cfg.policy is None:
#             raise ValueError("`acp_inference.enable=true` requires `policy` to be set.")
#         if cfg.policy is not None:
#             preprocessor, postprocessor = make_pre_post_processors(
#                 policy_cfg=cfg.policy,
#                 pretrained_path=cfg.policy.pretrained_path,
#                 dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
#                 preprocessor_overrides={
#                     "device_processor": {"device": cfg.policy.device},
#                     "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
#                 },
#             )

#         robot.connect()    
#         policy_sync_executor = ArmExecutor(robot=robot)

#         inference_loop(
#             robot=robot,
#             robot_action_processor=robot_action_processor,
#             robot_observation_processor=robot_observation_processor,
#             policy=policy,
#             preprocessor=preprocessor,
#             postprocessor=postprocessor,
#             dataset_features=dataset_features,
#             single_task=cfg.single_task,
#             policy_sync_executor=policy_sync_executor,
#             acp_inference=cfg.acp_inference,
#             communication_retry_timeout_s=cfg.communication_retry_timeout_s,
#             communication_retry_interval_s=cfg.communication_retry_interval_s,
#         )
#     finally:
#         if policy_sync_executor is not None:
#             policy_sync_executor.shutdown()

#         if robot.is_connected:
#             robot.disconnect()



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
        """初始化推理系统

        参数:
            cfg: 推理配置对象
        """
        self.cfg = cfg
        self.robot = None  # 机器人对象
        self.policy = None  # 策略模型
        self.preprocess = None  # 预处理函数
        self.postprocess = None  # 后处理函数
        self.device = "cuda"  # 运行设备

        self.running = False  # 线程运行标志
        self.infer_collect_thread = None  # 采集+推理线程
        self.execute_thread = None  # 动作执行线程
        self.subtask_planning_thread = None  # 子任务规划线程

        self.SMOOTH_WINDOW = 9  # 平滑窗口大小
        self.INFERENCE_PERIOD = 1.5  # 采集+推理周期(秒)
        self.EXECUTION_PERIOD = 0.05  # 执行周期(秒)，约20Hz
        self.SUBTASK_PLANNING_PERIOD = cfg.subtask_planning_period  # 子任务规划周期(秒)

        # self.smoother = ArmSmoother(self.SMOOTH_WINDOW)  # 动作平滑器

        # 动作数据缩放因子(将归一化动作转换为实际电机指令)
        self.pos_factor = 1000000  # 位置缩放因子
        self.angel_factor = 1000  # 角度缩放因子(0.001度)
        self.grapper_factor = 70000  # 夹爪缩放因子

        # 子任务规划相关
        self.subtask_model = None  # 子任务规划模型
        self.subtask_processor = None  # 子任务规划处理器
        self.current_subtask = None  # 当前子任务
        self.subtask_lock = threading.Lock()  # 子任务线程锁
        
        # 子任务稳定检测
        self.candidate_subtask = None
        self.candidate_count = 0
        self.required_confirmations = 3  # 需要连续3次确认

    def init_robot(self):
        """初始化机器人并完成使能

        流程:
            1. 创建机器人实例并连接
            2. 初始化策略模型
            3. 机器人使能
            4. 等待使能完成
        """
        # # 创建机器人实例并连接
        # self.robot = make_robot_from_config(self.cfg.robot)
        # if not self.robot.is_connected:
        #     self.robot.connect()

        # # 初始化策略模型
        # self._init_policy()

        # # 机器人使能
        # self._enable_robot()
        # self.robot.piper.MotorAngleLimitMaxSpdSet(1, 1500, -420)   #
        # print(self.robot.piper.GetAllMotorAngleLimitMaxSpd())
        # # 等待使能完成
        # time.sleep(2.0)
        cfg = self.cfg
        init_logging(log_file="inloop_record.log", file_level="INFO")
        logging.info(pformat(asdict(cfg)))

        robot = make_robot_from_config(cfg.robot)
        teleop_action_processor, robot_action_processor, robot_observation_processor = make_default_processors()

        dataset_features = combine_feature_dicts(
            aggregate_pipeline_dataset_features(
                pipeline=teleop_action_processor,
                initial_features=create_initial_features(
                    action=robot.action_features
                ),  # TODO(steven, pepijn): in future this should be come from teleop or policy
                use_videos=cfg.dataset.video,
            ),
            aggregate_pipeline_dataset_features(
                pipeline=robot_observation_processor,
                initial_features=create_initial_features(observation=robot.observation_features),
                use_videos=cfg.dataset.video,
            ),
        )
        logging.info(f"dataset_features: {dataset_features}")

        dataset = None
        policy_sync_executor = None
        # 初始化子任务规划模型
        self._init_subtask_planning_model()

        dataset = LeRobotDataset.create(
            cfg.dataset.repo_id,
            30,
            root=cfg.dataset.root,
            robot_type=robot.name,
            features=dataset_features,
            use_videos=cfg.dataset.video,
            image_writer_processes=cfg.dataset.num_image_writer_processes,
            image_writer_threads=cfg.dataset.num_image_writer_threads_per_camera * len(robot.cameras),
            batch_encoding_size=cfg.dataset.video_encoding_batch_size,
            vcodec=cfg.dataset.vcodec,
        )

        policy = None if cfg.policy is None else make_policy(cfg.policy, ds_meta=dataset.meta)
        preprocessor = None
        postprocessor = None
        if cfg.acp_inference.enable and cfg.policy is None:
            raise ValueError("`acp_inference.enable=true` requires `policy` to be set.")
        if cfg.policy is not None:
            preprocessor, postprocessor = make_pre_post_processors(
                policy_cfg=cfg.policy,
                pretrained_path=cfg.policy.pretrained_path,
                dataset_stats=rename_stats(dataset.meta.stats, cfg.dataset.rename_map),
                preprocessor_overrides={
                    "device_processor": {"device": cfg.policy.device},
                    "rename_observations_processor": {"rename_map": cfg.dataset.rename_map},
                },
            )

        robot.connect()    
        policy_sync_executor = ArmExecutor(robot=robot)

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
        robot_type = self.cfg.robot.type
        if robot_type == "single_piper":
            # 使能机械臂
            self.robot.piper.EnableArm(7)
            # 等待并检测使能状态
            enable_fun(piper=self.robot.piper)
            # 设置运动控制参数
            self.robot.piper.MotionCtrl_2(0x01, 0x00, 10, 0x00)
            # 设置夹爪初始开合程度
            self.robot.piper.GripperCtrl(round(1.0 * 70 * 1000), 2000, 0x01, 0)
        else:
            raise ValueError(f"Unsupported robot type: {robot_type}")

    def grapper_attenuation(self, raw_value, full_open, mid=0.35, steepness=4):
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
    
    @safe_stop_image_writer
    def inference_loop(
        robot: Robot,
        robot_action_processor: RobotProcessorPipeline[
            tuple[RobotAction, RobotObservation], RobotAction
        ],  # runs before robot
        robot_observation_processor: RobotProcessorPipeline[
            RobotObservation, RobotObservation
        ],  # runs after robot
        dataset_features: dict,
        policy_sync_executor: ArmExecutor,
        policy: PreTrainedPolicy | None = None,
        preprocessor: PolicyProcessorPipeline[dict[str, Any], dict[str, Any]] | None = None,
        postprocessor: PolicyProcessorPipeline[PolicyAction, PolicyAction] | None = None,
        single_task: str | None = None,
        acp_inference: ACPInferenceConfig | None = None,
        communication_retry_timeout_s: float = 2.0,
        communication_retry_interval_s: float = 0.1,
    ):
        if acp_inference is None:
            acp_inference = ACPInferenceConfig()

        action_feature_names = dataset_features[ACTION]["names"]
        if action_feature_names is None:
            if hasattr(robot.action_features, "keys"):
                action_feature_names = list(robot.action_features.keys())
            else:
                action_feature_names = list(robot.action_features)
        intervention_state = INTERVENTION_STATE_POLICY

        if policy is not None and preprocessor is not None and postprocessor is not None:
            policy.reset()
            preprocessor.reset()
            postprocessor.reset()

        cond_policy_runtime_state: dict[str, Any] | None = None
        uncond_policy_runtime_state: dict[str, Any] | None = None
        if policy is not None and acp_inference.enable and acp_inference.use_cfg:
            cond_policy_runtime_state = _capture_policy_runtime_state(policy)
            uncond_policy_runtime_state = _capture_policy_runtime_state(policy)

        def run_with_connection_retry(action_name: str, fn: Callable[[], T]) -> T:
            timeout_s = max(communication_retry_timeout_s, 0.0)
            interval_s = max(communication_retry_interval_s, 0.0)
            deadline_t = time.perf_counter() + timeout_s
            attempts = 0
            first_error: ConnectionError | None = None

            while True:
                attempts += 1
                try:
                    result = fn()
                    if attempts > 1:
                        elapsed_s = timeout_s - max(deadline_t - time.perf_counter(), 0.0)
                        logging.warning(
                            "%s recovered after %d retries in %.2fs.",
                            action_name,
                            attempts - 1,
                            elapsed_s,
                        )
                    return result
                except ConnectionError as error:
                    if first_error is None:
                        first_error = error
                        logging.warning(
                            "%s failed with transient communication error; retrying for up to %.2fs (%s)",
                            action_name,
                            timeout_s,
                            error,
                        )

                    if timeout_s <= 0.0:
                        raise

                    remaining_s = deadline_t - time.perf_counter()
                    if remaining_s <= 0.0:
                        raise

                    sleep_s = interval_s if interval_s > 0.0 else remaining_s
                    time.sleep(min(sleep_s, remaining_s))

        timespent = 0
        start_episode_t = time.perf_counter()
        run_count = 0
        while run_count <  12000:  # 10min
            run_count = run_count + 1
            start_loop_t = time.perf_counter()

            obs = robot.get_observation()
            obs_processed = robot_observation_processor(obs)
            observation_frame = build_dataset_frame(dataset_features, obs_processed, prefix=OBS_STR)
            act_processed_policy: RobotAction | None = None
            # 获取当前子任务(线程安全)
            with self.subtask_lock:
                current_task = self.current_subtask
            policy_action = _predict_policy_action_with_acp_inference(
                observation_frame=observation_frame,
                policy=policy,
                device=get_safe_torch_device(policy.config.device),
                preprocessor=preprocessor,
                postprocessor=postprocessor,
                use_amp=policy.config.use_amp,
                task=current_task,
                robot_type=robot.robot_type,
                acp_inference=acp_inference,
                cond_runtime_state=cond_policy_runtime_state,
                uncond_runtime_state=uncond_policy_runtime_state,
            )
        # return policy_action
            act_processed_policy = make_robot_action(policy_action, dataset_features)

            robot_action_to_send = robot_action_processor((act_processed_policy, obs))
            run_with_connection_retry(
                "policy_sync_executor.send_action",
                lambda robot_action_to_send=robot_action_to_send: policy_sync_executor.send_action(
                    robot_action_to_send
                ),
            )
            
            if intervention_state == INTERVENTION_STATE_RELEASE:
                intervention_state = INTERVENTION_STATE_POLICY

            # dt_s = time.perf_counter() - start_loop_t
            # precise_sleep(max(1 / 15 - dt_s, 0.0))  # 控制频率

            # timespent = time.perf_counter() - start_episode_t
    def _collect_and_infer(self):
        """采集+推理线程函数

        循环执行:
            1. 获取机器人观测(相机图像+关节状态)
            2. 策略推理得到动作
            3. 动作平滑处理
            4. 放入动作队列
            5. 控制推理频率
        """
        while self.running:
            start_t = time.perf_counter()
            try:
                inference_loop(
                    robot=robot,
                    robot_action_processor=robot_action_processor,
                    robot_observation_processor=robot_observation_processor,
                    policy=policy,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                    dataset_features=dataset_features,
                    single_task=single_task,
                    policy_sync_executor=policy_sync_executor,
                    acp_inference=cfg.acp_inference,
                    communication_retry_timeout_s=cfg.communication_retry_timeout_s,
                    communication_retry_interval_s=cfg.communication_retry_interval_s,
                )
                # # 动作平滑处理
                # smoothed_batch = self.smoother.smooth(smoothed_batch)

                # # 合并历史动作与新动作
                # with self.position_queue_lock:
                #     executed_steps = action_queue_len_pre - len(self.position_queue)  # 已执行步数
                #     smoothed_batch = smoothed_batch[executed_steps:]  # 跳过已执行的
                #     smoothed_batch = self.smoother.merge_with_queue(smoothed_batch, self.position_queue)

                # # 放入动作队列
                # with self.position_queue_lock:
                #     self.position_queue.clear()  # 清空旧队列
                #     smoothed_batch = smoothed_batch.astype(np.int64)
                #     for pos in smoothed_batch:
                #         self.position_queue.append(pos)

                # 控制采集+推理频率
                dt = time.perf_counter() - start_t
                print(f"inference over. cost time:{dt} s.")
                wait_time = self.INFERENCE_PERIOD - dt
                if wait_time > 0:
                    busy_wait(wait_time)

            except Exception as e:
                print(f"采集+推理线程异常: {e}")
                time.sleep(0.001)

    # def _execute_action(self):
    #     """动作执行线程函数(独立线程，保证控制实时性)

    #     循环执行:
    #         1. 从动作队列获取动作(阻塞等待)
    #         2. 发送给机器人执行
    #         3. 控制执行频率
    #     """
    #     last_queue_len = 1000  # 上次队列长度
    #     is_new_batch = False  # 是否有新批次
    #     while self.running:
    #         start_t = time.perf_counter()
    #         position = None
    #         try:
    #             # # 阻塞等待获取动作数据
    #             # while self.running:
    #             #     with self.position_queue_lock:
    #             #         if self.position_queue:
    #             #             is_new_batch = False
    #             #             if last_queue_len < len(self.position_queue):
    #             #                 is_new_batch = True  # 检测到新批次
    #             #             last_queue_len = len(self.position_queue)
    #             #             position = self.position_queue.popleft()  # 从队列取出动作
    #             #             break
    #             #     time.sleep(0.001)
    #             # if not self.running:
    #             #     break

    #             # # 打印动作信息
    #             # # print(f"{'new ' if is_new_batch else ''}{position}")
    #             # # 执行动作
    #             # self._execute_robot_action(position)

    #             # 控制执行频率
    #             dt = time.perf_counter() - start_t
    #             wait_time = self.EXECUTION_PERIOD - dt
    #             if wait_time > 0:
    #                 busy_wait(wait_time)

    #         except Exception as e:
    #             print(f"动作执行线程异常: {e}")
    #             time.sleep(0.001)

    # def _execute_robot_action(self, position):
    #     """根据机器人类型执行具体的动作控制

    #     参数:
    #         position: 动作位置数据 (7,) - [x, y, z, roll, pitch, yaw, gripper]
    #     """
    #     robot_type = self.cfg.robot.type
    #     if robot_type == "single_piper":
    #         # 控制末端位姿(x, y, z, roll, pitch, yaw)
    #         self.robot.piper.EndPoseCtrl(*(int(x) for x in position[:6]))
    #         # 控制夹爪开合(应用非线性衰减)
    #         gripper = self.grapper_attenuation(position[6], self.grapper_factor)
    #         self.robot.piper.GripperCtrl(gripper, 1000, 0x01, 0)
    #     else:
    #         raise ValueError(f"Unsupported robot type for action execution: {robot_type}")

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
                print(task_id)
                if task_id in ["1","2","3"]:
                    new_subtask =  task_dict[output_text[0]]
                else:
                    new_subtask = self.current_subtask
                print(new_subtask)
                with self.subtask_lock:
                    # 如果预测任务和当前任务一样，清空候选
                    if new_subtask == self.current_subtask:
                        self.candidate_subtask = None
                        self.candidate_count = 0

                    else:
                        # 如果和候选任务一致
                        if new_subtask == self.candidate_subtask:
                            self.candidate_count += 1
                        else:
                            # 新候选任务
                            self.candidate_subtask = new_subtask
                            self.candidate_count = 1

                        print(
                            f"候选子任务: {self.candidate_subtask}, "
                            f"连续次数: {self.candidate_count}"
                        )

                        # 达到确认次数才切换
                        if self.candidate_count >= self.required_confirmations:
                            print(
                                f"子任务确认切换: "
                                f"{self.current_subtask} -> {self.candidate_subtask}"
                            )

                            self.current_subtask = self.candidate_subtask

                            # 重置候选
                            self.candidate_subtask = None
                            self.candidate_count = 0
                        
                        # 子任务变化时重置策略
                        self.policy.reset()
                
                # 控制规划频率
                dt = time.perf_counter() - start_t
                print(f"子任务规划完成. 耗时: {dt:.2f}s")
                wait_time = self.SUBTASK_PLANNING_PERIOD - dt
                if wait_time > 0:
                    busy_wait(wait_time)
                    
            except Exception as e:
                print(f"子任务规划线程异常: {e}")
                time.sleep(0.1)

    def start(self):
        """启动推理系统

        流程:
            1. 初始化机器人
            2. 创建并启动三个工作线程(采集推理、执行动作、子任务规划)
            3. 进入主循环
        """
        if self.running:
            return
        self.running = True

        # 初始化机器人(连接、加载策略、使能)
        self.init_robot()

        # 创建工作线程
        self.infer_collect_thread = threading.Thread(target=self._collect_and_infer, daemon=True)
        # self.execute_thread = threading.Thread(target=self._execute_action, daemon=True)
        
        # 只有当子任务规划模型可用时才创建该线程
        if self.subtask_model is not None:
            self.subtask_planning_thread = threading.Thread(target=self._plan_subtask, daemon=True)

        # 启动线程
        self.infer_collect_thread.start()
        # self.execute_thread.start()
        if self.subtask_planning_thread is not None:
            self.subtask_planning_thread.start()

        print("所有线程已启动，开始机器人控制循环...")

    def stop(self):
        """停止所有线程并清理资源

        流程:
            1. 设置停止标志
            2. 等待线程结束
        """
        self.running = False

        # 等待线程结束
        if self.infer_collect_thread:
            self.infer_collect_thread.join(timeout=1.0)
        # if self.execute_thread:
        #     self.execute_thread.join(timeout=1.0)
        if self.subtask_planning_thread:
            self.subtask_planning_thread.join(timeout=1.0)
        print("所有线程已停止")

@parser.wrap()
def inference(cfg: InferenceConfig) -> None:
    """推理入口函数"""
    # 创建推理系统实例
    inference_system = RobotInferenceSystem(cfg)
    
    try:
        # 启动线程化推理
        inference_system.start()
        
        # 主进程空闲运行
        inference_time_s = 3600  # 推理总时间, 1小时
        start_time = time.time()
        while time.time() - start_time < inference_time_s:
            if not inference_system.running:
                break
            time.sleep(1.0)
        
    except KeyboardInterrupt:
        print("\n接收到停止信号，正在关闭系统...")
    finally:
        # 停止所有线程
        inference_system.stop()
        print("推理系统已正常关闭")

if __name__ == "__main__":
    register_third_party_plugins()
    inference()
