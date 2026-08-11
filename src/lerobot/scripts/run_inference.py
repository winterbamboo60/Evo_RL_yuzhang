import time
from pathlib import Path
from typing import Any

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
        policy_action = _predict_policy_action_with_acp_inference(
            observation_frame=observation_frame,
            policy=policy,
            device=get_safe_torch_device(policy.config.device),
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            use_amp=policy.config.use_amp,
            task=single_task,
            robot_type=robot.robot_type,
            acp_inference=acp_inference,
            cond_runtime_state=cond_policy_runtime_state,
            uncond_runtime_state=uncond_policy_runtime_state,
        )
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

        dt_s = time.perf_counter() - start_loop_t
        precise_sleep(max(1 / 15 - dt_s, 0.0))  # 控制频率

        timespent = time.perf_counter() - start_episode_t


@parser.wrap()
def inference(cfg: InferenceConfig):
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

    try:
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

        inference_loop(
            robot=robot,
            robot_action_processor=robot_action_processor,
            robot_observation_processor=robot_observation_processor,
            policy=policy,
            preprocessor=preprocessor,
            postprocessor=postprocessor,
            dataset_features=dataset_features,
            single_task=cfg.single_task,
            policy_sync_executor=policy_sync_executor,
            acp_inference=cfg.acp_inference,
            communication_retry_timeout_s=cfg.communication_retry_timeout_s,
            communication_retry_interval_s=cfg.communication_retry_interval_s,
        )
    finally:
        if policy_sync_executor is not None:
            policy_sync_executor.shutdown()

        if robot.is_connected:
            robot.disconnect()


if __name__ == "__main__":
    register_third_party_plugins()
    inference()
