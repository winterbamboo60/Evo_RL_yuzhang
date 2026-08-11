# Evo-RL-0707版本修改说明

## 总体变化
- 新增价值模型、策略模型训练脚本中“梯度累计训练”功能；
- 新增价值推理过程中三种可视化效果：
  - target_value + pred_value 在视频中逐帧变化曲线（判断价值模型训练效果）；
  - advantage 在视频中逐帧变化曲线；
  - indicator_advantage 在视频中逐帧变化曲线（即策略模型训练阶段实际使用的二值化优劣帧）；
- 新增SFT训练代码（改造./src/lerobot/scripts/lerobot_train.py）；
- VLA+人工数据录制过程中新增如下功能:
  - 开始录制前，主从臂归位（初始位置）；
  - 添加归位按键"r"，允许录制过程中放弃当前episode并回到初始位置，然后重新开始录制当前episode；
  - 新增质量事件，允许自定义惩罚事件，添加帧级标记（代码中还未将这一信息利用，可忽略）；
  - 新增主从臂位姿保持能力，在按下"s/f/<-"后，两臂将保持当前位置，作为下一episode的起点；
  - 新增VLA动作平滑功能，避免VLA输出动作的抖动，提高VLA自主采集数据质量；

## 如何运行
### 环境配置


```
# 解压conda-pack包
mkdir -p ./envs/package_sorting_env && tar -xzf ./package_sorting.tar.gz -C ./envs/package_sorting_env

# 激活环境
source ./envs/package_sorting_env/bin/activate
./envs/package_sorting_env/bin/conda-unpack

# 重定向lerobot
pip install -e . --no-deps --no-build-isolation

# 检查环境，当'Editable project location'指向项目目录就说明成功
pip show lerobot
```


### 运行
随后参考下列脚本完成数据录制、模型训练：
- ./scripts/RL_init.sh：机械臂初始化、相机查找等脚本配置
- ./scripts/RL_data.sh：
  - 纯人工SFT数据录制；
  - VLA+人工介入的离线RL数据采集；
- ./scripts/RL_train.sh：
  - 价值模型训练；
  - 价值模型推理&可视化；
  - 策略模型SFT/RL训练。


## 详细改动说明
### 录制与人工接管

- `src/lerobot/scripts/lerobot_record.py`
  - 增加录制前机械臂归位，录制间机械臂归位，录制间隔时保持机械臂位置。
  - 增加 episode 成功/失败标注、超时处理、重录当前 episode、质量事件标注等录制控制。
  - 增加 `policy_action`、`collector_policy_id`、ACP 推理相关信息写入录制数据的能力。
- `src/lerobot/scripts/lerobot_human_inloop_record.py`
  - 增加超时后的复位/保持行为说明。
  - 增加按 `r` 放弃当前 episode、机械臂回到初始位姿并重录当前 episode 的逻辑。
- `src/lerobot/scripts/recording_loop.py`
  - 增加质量事件逐帧记录能力，例如 `bad_depth`、`bad_edge` 这类由热键触发的事件。
  - 增加 `episode_timeout`、`reset_episode`、`rerecord_episode` 等事件状态，区分按键结束和录制超时结束。
  - 接管状态下调整动作发送逻辑，人工接管动作不叠加模型动作偏置。
- `src/lerobot/scripts/recording_hil.py`
  - `send_action` 增加 `add_offset` 参数，并同步传给 robot 和 teleop feedback。但实际上未生效，留待后续使用
- `src/lerobot/utils/control_utils.py`
  - 键盘监听增加重录快捷键和质量事件快捷键支持。
- `src/lerobot/utils/recording_events.py`
  - 新增质量事件配置模块，支持从 JSON 定义事件名称、按键绑定、逐帧 feature 和 episode 级事件列表。

### 训练与 ACP

- `src/lerobot/scripts/lerobot_train.py`
  - 增加梯度累积训练：一个 optimizer step 可累积多个 micro-batch。
  - ACP 标签注入在非 SFT 训练时启用，SFT 训练时跳过 ACP raw batch hook。
  - 日志写入 `policy_train.log`，并补充 effective batch size 计算。
- `src/lerobot/configs/train.py`
  - 增加梯度累积训练：新增 `gradient_accumulation_steps` 配置，并校验必须大于等于 1。
- `src/lerobot/scripts/lerobot_value_train.py`
  - 价值模型训练增加梯度累积。
  - 增加 `value_target` 注入说明和训练日志 `value_train.log`。
  - 增加 DataLoader、WandB 等退出清理逻辑，避免训练结束后进程卡住。
- `src/lerobot/configs/value_train.py`
  - 新增价值训练的 `gradient_accumulation_steps` 配置。
- `src/lerobot/values/pistar06/modeling_pistar06.py`
  - 补充 value target 计算与 raw batch hook 注入逻辑说明。

### 价值推理与可视化

- `src/lerobot/scripts/lerobot_value_infer.py`
  - 推理后可把 value、advantage、indicator 写回数据集。
  - 增加 `indicator_only` 模式，只导出二值优势/劣势可视化。
  - 增加 advantage overlay 和 indicator overlay 视频导出。
- `src/lerobot/scripts/value_infer_viz.py`
  - 可视化视频支持同时画 `V_pred` 和 `V_target` 曲线。
  - 新增 soft advantage 曲线视频导出。
  - 新增二值 advantage/disadvantage 判定叠加视频，按帧显示优势或劣势状态。
  - 输出视频文件名加入 episode success/failure 标识。
- `src/lerobot/configs/value.py`
  - 新增 `viz.indicator_only` 配置。