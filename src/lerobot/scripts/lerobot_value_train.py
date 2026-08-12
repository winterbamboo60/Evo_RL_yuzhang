#!/usr/bin/env python

import logging
import os
import sys
import threading
import time
from contextlib import nullcontext
from pprint import pformat
from typing import Any, Iterator

import torch
from accelerate import Accelerator
from termcolor import colored
from torch.optim import Optimizer

from lerobot.configs import parser
from lerobot.configs.value_train import ValueTrainPipelineConfig
from lerobot.datasets.factory import make_dataset
from lerobot.datasets.utils import cycle
from lerobot.optim.factory import make_optimizer_and_scheduler
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pretrained import PreTrainedPolicy
from lerobot.rl.wandb_utils import make_logger
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.logging_utils import AverageMeter, MetricsTracker
from lerobot.utils.random_utils import set_seed
from lerobot.utils.train_utils import (
    get_step_checkpoint_dir,
    load_training_state,
    save_checkpoint,
    update_last_checkpoint,
)
from lerobot.utils.utils import format_big_number, has_method, init_logging


def _run_cleanup_with_timeout(name: str, fn, timeout_s: float) -> bool:
    """Run best-effort shutdown code without letting it hang process exit."""
    result: dict[str, Any] = {"exc": None}

    def target() -> None:
        try:
            fn()
        except Exception as exc:  # pragma: no cover - shutdown best effort
            result["exc"] = exc

    logging.info("Starting %s cleanup", name)
    start_time = time.perf_counter()
    thread = threading.Thread(target=target, name=f"{name}-cleanup", daemon=True)
    thread.start()
    thread.join(timeout=timeout_s)

    elapsed_s = time.perf_counter() - start_time
    if thread.is_alive():
        logging.warning("%s cleanup did not finish within %.1fs; continuing exit.", name, timeout_s)
        return False
    if result["exc"] is not None:
        logging.debug("%s cleanup failed.", name, exc_info=result["exc"])
        return False

    logging.info("Finished %s cleanup in %.1fs", name, elapsed_s)
    return True


def _shutdown_dataloader_iterator(dl_iter: Iterator[Any] | None) -> None:
    """Close the active PyTorch DataLoader iterator held by `cycle(dataloader)`."""
    if dl_iter is None:
        return

    candidates: list[Any] = [dl_iter]
    frame = getattr(dl_iter, "gi_frame", None)
    if frame is not None:
        iterator = frame.f_locals.get("iterator")
        if iterator is not None:
            candidates.append(iterator)

    for iterator in candidates:
        shutdown_workers = getattr(iterator, "_shutdown_workers", None)
        if callable(shutdown_workers):
            try:
                shutdown_workers()
            except Exception:
                logging.debug("Failed to shutdown DataLoader workers.", exc_info=True)

    close = getattr(dl_iter, "close", None)
    if callable(close):
        try:
            close()
        except Exception:
            logging.debug("Failed to close DataLoader cycle iterator.", exc_info=True)


def _finish_training_logger(wandb_logger: Any | None) -> None:
    if wandb_logger is None:
        return

    logger_finish = getattr(wandb_logger, "finish", None)
    if callable(logger_finish):
        try:
            logger_finish()
        except Exception:
            logging.debug("Failed to finish experiment logger.", exc_info=True)
        return

    wandb = getattr(wandb_logger, "_wandb", None)
    if wandb is not None:
        try:
            wandb.finish()
        except Exception:
            logging.debug("Failed to finish WandB logger.", exc_info=True)

    run = getattr(wandb_logger, "_run", None)
    finish = getattr(run, "finish", None)
    if callable(finish):
        try:
            finish()
        except Exception:
            logging.debug("Failed to finish SwanLab logger.", exc_info=True)


def update_policy(
    train_metrics: MetricsTracker,
    policy: PreTrainedPolicy,
    dl_iter: Any,
    target_hook,
    preprocessor,
    optimizer: Optimizer,
    grad_clip_norm: float,
    accelerator: Accelerator,
    grad_accum_steps: int,
    step: int,
    lr_scheduler=None,
    lock=None,
) -> tuple[MetricsTracker, dict, Any]:
    """执行一次完整的优化更新。

    一个 step = 一次 optimizer 更新，覆盖 grad_accum_steps 个 micro-batch
    （即 batch_size * gradient_accumulation_steps * num_processes 个样本）。
    价值模型是 [Image Encoder, Language Encoder] + [Value Head] 架构：输入视觉 + 语言
    （含状态与 success 信号），输出标量 value 预测。
    """
    start_time = time.perf_counter()
    policy.train()

    optimizer.zero_grad()
    accum_loss = 0.0
    output_dict: dict = {}
    last_batch: Any = None
    dataloading_s = 0.0

    for micro_step in range(grad_accum_steps):
        # 取一个 micro-batch 并构建标签 / 预处理（计入数据加载耗时）。
        t0 = time.perf_counter()
        batch = next(dl_iter)
        if target_hook is not None:
            # 查表注入每帧 value_target 标签。
            batch = target_hook(batch, step)
        batch = preprocessor(batch)
        dataloading_s += time.perf_counter() - t0
        last_batch = batch

        # 多卡时，仅在最后一个 micro-step 做梯度 all-reduce，其余跳过以省通信。
        is_last_micro = micro_step == grad_accum_steps - 1
        sync_ctx = nullcontext() if is_last_micro else accelerator.no_sync(policy)
        with sync_ctx:
            with accelerator.autocast():
                loss, output_dict = policy.forward(batch)
            # accelerator.backward 内部会除以 gradient_accumulation_steps，
            # 因此逐 micro-batch 调用即可累加出整组的“平均”梯度。
            accelerator.backward(loss)
        accum_loss += loss.item()

    if grad_clip_norm > 0:
        # 在累加完成后、step 之前做一次梯度裁剪。
        grad_norm = accelerator.clip_grad_norm_(policy.parameters(), grad_clip_norm)
    else:
        grad_norm = torch.nn.utils.clip_grad_norm_(
            policy.parameters(), float("inf"), error_if_nonfinite=False
        )

    with lock if lock is not None else nullcontext():
        optimizer.step()

    optimizer.zero_grad()

    if lr_scheduler is not None:
        lr_scheduler.step()

    if has_method(accelerator.unwrap_model(policy, keep_fp32_wrapper=True), "update"):
        accelerator.unwrap_model(policy, keep_fp32_wrapper=True).update()

    train_metrics.loss = accum_loss / grad_accum_steps
    train_metrics.grad_norm = grad_norm.item()
    train_metrics.lr = optimizer.param_groups[0]["lr"]
    train_metrics.update_s = time.perf_counter() - start_time
    train_metrics.dataloading_s = dataloading_s
    return train_metrics, output_dict, last_batch


# parser.wrap() 来自配置解析库，自动解析命令行参数，将参数填充到 cfg:ValueTrainPipelineConfig 对象中，并传递给函数。
# 这样直接运行脚本时，就能从命令行读取配置，而不需要手动调用 argparse
@parser.wrap()
def value_train(
    cfg: ValueTrainPipelineConfig,
    accelerator: Accelerator | None = None,
):
    # 检查配置，包括value模型的设置、从头训练|续训、优化器和学习率调度器配置、job_name、输出目录设置等
    cfg.validate()

    if accelerator is None:
        # 未传入 accelerator 对象时，自动创建一个 Accelerator 实例。会自动处理设备分配、混合精度、梯度同步等分布式训练的复杂细节，让训练代码对单卡/多卡透明。
        from accelerate.utils import DistributedDataParallelKwargs

        ddp_kwargs = DistributedDataParallelKwargs(find_unused_parameters=False)
        force_cpu = cfg.value.device == "cpu"
        accelerator = Accelerator(
            gradient_accumulation_steps=cfg.gradient_accumulation_steps,
            step_scheduler_with_optimizer=False,
            kwargs_handlers=[ddp_kwargs],
            cpu=force_cpu,
        )

    init_logging(log_file=f"{cfg.output_dir}/value_train.log", accelerator=accelerator)
    is_main_process = accelerator.is_main_process

    if is_main_process:
        # 在主进程中打印完整的训练配置，方便调试和记录训练设置。pformat 用于格式化输出，使其更易读。
        logging.info(pformat(cfg.to_dict()))

    wandb_logger = make_logger(cfg) if is_main_process else None
    if wandb_logger is None and is_main_process:
        logging.info(colored("Logs will be saved locally.", "yellow", attrs=["bold"]))

    if cfg.seed is not None:
        set_seed(cfg.seed, accelerator=accelerator)

    device = accelerator.device
    torch.backends.cudnn.benchmark = True
    torch.backends.cuda.matmul.allow_tf32 = True

    if is_main_process:
        logging.info("Creating dataset")
        # 仅主进程加载数据集，避免多进程重复加载同一数据集导致的资源浪费。其他进程会在 accelerator.prepare() 时等待主进程完成数据集加载。
        dataset = make_dataset(cfg)

    accelerator.wait_for_everyone()

    if not is_main_process:
        # 非主进程创建数据集的占位对象，确保 accelerator.prepare() 时所有进程都有 dataset 对象，避免因某些进程缺少 dataset 而导致的错误。
        dataset = make_dataset(cfg)

    if is_main_process:
        logging.info("Creating value model")
    # 根据配置创建 value 模型实例。make_policy 函数会根据 cfg.value.type 来实例化对应类型的模型，并根据 cfg.value.input_features 和 cfg.value.output_features 来设置模型的输入输出特征维度。
    # 同时，cfg.rename_map 可以用于在预处理阶段重命名输入输出字段，以适配不同数据集和模型之间的字段命名差异。
    value_model = make_policy(
        cfg=cfg.value,
        ds_meta=dataset.meta,
        rename_map=cfg.rename_map,
    )

    value_target_raw_batch_hook = None
    if has_method(value_model, "build_training_raw_batch_hook"):
        # 训练开始前，代码会遍历数据集中所有帧，为每帧预算出一个标量值目标 value_target(即标签)，公式为：
        # g = -(remaining_steps)                    # 成功 episode
        # g = -(remaining_steps) - c_fail           # 失败 episode（额外惩罚）
        # c_fail = task_max_length * c_fail_coef    # 默认 c_fail_coef=1.0
        # value_target = clip(g / (task_max_length + c_fail), -1.0, 0.0)
        # 含义：当前帧距离 episode 结束越远，value 越低（越负）；失败的 episode 所有帧再额外扣分。最终值域为 [-1, 0]。
        value_target_raw_batch_hook = value_model.build_training_raw_batch_hook(
            dataset=dataset,
            targets_cfg=cfg.targets,
        )
        if is_main_process:
            logging.info("Using value model raw-batch hook for target construction.")
    elif is_main_process:
        logging.info("Value model does not define a raw-batch hook; using dataset targets as-is.")

    accelerator.wait_for_everyone()

    processor_kwargs = {}
    postprocessor_kwargs = {}
    if (cfg.value.pretrained_path and not cfg.resume) or not cfg.value.pretrained_path:
        # 如果是从头训练，或者“指定了预训练路径但不继续训练”（resume=false），则将数据集的统计信息（如均值、方差等）传递给预处理器，用于数据的标准化和归一化。这有助于模型更快地收敛。
        processor_kwargs["dataset_stats"] = dataset.meta.stats

    if cfg.value.pretrained_path is not None:
        # 如果加载了预训练模型
        # 1、强制 device_processor 使用当前设备类型（CPU/GPU）。
        # 2、为 normalizer_processor 指定统计信息、输入输出特征和归一化映射。
        # 3、为 rename_observations_processor 设置字段重命名映射 cfg.rename_map。
        # 4、为后处理器中的 unnormalizer_processor 指定反归一化所需的信息。
        processor_kwargs["preprocessor_overrides"] = {
            "device_processor": {"device": device.type},
            "normalizer_processor": {
                "stats": dataset.meta.stats,
                "features": {**(cfg.value.input_features or {}), **(cfg.value.output_features or {})},
                "norm_map": cfg.value.normalization_mapping,
            },
        }
        processor_kwargs["preprocessor_overrides"]["rename_observations_processor"] = {
            "rename_map": cfg.rename_map
        }
        postprocessor_kwargs["postprocessor_overrides"] = {
            "unnormalizer_processor": {
                "stats": dataset.meta.stats,
                "features": cfg.value.output_features or {},
                "norm_map": cfg.value.normalization_mapping,
            },
        }

    # 最终根据策略配置 cfg.value 和 pretrained_path，以及准备好的参数创建 preprocessor 和 postprocessor（前处理器和后处理器）。
    preprocessor, postprocessor = make_pre_post_processors(
        policy_cfg=cfg.value,
        pretrained_path=cfg.value.pretrained_path,
        **processor_kwargs,
        **postprocessor_kwargs,
    )

    if is_main_process:
        logging.info("Creating optimizer and scheduler")
    optimizer, lr_scheduler = make_optimizer_and_scheduler(cfg, value_model)

    step = 0
    if cfg.resume:
        if cfg.checkpoint_path is None:
            raise ValueError("'checkpoint_path' is missing while resume=true.")
        # 加载之前训练的模型权重、优化器状态、学习率调度器状态和训练步骤数，以便从上次中断的地方继续训练。
        step, optimizer, lr_scheduler = load_training_state(cfg.checkpoint_path, optimizer, lr_scheduler)
    
    # 统计模型的可训练参数数量和总参数数量，并在主进程中打印这些信息以及训练的总步骤数、数据集的帧数和集数、有效批量大小等关键信息，帮助用户了解模型规模和训练设置。
    num_learnable_params = sum(p.numel() for p in value_model.parameters() if p.requires_grad)
    num_total_params = sum(p.numel() for p in value_model.parameters())

    if is_main_process:
        # 输出训练的关键信息，包括输出目录、训练步骤数、数据集规模（帧数和集数）、有效批量大小（单卡批量大小乘以进程数）以及模型参数数量。这些信息对于监控训练过程和评估模型规模非常重要。
        logging.info(colored("Output dir:", "yellow", attrs=["bold"]) + f" {cfg.output_dir}")
        logging.info(f"{cfg.steps=} ({format_big_number(cfg.steps)})")      # 默认8K步
        logging.info(f"{dataset.num_frames=} ({format_big_number(dataset.num_frames)})")
        logging.info(f"{dataset.num_episodes=}")
        num_processes = accelerator.num_processes
        effective_bs = cfg.batch_size * num_processes * cfg.gradient_accumulation_steps
        logging.info(f"Effective batch size: {cfg.batch_size} x {num_processes} x {cfg.gradient_accumulation_steps} = {effective_bs}")
        logging.info(f"{num_learnable_params=} ({format_big_number(num_learnable_params)})")
        logging.info(f"{num_total_params=} ({format_big_number(num_total_params)})")

    # 创建 PyTorch DataLoader，负责从数据集中加载数据批次。根据配置设置 num_workers、batch_size、shuffle（如果数据集不是流式的）、pin_memory（如果使用 GPU）等参数，以优化数据加载性能。
    dataloader = torch.utils.data.DataLoader(
        dataset,
        num_workers=cfg.num_workers,
        batch_size=cfg.batch_size,
        shuffle=not cfg.dataset.streaming,
        sampler=None,
        pin_memory=device.type == "cuda",
        drop_last=False,
        prefetch_factor=2 if cfg.num_workers > 0 else None,
    )

    accelerator.wait_for_everyone()
    # 使用 accelerator.prepare() 来准备模型、优化器、数据加载器和学习率调度器，使它们适应分布式训练环境。这个方法会自动处理设备分配、混合精度、梯度同步等细节，让训练代码对单卡/多卡透明。
    value_model, optimizer, dataloader, lr_scheduler = accelerator.prepare(
        value_model, optimizer, dataloader, lr_scheduler
    )
    # 创建一个无限循环的数据加载器迭代器，确保在训练过程中不断从数据集中获取数据批次。当迭代器耗尽时会自动重新开始，从而实现连续训练。
    dl_iter = cycle(dataloader)

    value_model.train()

    train_metrics = {
        "loss": AverageMeter("loss", ":.3f"),
        "grad_norm": AverageMeter("grdn", ":.3f"),
        "lr": AverageMeter("lr", ":0.1e"),
        "update_s": AverageMeter("updt_s", ":.3f"),
        "dataloading_s": AverageMeter("data_s", ":.3f"),
    }

    # 一个 step 处理 batch_size * gradient_accumulation_steps * num_processes 个样本，
    # 据此统计每步样本量，使吞吐 / epoch 计数正确。
    effective_batch_size = (
        cfg.batch_size * accelerator.num_processes * cfg.gradient_accumulation_steps
    )
    # 创建一个 MetricsTracker 对象，用于跟踪和记录训练过程中的各种指标（如损失、梯度范数、学习率、更新时间和数据加载时间等）。这个 tracker 会在每个训练步骤更新指标，并在指定的日志频率时输出日志和记录到 WandB。
    train_tracker = MetricsTracker(
        effective_batch_size,
        dataset.num_frames,
        dataset.num_episodes,
        train_metrics,
        initial_step=step,
        accelerator=accelerator,
    )

    if is_main_process:
        logging.info(
            f"Start value training on a fixed dataset, with effective batch size: {effective_batch_size}"
        )

    logged_first_prompt = False

    # 一个 step = 一次 optimizer 更新，覆盖 batch_size * gradient_accumulation_steps
    # (* num_processes) 个样本。micro-batch 的取数/打标签/预处理已下沉进 update_policy：
    # 原始 batch（observation.state[B,7] / top / wrist / task / index）依次经过
    #   NormalizerProcessorStep → Pistar06PrepareTaskPromptProcessorStep →
    #   TokenizerProcessorStep → Pistar06PrepareImagesProcessorStep → value_target_hook，
    # 最终得到 input_ids / images / image_attention_mask 与 observation.value_target([-1,0])。
    for _ in range(step, cfg.steps):
        train_tracker, output_dict, last_batch = update_policy(
            train_tracker,
            value_model,
            dl_iter,
            value_target_raw_batch_hook,
            preprocessor,
            optimizer,
            cfg.optimizer.grad_clip_norm,
            accelerator=accelerator,
            grad_accum_steps=cfg.gradient_accumulation_steps,
            step=step,
            lr_scheduler=lr_scheduler,
        )

        if (
            is_main_process
            and not logged_first_prompt
            and last_batch is not None
            and cfg.value.task_field in last_batch
        ):
            # 记录首个 prompt，便于确认任务类型与输入格式。
            task_batch = last_batch[cfg.value.task_field]
            if isinstance(task_batch, str):
                first_prompt = task_batch
            elif len(task_batch) > 0:
                first_prompt = task_batch[0]
            else:
                first_prompt = None
            if first_prompt is not None:
                logging.info("First value prompt:\n%s", first_prompt)
                logged_first_prompt = True

        step += 1
        train_tracker.step()
        # 判断记录日志
        is_log_step = cfg.log_freq > 0 and step % cfg.log_freq == 0 and is_main_process
        # 每隔一段步数保存一次，并保存最后一次训练的结果
        is_saving_step = step % cfg.save_freq == 0 or step == cfg.steps

        if is_log_step:
            logging.info(train_tracker)
            if wandb_logger:
                wandb_log_dict = train_tracker.to_dict()
                if output_dict:
                    wandb_log_dict.update(output_dict)
                wandb_logger.log_dict(wandb_log_dict, step)
            train_tracker.reset_averages()

        if cfg.save_checkpoint and is_saving_step:
            if is_main_process:
                # 保存checkpoint
                logging.info(f"Checkpoint value model after step {step}")
                checkpoint_dir = get_step_checkpoint_dir(cfg.output_dir, cfg.steps, step)
                save_checkpoint(
                    checkpoint_dir=checkpoint_dir,
                    step=step,
                    cfg=cfg,
                    policy=accelerator.unwrap_model(value_model),
                    optimizer=optimizer,
                    scheduler=lr_scheduler,
                    preprocessor=preprocessor,
                    postprocessor=postprocessor,
                )
                update_last_checkpoint(checkpoint_dir)
                if wandb_logger:
                    wandb_logger.log_policy(checkpoint_dir)

            accelerator.wait_for_everyone()

    if is_main_process:
        logging.info("End of value training")

        if cfg.value.push_to_hub:
            unwrapped_value = accelerator.unwrap_model(value_model)
            if cfg.value.use_peft:
                unwrapped_value.push_model_to_hub(cfg, peft_model=unwrapped_value)
            else:
                unwrapped_value.push_model_to_hub(cfg)
            preprocessor.push_to_hub(cfg.value.repo_id)

    # 显式清理，避免程序卡在退出阶段。先释放 DataLoader worker / pin_memory
    # thread，再结束 accelerator；否则 final barrier 或 end_training 期间仍可能持有
    # cycle(dataloader) 的底层 iterator，导致主进程等待 worker 退出。
    _run_cleanup_with_timeout(
        "DataLoader",
        lambda: _shutdown_dataloader_iterator(dl_iter),
        timeout_s=30.0,
    )
    dl_iter = None
    dataloader = None

    _run_cleanup_with_timeout(
        "experiment logger",
        lambda: _finish_training_logger(wandb_logger),
        timeout_s=60.0,
    )

    logging.info("Waiting for accelerator processes to finish")
    accelerator.wait_for_everyone()
    logging.info("Ending accelerator training")
    accelerator.end_training()

    # 销毁 NCCL/分布式进程组，让 watchdog 线程退出，否则多卡训练结束后会 hang。
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        logging.info("Destroying distributed process group")
        torch.distributed.destroy_process_group()

    import gc

    gc.collect()
    logging.info("Value training cleanup complete")


def main():
    # 发现并导入第三方 LeRobot 插件，以便它们能够注册自身。
    register_third_party_plugins()
    value_train()
    logging.shutdown()
    if os.environ.get("LEROBOT_VALUE_TRAIN_HARD_EXIT", "1") != "0":
        os._exit(0)
    sys.exit(0)


if __name__ == "__main__":
    main()
