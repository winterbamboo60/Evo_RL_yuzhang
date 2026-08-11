#!/usr/bin/env python3
from piper_sdk import *
import time
import math

PI = math.pi
factor = 1000 * 180 / PI
receive_object_center = False
object_center = []
simulation = True


def control_arm(arm, joints, speed=2, mit_mode=0x00):
    # arm：要控制的 piper 接口（显式传入，避免误用全局变量）。
    # mit_mode：从臂用 0x00（位置模式）；主臂 command_high_follow 时需 0xAD（MIT high-follow），
    #           否则发了 JointCtrl 也不动。
    position = joints

    joint_0 = int(position[0] * factor)
    joint_1 = int(position[1] * factor)
    joint_2 = int(position[2] * factor)
    joint_3 = int(position[3] * factor)
    joint_4 = int(position[4] * factor)
    joint_5 = int(position[5] * factor)

    if (joint_4 < -70000) :
        joint_4 = -70000

    arm.MotionCtrl_2(0x01, 0x01, speed, mit_mode)
    arm.JointCtrl(joint_0, joint_1, joint_2, joint_3, joint_4, joint_5)

    # if len(joints) > 6:
    #     joint_6 = round(position[6] * 1000 * 1000)
    #     piper.GripperCtrl(abs(joint_6), 1000, 0x01, 0)

    # print(piper.GetArmStatus())
    # print(position)


def _read_arm_joints_raw(arm):
    """读取当前 6 关节反馈（raw 单位=毫度，与 JointCtrl 入参一致）。"""
    js = arm.GetArmJointMsgs().joint_state
    return [int(getattr(js, f"joint_{i}", 0)) for i in range(1, 7)]


def control_arm_smooth(arm, joints, mit_mode=0xAD, duration_s=4.0, step_dt_s=0.02, speed=20):
    """从“当前位姿”插值到目标，逐步发 JointCtrl，自己控速（不依赖固件 speed）。

    主臂在 0xAD(MIT high-follow) 下 MotionCtrl_2 的 speed 不生效，单条 JointCtrl 会全速冲向目标。
    这里把“一次大跳”拆成 duration_s/step_dt_s 个小步、每步 sleep，复刻 lerobot 实时镜像时
    “高频小步”的平滑效果；整体快慢由 duration_s 决定，与模式无关。
    """
    goal = [int(joints[i] * factor) for i in range(6)]
    if goal[4] < -70000:
        goal[4] = -70000

    start = _read_arm_joints_raw(arm)
    steps = max(int(duration_s / step_dt_s), 1)
    for s in range(1, steps + 1):
        a = s / steps
        cmd = [int(start[i] + (goal[i] - start[i]) * a) for i in range(6)]
        if cmd[4] < -70000:
            cmd[4] = -70000
        arm.MotionCtrl_2(0x01, 0x01, speed, mit_mode)  # 保持模式刷新（0xAD 下 speed 仅占位）
        arm.JointCtrl(*cmd)
        time.sleep(step_dt_s)


def enable_fun(piper: C_PiperInterface_V2):
    '''
    使能机械臂并检测使能状态,尝试5s,如果使能超时则退出程序
    '''
    enable_flag = False
    # 设置超时时间（秒）
    timeout = 5
    # 记录进入循环前的时间
    start_time = time.time()
    elapsed_time_flag = False
    while not (enable_flag):
        elapsed_time = time.time() - start_time
        enable_flag = piper.GetArmLowSpdInfoMsgs().motor_1.foc_status.driver_enable_status and \
                      piper.GetArmLowSpdInfoMsgs().motor_2.foc_status.driver_enable_status and \
                      piper.GetArmLowSpdInfoMsgs().motor_3.foc_status.driver_enable_status and \
                      piper.GetArmLowSpdInfoMsgs().motor_4.foc_status.driver_enable_status and \
                      piper.GetArmLowSpdInfoMsgs().motor_5.foc_status.driver_enable_status and \
                      piper.GetArmLowSpdInfoMsgs().motor_6.foc_status.driver_enable_status
        print("使能状态:", enable_flag)
        piper.EnableArm(7)

        # 检查是否超过超时时间
        if elapsed_time > timeout:
            print("超时....")
            elapsed_time_flag = True
            enable_flag = True
            break
        time.sleep(1)
        pass
    if (elapsed_time_flag):
        print("程序自动使能超时,退出程序")
        exit(0)


if __name__ == "__main__":
    print("OK")
    piper = C_PiperInterface_V2("can0")
    piper.ConnectPort()
    piper.EnableArm(7)
    enable_fun(piper=piper)
    time.sleep(2)
    piper.GripperCtrl(70000, 1000, 0x01, 0)

    # 设置初始位置：can0 是主臂(leader)。0xAD 下 speed 无效，用插值平滑慢速归位（约 4 秒）。
    joints = [0, 0, 0, 0, 0, 0, 0]
    control_arm_smooth(piper, joints, mit_mode=0xAD, duration_s=4.0)
    time.sleep(1)


    piper2 = C_PiperInterface_V2("can1")
    print(f"piper2: {piper2}")
    piper2.ConnectPort()
    piper2.EnableArm(7)
    enable_fun(piper=piper2)
    time.sleep(2)
    piper2.GripperCtrl(70000, 1000, 0x01, 0)

    # 设置初始位置：can1 是从臂(follower)，0x00 位置模式下 speed 生效，单条指令即可（speed=50）。
    # 如也想更平滑，可换成 control_arm_smooth(piper2, joints, mit_mode=0x00, duration_s=4.0)。
    joints = [0, 0, 0, 0, 0, 0, 0]
    control_arm(piper2, joints, 50, mit_mode=0x00)
    time.sleep(2)
 
