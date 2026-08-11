```bash
# 此处对外仅暴露lerobot-setup-can

Step0. 主从臂都设置成从动模式

Step1. 切换环境

```bash
# mkdir -p ./package_sorting_env_raw && tar -xzf ./package_sorting.tar.gz -C ./package_sorting_env_raw
source /home/hpc/yuzhang/envs/package_sorting_env/bin/activate
/home/hpc/yuzhang/envs/package_sorting_env/bin/conda-unpack
```

模型下载A800
downloadyuzhang() {
    if [ $# -eq 0 ]; then
	echo "用法： d <filename>"
        return 1
    fi
    DOWNLOAD_URL="http://192.168.110.115:20181/yuzhang/download.php"
    REMOTE_FILE="$1"
    if [ -f "${REMOTE_FILE}" ]; then
        rm "${REMOTE_FILE}"
    fi
    if ! curl -f -OJ "${DOWNLOAD_URL}?file=${REMOTE_FILE}"; then
        echo "文件不存在"
    fi
}

数据上传A800

Step2. 查看所有CAN口号
# 检查 ethtool 是否已安装
if ! dpkg -l | grep -q "ethtool"; then
    echo "\e[31m错误: 系统中未检测到 ethtool。\e[0m"
    echo "请使用以下命令安装 ethtool:"
    echo "sudo apt update && sudo apt install ethtool"
    exit 1
fi

# 检查 can-utils 是否已安装
if ! dpkg -l | grep -q "can-utils"; then
    echo "\e[31m错误: 系统中未检测到 can-utils。\e[0m"
    echo "请使用以下命令安装 can-utils:"
    echo "sudo apt update && sudo apt install can-utils"
    exit 1
fi

echo "ethtool 和 can-utils 均已安装。"

# 遍历所有 CAN 接口
for iface in $(ip -br link show type can | awk '{print $1}'); do
    # 使用 ethtool 获取 bus-info
    BUS_INFO=$(sudo ethtool -i "$iface" | grep "bus-info" | awk '{print $2}')
    
    if [ -z "$BUS_INFO" ];then
        echo "错误: 无法获取接口 $iface 的 bus-info 信息。"
        continue
    fi
    
    echo "接口 $iface 插入在 USB 端口 $BUS_INFO"
done



Step3. 初始化2个CAN口
lerobot-setup-can --mode=setup --interfaces=can0,can1

Step4. CAN口模式测试，需要能看到持续输出的数据
lerobot-setup-can --mode=test --interfaces=can0
lerobot-setup-can --mode=test --interfaces=can1

Step5. 摄影机模式测试，需要根据输出确认摄像头索引
python ./lerobot/srclerobot/find_cameras.py opencv && ll ./lerobot/src/outputs/captured_images

进入lerobot虚拟环境：  conda activate lerobot
进入工作目录： cd ~/VLA/lerobot/src/
查询can口号： bash piper_sdk/find_all_can_port.sh 
根据can 口号修改： /home/hpc/VLA/lerobot/src/activate_single_arm_can.sh  里的can口号
激活can口： bash activate_single_arm_can.sh
查询摄像头index号： PYTHONPATH=. python lerobot/find_cameras.py opencv    确认top和wrist视角的index，然后在对应的命令里修改index

conda activate lerobot; cd ~/VLA/lerobot/src/; PYTHONPATH=. python lerobot/find_cameras.py opencv