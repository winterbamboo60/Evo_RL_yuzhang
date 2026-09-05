# 下载项目
## 直接下载
下载main分支项目

git clone https://github.com/winterbamboo60/Evo_RL_yuzhang.git

直接下载dev分支项目
git clone -b dev --single-branch https://github.com/winterbamboo60/RLinf_yuzhang.git

## 切换分支
查看当前分支
git branch --show-current

项目下载后获取dev分支并下载
git fetch origin dev
git switch --track origin/dev

如果本地已经存在 dev 分支，只需更新：
git switch dev
git pull --ff-only origin dev

## 重新建立连接
如果项目中确实没有 .git，可以重新建立连接：

cd Evo_RL_yuzhang
git init
git remote add origin https://github.com/winterbamboo60/Evo_RL_yuzhang.git
git fetch origin
git branch -r

# 项目上传
git add .
git commit -m "dev:[本次更新的内容]"
git push

# 项目更新
## 完全采用最新git代码
cd 你的项目目录
git fetch origin
git reset --hard origin/main
git clean -fd

## 切换分支，完全采用最新git代码
git fetch origin
git switch dev
git reset --hard origin/dev
git clean -fd

## 主动整合
选择merge模式：
git pull

如果已经发生 merge 冲突，需要逐个选择版本：
git status

保留当前本地分支版本：
git checkout --ours 文件路径
git add 文件路径

保留待合并进来的云端分支版本：
git checkout --theirs 文件路径
git add 文件路径

全部冲突文件处理完后提交：
git commit -m "Resolve merge conflicts"