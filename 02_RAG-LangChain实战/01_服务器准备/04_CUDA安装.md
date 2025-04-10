## Ubuntu的CUDA和Python的CUDA


Ubuntu 系统中的 CUDA 与 Anaconda 环境中的 CUDA 存在显著差异，主要体现在安装方式、功能范围和使用场景上。以下是两者的核心区别与联系：

---

### **功能定位**

• **Ubuntu 的 CUDA**  
  指通过 NVIDIA 官方安装的完整 CUDA Toolkit，包含编译器（如 `nvcc`）、调试工具、库文件（如 `libcudart.so`）以及 GPU 驱动支持。它是开发 CUDA 程序的基础环境，适合需要编译自定义 CUDA 内核或调试 GPU 代码的场景。

• **Anaconda 的 CUDA**  
  通过 Conda 安装的 `cudatoolkit` 或 `cudnn` 包，仅包含运行时所需的动态链接库（如 `libcuda.so`），不包含开发工具。其目的是为深度学习框架（如 PyTorch、TensorFlow）提供预编译的 GPU 加速支持，无需用户手动编译代码。

---

### **安装方式**

• **Ubuntu 的 CUDA**  
  • 需从 NVIDIA 官网下载 `.run` 或 `.deb` 安装包，手动安装并配置环境变量（如 `PATH` 和 `LD_LIBRARY_PATH`）。
  • 安装过程中需注意驱动版本兼容性，例如 CUDA 12.2 要求 NVIDIA 驱动版本 ≥535.104.05。

• **Anaconda 的 CUDA**  
  • 通过 Conda 命令直接安装（如 `conda install cudatoolkit=11.8`），自动集成到虚拟环境中。
  • 无需单独配置系统路径，依赖项由 Conda 管理，适合多版本切换（例如不同项目使用不同 CUDA 版本）。

---

### 版本管理

• **Ubuntu 的 CUDA**  
  系统全局安装，同一时间仅支持一个主版本。升级需卸载旧版本并重新安装，可能涉及驱动更新。

• **Anaconda 的 CUDA**  
  支持为每个虚拟环境独立指定 CUDA 版本（如环境 A 使用 CUDA 11.8，环境 B 使用 CUDA 12.1），避免版本冲突。

---

### **兼容性与依赖**

• **Ubuntu 的 CUDA**  
  需严格匹配 NVIDIA 驱动版本。例如 CUDA 12.2 要求驱动 ≥535.104.05，且需通过 `nvidia-smi` 验证驱动状态。

• **Anaconda 的 CUDA**  
  仅需系统驱动满足最低要求，无需精确匹配。例如 PyTorch 的 CUDA 12.1 环境可兼容驱动版本 ≥450.80.02。

---

### **典型使用场景**

• **Ubuntu 的 CUDA**  
  • 开发 CUDA 原生程序（如自定义 GPU 加速算法）。
  • 需要调试 CUDA 内核或使用 `nsight` 分析工具。

• **Anaconda 的 CUDA**  
  • 运行预编译的深度学习框架（如 `torch.cuda.is_available()` 验证 GPU 加速是否生效）。
  • 快速部署多版本实验环境，避免系统级配置干扰。

---

### 总结建议

• **普通用户**：若仅需运行 PyTorch/TensorFlow 等框架，优先使用 Anaconda 的 `cudatoolkit`，简化依赖管理。
• **开发者/研究者**：需编译 CUDA 代码时，必须安装 NVIDIA 官方的完整 CUDA Toolkit，并确保驱动兼容性。

## Ubuntu

查看最高可以支持的CUDA版本，比如下面的例子最高支持`CUDA Version: 12.2`

```
(base) root@server1:/opt/software# nvidia-smi 
Wed Mar 26 11:08:57 2025       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       Off | 00000000:17:00.0 Off |                    0 |
| N/A   35C    P8              11W /  70W |      6MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  Tesla T4                       Off | 00000000:31:00.0 Off |                    0 |
| N/A   36C    P8              11W /  70W |      6MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   2  Tesla T4                       Off | 00000000:98:00.0 Off |                    0 |
| N/A   38C    P8              11W /  70W |      6MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   3  Tesla T4                       Off | 00000000:B1:00.0 Off |                    0 |
| N/A   38C    P8              12W /  70W |      6MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      2312      G   /usr/lib/xorg/Xorg                            4MiB |
|    1   N/A  N/A      2312      G   /usr/lib/xorg/Xorg                            4MiB |
|    2   N/A  N/A      2312      G   /usr/lib/xorg/Xorg                            4MiB |
|    3   N/A  N/A      2312      G   /usr/lib/xorg/Xorg                            4MiB |
+---------------------------------------------------------------------------------------+
```





### CUDA的安装

以12.2版本为例：

#### 下载 

下载地址： https://developer.nvidia.com/cuda-12-4-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=22.04&target_type=runfile_local

下载方法：wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run

```
root@server5:/opt/software# wget https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
--2025-04-10 00:33:17--  https://developer.download.nvidia.com/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
正在解析主机 developer.download.nvidia.com (developer.download.nvidia.com)... 184.26.91.210, 184.26.91.211
正在连接 developer.download.nvidia.com (developer.download.nvidia.com)|184.26.91.210|:443... 已连接。
已发出 HTTP 请求，正在等待回应... 301 Moved Permanently
位置：https://developer.download.nvidia.cn/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run [跟随至新的 URL]
--2025-04-10 00:33:17--  https://developer.download.nvidia.cn/compute/cuda/12.4.0/local_installers/cuda_12.4.0_550.54.14_linux.run
正在解析主机 developer.download.nvidia.cn (developer.download.nvidia.cn)... 36.153.62.132, 36.153.62.131, 36.153.62.130, ...
正在连接 developer.download.nvidia.cn (developer.download.nvidia.cn)|36.153.62.132|:443... 已连接。
已发出 HTTP 请求，正在等待回应... 200 OK
长度： 4454353277 (4.1G) [application/octet-stream]
正在保存至: ‘cuda_12.4.0_550.54.14_linux.run.1’

cuda_12.4.0_550.54.14_linux.r   1%[                                                 ]  78.62M  5.10MB/s    剩余 13m 22s
```



#### 安装

```
sudo sh cuda_12.4.0_550.54.14_linux.run
```

然后会弹出选择界面，请用手指点击你键盘的↑ ↓按键。

我们选择Continue ，然后输入accept：

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ Existing package manager installation of the driver found. It is strongly    │
│ recommended that you remove this before continuing.                          │
│ Abort                                                                        │
│ **Continue**                                                                     │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│ Up/Down: Move | 'Enter': Select                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```



```
┌──────────────────────────────────────────────────────────────────────────────┐
│  End User License Agreement                                                  │
│  --------------------------                                                  │
│                                                                              │
│  NVIDIA Software License Agreement and CUDA Supplement to                    │
│  Software License Agreement. Last updated: October 8, 2021                   │
│                                                                              │
│  The CUDA Toolkit End User License Agreement applies to the                  │
│  NVIDIA CUDA Toolkit, the NVIDIA CUDA Samples, the NVIDIA                    │
│  Display Driver, NVIDIA Nsight tools (Visual Studio Edition),                │
│  and the associated documentation on CUDA APIs, programming                  │
│  model and development tools. If you do not agree with the                   │
│  terms and conditions of the license agreement, then do not                  │
│  download or use the software.                                               │
│                                                                              │
│  Last updated: October 8, 2021.                                              │
│                                                                              │
│                                                                              │
│  Preface                                                                     │
│  -------                                                                     │
│                                                                              │
│──────────────────────────────────────────────────────────────────────────────│
│ Do you accept the above EULA? (accept/decline/quit):                         │
│ **accept**                                                                       │
└──────────────────────────────────────────────────────────────────────────────┘

```



按回车取消 Driver 的安装，也就是[ ]里面没有X（因为一般已经装好驱动了），然后回车选择 Install：

```
┌──────────────────────────────────────────────────────────────────────────────┐
│ CUDA Installer                                                               │
│ - [ ] Driver                                                                 │
│      [ ] 535.54.03                                                           │
│ + [X] CUDA Toolkit 12.2                                                      │
│   [X] CUDA Demo Suite 12.2                                                   │
│   [X] CUDA Documentation 12.2                                                │
│ - [ ] Kernel Objects                                                         │
│      [ ] nvidia-fs                                                           │
│   Options                                                                    │
│  **Install**                                                                    │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│                                                                              │
│ Up/Down: Move | Left/Right: Expand | 'Enter': Select | 'A': Advanced options │
└──────────────────────────────────────────────────────────────────────────────┘

```



上面一步，选择 Install后,终端就会跳出选择界面，回到正常终端模式，并且没有任何提示。

这是正常现象，等待安装完成即可。安装完成后，会出现如下界面：

```
(base) root@server1:/opt/software# sudo sh cuda_12.2.0_535.54.03_linux.run
===========
= Summary =
===========

Driver:   Not Selected
Toolkit:  Installed in /usr/local/cuda-12.2/

Please make sure that
 -   PATH includes /usr/local/cuda-12.2/bin
 -   LD_LIBRARY_PATH includes /usr/local/cuda-12.2/lib64, or, add /usr/local/cuda-12.2/lib64 to /etc/ld.so.conf and run ldconfig as root

To uninstall the CUDA Toolkit, run cuda-uninstaller in /usr/local/cuda-12.2/bin
***WARNING: Incomplete installation! This installation did not install the CUDA Driver. A driver of version at least 535.00 is required for CUDA 12.2 functionality to work.
To install the driver using this installer, run the following command, replacing <CudaInstaller> with the name of this run file:
    sudo <CudaInstaller>.run --silent --driver

Logfile is /var/log/cuda-installer.log
```



接着我们需要添加CUDA的环境变量，以便于可以任何地方都能启动它（不知道Linux的环境变量知识点可自行搜索），输入以下命令：

```
sudo vim ~/.bashrc
```



在其中添加并保存：

```
export PATH=/usr/local/cuda-12.2/bin:$PATH
```





接着：

```
source ~/.bashrc
```



至此我们安装cuda成功，可以通过以下命令查看cuda信息：

查看CUDA版本信息：

```
nvcc --version
```



没有问题的话，一般会输出以下信息，里面就可以显示你的CUDA信息了。

```
(base) root@server1:/opt/software# nvcc --version 
nvcc: NVIDIA (R) Cuda compiler driver
Copyright (c) 2005-2023 NVIDIA Corporation
Built on Tue_Jun_13_19:16:58_PDT_2023
Cuda compilation tools, release 12.2, V12.2.91
Build cuda_12.2.r12.2/compiler.32965470_0
```



### CUDA的卸载


在Ubuntu系统中彻底卸载CUDA需要根据安装方式（如Runfile或Deb包）选择对应方法，并清理残留文件及环境变量。以下是具体步骤及注意事项：

---

#### 常规卸载步骤

**1. 通过官方卸载程序卸载（适用于Runfile安装）**

• **执行卸载脚本**：  
  进入CUDA安装目录的`bin`文件夹，运行卸载程序（需替换版本号，如11.4）：  

  ```bash
(base) root@server1~# cd /usr/local/cuda-12.2/bin/
(base) root@server1:# ls
bin2c              crt       cuda-gdbserver    cuobjdump  ncu-ui                       nsys           nvcc                 nvdisasm  nvprune
computeprof        cudafe++  cuda-uninstaller  fatbinary  nsight_ee_plugins_manage.sh  nsys-exporter  __nvcc_device_query  nvlink    nvvp
compute-sanitizer  cuda-gdb  cu++filt          ncu        nsight-sys                   nsys-ui        nvcc.profile         nvprof    ptxas

(base) root@server1:/usr/local/cuda-12.2/bin# ./cuda-uninstaller 
 Successfully uninstalled 
  ```

  此方法适用于CUDA 11.4及更高版本。

• **删除安装目录**：  
  手动移除CUDA文件夹：  

  ```bash
(base) root@server1:/usr/local# rm -rf /usr/local/cuda-12.2/
(base) root@server1:/usr/local# ls
bin  etc  games  include  lib  man  sbin  share  src
  ```

**2. 通过包管理器卸载（适用于Deb包安装）**

**移除CUDA Toolkit及相关包**：  

  ```bash
sudo apt-get remove --purge nvidia-cuda-toolkit cuda-*
sudo apt autoremove  # 清理依赖项
sudo apt autoclean
  ```

#### **清理残留文件**

• **检查并删除残留配置**：  

  ```bash
sudo dpkg -l | grep cuda  # 列出所有CUDA相关包
sudo dpkg -P <包名>        # 逐个强制卸载残留包
  ```

  手动删除可能的残留路径：  

  ```bash
sudo rm -rf /etc/apt/sources.list.d/cuda*.list  # 删除CUDA软件源配置
  ```

---

#### 环境变量清理

• **修改环境变量文件**：  
  打开`~/.bashrc`或`~/.bash_profile`，注释或删除以下类似行：  

  ```bash
# export PATH=/usr/local/cuda-12.2/bin:$PATH
  ```

  更新配置：  

  ```bash
source ~/.bashrc
  ```

---

#### 验证卸载结果

• **检查CUDA版本**：  

  ```bash
nvcc --version  # 若无输出或报错，说明已卸载
  ```