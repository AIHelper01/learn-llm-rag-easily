# NVIDIA驱动安装

## 联网场景

查看驱动是否已安装

```shell
root@server4:~# nvidia-smi 
找不到命令 “nvidia-smi”，但可以通过以下软件包安装它：
apt install nvidia-utils-390         # version 390.157-0ubuntu0.22.04.2, or
apt install nvidia-utils-418-server  # version 418.226.00-0ubuntu5~0.22.04.1
apt install nvidia-utils-450-server  # version 450.248.02-0ubuntu0.22.04.1
apt install nvidia-utils-470         # version 470.256.02-0ubuntu0.22.04.1
apt install nvidia-utils-470-server  # version 470.256.02-0ubuntu0.22.04.1
apt install nvidia-utils-535         # version 535.183.01-0ubuntu0.22.04.1
apt install nvidia-utils-535-server  # version 535.230.02-0ubuntu0.22.04.3
apt install nvidia-utils-545         # version 545.29.06-0ubuntu0.22.04.2
apt install nvidia-utils-550         # version 550.120-0ubuntu0.22.04.1
apt install nvidia-utils-550-server  # version 550.144.03-0ubuntu0.22.04.1
apt install nvidia-utils-565-server  # version 565.57.01-0ubuntu0.22.04.4
apt install nvidia-utils-570-server  # version 570.86.15-0ubuntu0.22.04.4
apt install nvidia-utils-510         # version 510.60.02-0ubuntu1
apt install nvidia-utils-510-server  # version 510.47.03-0ubuntu3
```



获取驱动版本

```shell
sudo ubuntu-drivers devices
```



输出

```shell
root@server5:~# sudo ubuntu-drivers devices
== /sys/devices/pci0000:b0/0000:b0:02.0/0000:b1:00.0 ==
modalias : pci:v000010DEd00001EB8sv000010DEsd000012A2bc03sc02i00
vendor   : NVIDIA Corporation
model    : TU104GL [Tesla T4]
driver   : nvidia-driver-535-server - distro non-free
driver   : nvidia-driver-535 - distro non-free recommended
driver   : nvidia-driver-470 - distro non-free
driver   : nvidia-driver-418-server - distro non-free
driver   : nvidia-driver-545 - distro non-free
driver   : nvidia-driver-570-server - distro non-free
driver   : nvidia-driver-450-server - distro non-free
driver   : nvidia-driver-470-server - distro non-free
driver   : xserver-xorg-video-nouveau - distro free builtin
```





安装驱动

根据推荐安装`nvidia-driver-535`

```
sudo apt install nvidia-driver-535 -y
```



```
sudo apt update 
```



```
#更新好之后重启机器
sudo reboot
```



查看安装结果

```
sudo nvidia-smi 
```


输出

```
root@server5:~# nvidia-smi 
Wed Mar 26 10:03:28 2025       
+---------------------------------------------------------------------------------------+
| NVIDIA-SMI 535.183.01             Driver Version: 535.183.01   CUDA Version: 12.2     |
|-----------------------------------------+----------------------+----------------------+
| GPU  Name                 Persistence-M | Bus-Id        Disp.A | Volatile Uncorr. ECC |
| Fan  Temp   Perf          Pwr:Usage/Cap |         Memory-Usage | GPU-Util  Compute M. |
|                                         |                      |               MIG M. |
|=========================================+======================+======================|
|   0  Tesla T4                       Off | 00000000:17:00.0 Off |                    0 |
| N/A   35C    P8              12W /  70W |      6MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   1  Tesla T4                       Off | 00000000:31:00.0 Off |                    0 |
| N/A   36C    P8              11W /  70W |      6MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   2  Tesla T4                       Off | 00000000:98:00.0 Off |                    0 |
| N/A   39C    P8              12W /  70W |      6MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
|   3  Tesla T4                       Off | 00000000:B1:00.0 Off |                    0 |
| N/A   36C    P8              11W /  70W |      6MiB / 15360MiB |      0%      Default |
|                                         |                      |                  N/A |
+-----------------------------------------+----------------------+----------------------+
                                                                                         
+---------------------------------------------------------------------------------------+
| Processes:                                                                            |
|  GPU   GI   CI        PID   Type   Process name                            GPU Memory |
|        ID   ID                                                             Usage      |
|=======================================================================================|
|    0   N/A  N/A      2332      G   /usr/lib/xorg/Xorg                            4MiB |
|    1   N/A  N/A      2332      G   /usr/lib/xorg/Xorg                            4MiB |
|    2   N/A  N/A      2332      G   /usr/lib/xorg/Xorg                            4MiB |
|    3   N/A  N/A      2332      G   /usr/lib/xorg/Xorg                            4MiB |
+---------------------------------------------------------------------------------------+
```



## 非联网场景

以T4卡为例

1. 查询网站：https://www.nvidia.cn/drivers/lookup/
2. 下载地址：https://www.nvidia.cn/drivers/details/238088/
3. 安装： `sh NVIDIA-Linux-x86_64-535.154.05.run --kernel-source-path=/usr/src/kernels/4.18.0-193.14.2.el8_2.x86_64`