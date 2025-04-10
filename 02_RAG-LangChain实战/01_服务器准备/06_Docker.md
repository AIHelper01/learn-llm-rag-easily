# Docker

> 参考文档 https://developer.aliyun.com/mirror/docker-ce?spm=a2c6h.13651102.0.0.297b1b118YKpKj

## 检查卸载老版本Docker

```
sudo apt-get remove docker docker-engine docker.io containerd runc
```

## 安装必要的一些系统工具

```
sudo apt-get update
sudo apt-get install ca-certificates curl gnupg
```

## 信任 Docker 的 GPG 公钥

```
sudo install -m 0755 -d /etc/apt/keyrings
```

```
curl -fsSL https://mirrors.aliyun.com/docker-ce/linux/ubuntu/gpg | sudo gpg --dearmor -o /etc/apt/keyrings/docker.gpg
```

```
sudo chmod a+r /etc/apt/keyrings/docker.gpg
```

## 写入软件源信息

```
echo \
  "deb [arch=$(dpkg --print-architecture) signed-by=/etc/apt/keyrings/docker.gpg] https://mirrors.aliyun.com/docker-ce/linux/ubuntu \
  "$(. /etc/os-release && echo "$VERSION_CODENAME")" stable" | \
  sudo tee /etc/apt/sources.list.d/docker.list > /dev/null
```

## 安装Docker

```
sudo apt-get update
```

```
sudo apt-get install docker-ce docker-ce-cli containerd.io docker-buildx-plugin docker-compose-plugin
```

安装指定版本的Docker-CE:

```
# Step 1: 查找Docker-CE的版本:
# apt-cache madison docker-ce
#   docker-ce | 17.03.1~ce-0~ubuntu-xenial | https://mirrors.aliyun.com/docker-ce/linux/ubuntu xenial/stable amd64 Packages
#   docker-ce | 17.03.0~ce-0~ubuntu-xenial | https://mirrors.aliyun.com/docker-ce/linux/ubuntu xenial/stable amd64 Packages
# Step 2: 安装指定版本的Docker-CE: (VERSION例如上面的17.03.1~ce-0~ubuntu-xenial)
# sudo apt-get -y install docker-ce=[VERSION]
```



## 配置镜像源

首先进入`/etc/docker/daemon.json`文件

然后在里面加入下面的配置

```
{
"registry-mirrors": ["https://docker.registry.cyou",
"https://docker-cf.registry.cyou",
"https://dockercf.jsdelivr.fyi",
"https://docker.jsdelivr.fyi",
"https://dockertest.jsdelivr.fyi",
"https://mirror.aliyuncs.com",
"https://dockerproxy.com",
"https://mirror.baidubce.com",
"https://docker.m.daocloud.io",
"https://docker.nju.edu.cn",
"https://docker.mirrors.sjtug.sjtu.edu.cn",
"https://docker.mirrors.ustc.edu.cn",
"https://mirror.iscas.ac.cn",
"https://docker.rainbond.cc"]
}
```

重新启动docker

```
systemctl daemon-reload

systemctl restart docker
```

然后再拉镜像





## 卸载



1.停止 Docker 服务

```
sudo systemctl stop docker.socket
sudo systemctl stop docker.service
```





2.卸载 Docker 软件包

```
# 移除 Docker 核心组件

sudo apt-get purge -y \
    docker-ce \
    docker-ce-cli \
    containerd.io \
    docker-buildx-plugin \
    docker-compose-plugin
```



```
# 移除残留依赖

sudo apt-get autoremove -y --purge
```



3.删除 Docker 数据和配置文件

```
# 删除 Docker 数据目录

sudo rm -rf /var/lib/docker
sudo rm -rf /var/lib/containerd

# 删除配置文件

sudo rm -rf /etc/docker
sudo rm -rf /etc/containerd

# 删除用户组（如有）

sudo groupdel docker
```



4.清理仓库配置

```
# 删除 Docker 官方仓库源

sudo rm -f /etc/apt/sources.list.d/docker.list
sudo rm -f /etc/apt/keyrings/docker.gpg
```

5.删除残留镜像和容器（可选）

```
# 强制删除所有容器、镜像、卷和网络（谨慎操作！）

sudo docker rm -f $(sudo docker ps -aq) 2>/dev/null
sudo docker rmi -f $(sudo docker images -aq) 2>/dev/null
sudo docker volume prune -f
sudo docker network prune -f
```



6.验证卸载

```
# 检查 Docker 命令是否存在（应提示未找到）

which docker
```

```
# 检查 Docker 服务状态（应提示无此服务）

systemctl status docker
```

