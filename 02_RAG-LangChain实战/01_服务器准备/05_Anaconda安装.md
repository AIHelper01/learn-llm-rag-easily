# Anaconda

## 安装

### Anaconda安装包下载

使用 wget下载

```
wget https://repo.anaconda.com/archive/Anaconda3-2024.10-1-Linux-x86_64.sh
```

### 安装Anaconda

安装软件依赖包：

```
apt install libgl1-mesa-glx libegl1-mesa libxrandr2 libxrandr2 libxss1 libxcursor1 libxcomposite1 libasound2 libxi6 libxtst6
```

运行脚本:

```
bash Anaconda3-2024.10-1-Linux-x86_64.sh
```

注意设置安装目录

```
Version 4.0 | Last Modified: March 31, 2024 | ANACONDA TOS


Do you accept the license terms? [yes|no]
>>>   yes

Anaconda3 will now be installed into this location:
/opt/anaconda3

  - Press ENTER to confirm the location
  - Press CTRL-C to abort the installation
  - Or specify a different location below

[/opt/anaconda3] >>> /opt/anaconda3
PREFIX=/opt/anaconda3
Unpacking payload ...
Extracting: astropy-6.1.3-py312h5eee18b_0.conda:  59%|███████████████████████████████▌                     | 302/508 [00:06<00:05, 39.65it/s]
```



### 修改 .bashrc

通过修改 ~/.bashrc文件来激活安装

在 ~/.bashrc 末尾添加:

```
vim  ~/.bashrc
```



```
export PATH=/opt/anaconda3/bin:$PATH
source /opt/anaconda3/bin/activate
```

 后执行：

```
source ~/.bashrc
```

完成后，您将被置于Anaconda的默认base编程环境中。

虽然Anaconda附带了这个默认的base编程环境,但您应该为您的程序创建单独的环境,并使它们彼此隔离。

您可以使用conda命令进一步验证安装,例如list:

```
conda env list
```

```
(base) root@server2:/opt/software# conda env list 
# conda environments:
#
base                  *  /opt/anaconda3
```







## 卸载

找到 Anaconda 安装目录
首先，确认 Anaconda 的安装目录。可以通过以下命令查找：

```
which conda
```

输出示例：

```
(base) root@server2:/opt/software# which conda 
/opt/anaconda3/bin/conda
```

这表明 Anaconda 安装在 `/opt/anaconda3` 目录。

删除 Anaconda 安装目录
使用 rm -rf 命令删除整个 Anaconda 安装目录。例如：

```
rm -rf /opt/anaconda3
```

清理环境变量

使更改生效：

```
source ~/.bashrc 
```



检查是否卸载成功
运行以下命令检查 conda 是否已卸载：

```
conda --version
```

如果返回 `conda: command not found`，说明卸载成功。



# 修改pip镜像源

**PIP 镜像源**是指用于加速 Python 包管理工具 `pip` 下载速度的镜像站点。默认情况下，`pip` 会从 Python 官方的 PyPI（Python Package Index）服务器下载包，但在某些地区访问速度可能较慢，因此可以切换到更快的镜像源。

以下操作以aliyun镜像源为例：

**命令行修改源（推荐）**

```
pip config set global.index-url https://mirrors.aliyun.com/pypi/simple/  
```

提示：

成功

查看当前源：

```
pip3 config list
```

显示：

```
global.index-url='https://mirrors.aliyun.com/pypi/simple/'
```



# 安装Jupyter

适合数据分析、人工智能Python项目，创建的文件为xxx.ipynb

## Jupyter安装

使用 `pip`（Python 包管理工具）来安装 Jupyter。按照以下步骤进行：

1. **安装 Jupyter**：在终端中运行以下命令：

   ```bash
   pip install jupyter
   ```

   这将安装 Jupyter， 包括Jupyter Notebook 、 Jupyter Lab 及其依赖项。

   

2. **验证安装成功**

   ```
   jupyter lab
   ```

   自动弹出浏览器，打开网页版编辑器

   


## 修改默认工作目录

建议修改jupyter的默认工作目录为本地常用路径，方便使用。

创建notebook的配置文件，命令行中执行：

```bash
jupyter notebook --generate-config
```

打开`upyter_notebook_config.py`，例如记事本软件

搜索notebook_dir字段进行修改,

注意：删除#，去注释；顶格写，开头的空格也要删除

再次**启动 Jupyter Lab**，修改生效：

```bash
jupyter lab
```



## Jupyter修改解释器

初始的Jupyter只有一个系统内核

当我们想要像Pycharm中一样自由地选择python解析器时，

需要经历两步：创建Python系统级虚拟环境 + 添加新的kernel



## 虚拟环境

```
conda create --name <虚拟环境名称> python=3.12
```



## 添加新的kernel

使用以下指令添加新的kernel: 


```bash
(skleran) E:\ProgramData\Python312\env\skleran\Scripts>pip install ipykernel
```

```bash
(skleran) E:\ProgramData\Python312\env\skleran\Scripts>python -m ipykernel install --name sklearn
```

刷新JupyterLab页面，新增的解释器可选，

后续pip install xxx 时，建议通过DOS界面，进入虚拟环境进行安装



## 删除kernel

查看安装的内核和位置

```shell
jupyter kernelspec list
```

<img src="E:/Learn-Python-Easily/00_开发环境搭建/assets/python30.png" width="100%">

删除Python解析器

```shell
jupyter kernelspec remove sklearn		# 将Jupyter中的sklearn虚拟环境删除掉
```





## 设置远程登陆

生成配置文件

```
jupyter notebook --generate-config
```

设置密码

```
jupyter notebook password
(base) root@iZwz9fhjq09pqz5njvics0Z:/opt/software# jupyter notebook password
Enter password:
Verify password:
[JupyterPasswordApp] Wrote hashed password to /root/.jupyter/jupyter_server_config.json
(base) root@iZwz9fhjq09pqz5njvics0Z:/opt/software#
```

获取密码

```
cat /root/.jupyter/jupyter_server_config.json
{
  "IdentityProvider": {
    "hashed_password": "hashed_password": "argon2:$argon2id$v=19$m=10240,t=10,p=8$OoFDoyZtpDpSErhEa5ebSw$YOzERE1lXToSyowDpAzyzIS+sEoBpmlu+jvp8rRX+yw"
  }
}
```

在jupyter_notebook_config.py配置文件文末添加以下配置

```
vim /root/.jupyter/jupyter_notebook_config.py

c.ServerApp.ip = '*' #本机静态IP 建议使用*
c.ServerApp.password = "argon2:$argon2id$v=19$m=10240,t=10,p=8$OoFDoyZtpDpSErhEa5ebSw$YOzERE1lXToSyowDpAzyzIS+sEoBpmlu+jvp8rRX+yw'
# 这个是刚要保存的秘钥
c.ServerApp.open_browser = False # 运行时不打开本机浏览器
c.ServerApp.port = 8888    #端口，可以随意指定 不与系统其他端口冲突即可
c.ServerApp.allow_remote_access = True  #允许远程访问
```
