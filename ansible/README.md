# SenseVoice APP 部署文档

## 部署概述

本文档描述了如何使用Ansible和Docker Compose将SenseVoice APP部署到ttd-worker和ttd-edge服务器上。

## 前提条件

1. 部署机器上已安装Ansible (2.9+)
2. 目标服务器可以通过SSH访问
3. 目标服务器上有足够的磁盘空间和内存

## 部署步骤

### 1. 配置目标服务器信息

编辑`inventory.ini`文件，设置正确的服务器地址和用户名：

```ini
[ttd-worker]
ttd-worker ansible_host=<实际IP地址> ansible_user=<用户名>

[ttd-edge]
ttd-edge ansible_host=<实际IP地址> ansible_user=<用户名>
```

### 2. 配置部署参数

如果需要，可以修改`roles/sensevoice/defaults/main.yml`中的默认参数：

```yaml
app_dir: /opt/sensevoice  # 应用部署目录
data_dir: /opt/sensevoice/data  # 数据目录
app_port: 7086  # 应用端口
```

### 3. 执行部署

运行以下命令开始部署：

```bash
cd ansible
ansible-playbook deploy.yml
```

如果只想部署到特定服务器，可以使用`-l`参数：

```bash
ansible-playbook deploy.yml -l ttd-worker  # 只部署到ttd-worker
ansible-playbook deploy.yml -l ttd-edge    # 只部署到ttd-edge
```

### 4. 验证部署

部署完成后，可以通过以下方式验证：

1. 访问 `http://<服务器IP>:7086` 查看应用是否正常运行
2. 检查Docker容器状态：

```bash
ssh <用户名>@<服务器IP> "docker ps | grep sensevoice"
```

## 故障排除

如果部署过程中遇到问题，可以尝试以下步骤：

1. 检查Ansible日志输出，查找错误信息
2. 登录服务器，检查Docker容器日志：

```bash
docker logs sensevoice-app
docker logs ttd-server
```

3. 检查Docker Compose配置：

```bash
cat /opt/sensevoice/docker-compose.yml
```

4. 如果需要重新部署，可以先停止并删除现有容器：

```bash
cd /opt/sensevoice
docker-compose down
```

然后重新运行Ansible部署脚本。

## 维护

### 更新应用

要更新应用，只需更新源代码并重新运行部署脚本：

```bash
ansible-playbook deploy.yml
```

### 备份数据

数据目录位于服务器的`/opt/sensevoice/data`，可以通过以下命令备份：

```bash
ssh <用户名>@<服务器IP> "tar -czf /tmp/sensevoice-data-backup.tar.gz /opt/sensevoice/data"
scp <用户名>@<服务器IP>:/tmp/sensevoice-data-backup.tar.gz ./
```
