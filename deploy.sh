#!/bin/bash

# 设置变量
ANSIBLE_DIR="./ansible"
INVENTORY_FILE="$ANSIBLE_DIR/inventory.ini"
PLAYBOOK_FILE="$ANSIBLE_DIR/deploy.yml"
DEPLOY_TYPE="standard"
TARGET_HOSTS="sensevoice_servers"
ANSIBLE_ARGS=""

# 解析第一个参数作为部署类型
case $1 in
  stage)
    DEPLOY_TYPE="opensmile"
    TARGET_HOSTS="stage"
    shift
    ;;
  api)
    TARGET_HOSTS="sensevoice_servers"
    shift
    ;;
  *)
    echo "未知部署类型: $1"
    echo "Usage: $0 [stage | api] [other ansible args]"
    exit 1
    ;;
esac

# 剩余的所有参数都传递给 Ansible
ANSIBLE_ARGS="$@"

# 检查Ansible是否安装
if ! command -v ansible &> /dev/null; then
    echo "Ansible未安装。请先安装Ansible。"
    exit 1
fi

# 检查必要文件是否存在
if [ ! -f "$INVENTORY_FILE" ]; then
    echo "未找到Inventory文件: $INVENTORY_FILE"
    exit 1
fi

if [ ! -f "$PLAYBOOK_FILE" ]; then
    echo "未找到Playbook文件: $PLAYBOOK_FILE"
    exit 1
fi

# 执行Ansible部署
echo "使用Ansible开始部署..."
if [ "$DEPLOY_TYPE" == "opensmile" ]; then
    echo "部署类型: SmileUI (目标: $TARGET_HOSTS)"
else
    echo "部署类型: 标准 (目标: $TARGET_HOSTS)"
fi

ANSIBLE_STDOUT_CALLBACK=debug ANSIBLE_CONFIG="$ANSIBLE_DIR/ansible.cfg" ansible-playbook -i "$INVENTORY_FILE" "$PLAYBOOK_FILE" -e "deploy_type=$DEPLOY_TYPE target_hosts=$TARGET_HOSTS" $ANSIBLE_ARGS

# 检查部署结果
if [ $? -eq 0 ]; then
    echo "部署成功完成。"
else
    echo "部署失败。请检查Ansible输出以获取错误信息。"
    exit 1
fi