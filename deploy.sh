#!/bin/bash

# 设置变量
ANSIBLE_DIR="./ansible"
INVENTORY_FILE="$ANSIBLE_DIR/inventory.ini"
PLAYBOOK_FILE="$ANSIBLE_DIR/deploy.yml"

# 检查Ansible是否安装
if ! command -v ansible &> /dev/null; then
    echo "Ansible is not installed. Please install it first."
    exit 1
fi

# 检查必要文件是否存在
if [ ! -f "$INVENTORY_FILE" ]; then
    echo "Inventory file not found: $INVENTORY_FILE"
    exit 1
fi

if [ ! -f "$PLAYBOOK_FILE" ]; then
    echo "Playbook file not found: $PLAYBOOK_FILE"
    exit 1
fi

# 检查 Dryrun 模式
if [ "$1" == "--dryrun" ]; then
    echo "Dryrun mode enabled."
    DRYRUN="--check"
fi

# 执行Ansible部署
echo "Starting deployment using Ansible..."
ANSIBLE_CONFIG="$ANSIBLE_DIR/ansible.cfg" ansible-playbook -i "$INVENTORY_FILE" "$PLAYBOOK_FILE" $DRYRUN

# 检查部署结果
if [ $? -eq 0 ]; then
    echo "Deployment completed successfully."
else
    echo "Deployment failed. Please check the Ansible output for errors."
    exit 1
fi