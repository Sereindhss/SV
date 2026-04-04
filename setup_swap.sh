#!/bin/bash
# 创建 4GB 交换文件以避免 OOM
# 使用方法: sudo bash setup_swap.sh

echo "正在创建 4GB 交换空间..."

# 创建交换文件
sudo fallocate -l 4G /swapfile || sudo dd if=/dev/zero of=/swapfile bs=1M count=4096

# 设置权限
sudo chmod 600 /swapfile

# 设置为交换文件
sudo mkswap /swapfile

# 启用交换
sudo swapon /swapfile

# 验证
echo "交换空间已启用："
free -h

echo ""
echo "如果想永久启用，请运行："
echo "echo '/swapfile none swap sw 0 0' | sudo tee -a /etc/fstab"
