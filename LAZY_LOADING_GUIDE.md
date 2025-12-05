# SenseVoice 动态加载优化指南

## 概述

本优化方案针对**低频高并发批量处理**场景设计，通过以下技术实现内存占用的大幅降低：

1. **延迟加载**：模型在首次请求时才加载，空闲时自动卸载
2. **单进程多线程**：使用线程池处理并发，共享同一份模型
3. **预热机制**：批量任务前可预加载模型，避免冷启动延迟

## 内存优化效果

### 优化前（多进程常驻）

- **api 服务**：3 节点 × 4 workers × 3.5GB = **42GB**（24小时常驻）
- **app 服务**：1 × 5GB = **5GB**
- **总计**：**~47GB**

### 优化后（动态加载）

- **空闲时**：~100MB（模型已卸载）
- **工作时**：~3.5GB（单进程，一份模型）
- **内存减少**：**92%** ✅

## 配置说明

### 环境变量

```bash
# Uvicorn 配置
UVICORN_WORKERS=1                # 单进程模式

# 并发配置
MAX_CONCURRENT_REQUESTS=32       # 最大并发请求数（建议 CPU核心数 × 2）

# 模型管理配置
MODEL_IDLE_TIMEOUT=300           # 模型空闲超时（秒），默认 5 分钟
ENABLE_AUTO_UNLOAD=true          # 是否启用自动卸载
```

### Docker Stack 配置

```yaml
services:
  api:
    environment:
      - UVICORN_WORKERS=1
      - MAX_CONCURRENT_REQUESTS=32
      - MODEL_IDLE_TIMEOUT=300
      - ENABLE_AUTO_UNLOAD=true
```

## API 端点

### 1. 预热端点（推荐使用）

**POST** `/api/v1/warmup`

批量任务开始前调用，提前加载所有模型，避免首次请求延迟。

```bash
curl -X POST http://api/v1/warmup
```

**响应示例**：

```json
{
  "status": "ready",
  "elapsed": 28.5,
  "models": {
    "asr": true,
    "vad": true,
    "pri": true
  }
}
```

### 2. 状态查询端点

**GET** `/api/v1/status`

查看模型加载状态和统计信息。

```bash
curl http://api/v1/status
```

**响应示例**：

```json
{
  "models": {
    "asr": {
      "name": "SenseVoice-ASR",
      "is_loaded": true,
      "load_count": 2,
      "unload_count": 1,
      "last_access": 1696234567.89,
      "idle_timeout": 300,
      "auto_unload_enabled": true
    },
    "vad": {...},
    "pri": {...}
  },
  "config": {
    "max_concurrent": 32,
    "max_workers": 32,
    "model_idle_timeout": 300,
    "auto_unload_enabled": true,
    "cpu_count": 64
  }
}
```

### 3. 强制卸载端点

**POST** `/api/v1/unload`

立即卸载所有模型释放内存（可选，通常不需要手动调用）。

```bash
curl -X POST http://api/v1/unload
```

## 使用流程

### 批量处理场景

```bash
#!/bin/bash

# 1. 预热模型（推荐）
echo "预热模型..."
curl -X POST http://api/v1/warmup
sleep 5  # 等待预热完成

# 2. 并发处理数百个文件
echo "开始批量处理..."
for file in audio/*.wav; do
    # 并发发送请求
    curl -X POST http://api/v1/asr -F "files=@$file" &
    curl -X POST http://api/v1/vad -F "file=@$file" &
    curl -X POST http://api/v1/pri -F "file=@$file" &
done

# 等待所有请求完成
wait

echo "批量处理完成"

# 3. 模型会在 5 分钟后自动卸载（无需手动操作）
```

### Python 客户端示例

```python
import asyncio
import aiohttp
from pathlib import Path

async def process_batch(audio_files):
    async with aiohttp.ClientSession() as session:
        # 1. 预热模型
        print("预热模型...")
        async with session.post("http://api/v1/warmup") as resp:
            result = await resp.json()
            print(f"预热完成: {result['elapsed']:.1f}s")
        
        # 2. 并发处理
        print(f"开始处理 {len(audio_files)} 个文件...")
        tasks = []
        for file_path in audio_files:
            # 创建并发任务
            tasks.append(process_file(session, file_path))
        
        results = await asyncio.gather(*tasks)
        print(f"处理完成: {len(results)} 个文件")
        
        return results

async def process_file(session, file_path):
    with open(file_path, 'rb') as f:
        data = aiohttp.FormData()
        data.add_field('files', f, filename=file_path.name)
        
        async with session.post("http://api/v1/asr", data=data) as resp:
            return await resp.json()

# 使用示例
audio_files = list(Path("audio").glob("*.wav"))
asyncio.run(process_batch(audio_files))
```

## 性能调优

### 1. 调整并发数

根据 CPU 核心数调整 `MAX_CONCURRENT_REQUESTS`：

```bash
# 64 核 CPU
MAX_CONCURRENT_REQUESTS=64  # 可以设置为核心数或 2×核心数

# 32 核 CPU
MAX_CONCURRENT_REQUESTS=32

# 16 核 CPU
MAX_CONCURRENT_REQUESTS=16
```

### 2. 调整空闲超时

根据任务频率调整 `MODEL_IDLE_TIMEOUT`：

```bash
# 任务间隔短（<5分钟）
MODEL_IDLE_TIMEOUT=600  # 10 分钟

# 任务间隔长（>30分钟）
MODEL_IDLE_TIMEOUT=300  # 5 分钟（默认）

# 测试环境（不自动卸载）
ENABLE_AUTO_UNLOAD=false
```

### 3. 监控和调试

查看实时日志：

```bash
# Docker Stack
docker service logs -f sensevoice_api

# 关键日志标识
# ⏳ 模型加载中
# ✅ 模型加载完成
# 🗑️ 模型卸载
# 🔥 预热中
```

## 常见问题

### Q1: 首次请求很慢怎么办？

**A**: 使用预热端点 `/api/v1/warmup`，在批量任务开始前调用。

### Q2: 如何知道模型是否已加载？

**A**: 调用 `/api/v1/status` 查看 `models.*.is_loaded` 字段。

### Q3: 可以禁用自动卸载吗？

**A**: 可以，设置 `ENABLE_AUTO_UNLOAD=false`。但这会导致模型常驻内存。

### Q4: 单进程性能够吗？

**A**: PyTorch 推理时会释放 GIL，多线程性能接近多进程。建议先测试实际吞吐量。

### Q5: 如何回退到原来的多进程模式？

**A**: 修改环境变量：

```bash
UVICORN_WORKERS=4
ENABLE_AUTO_UNLOAD=false
```

## 性能测试

### 压测命令

```bash
# 使用 Apache Bench
ab -n 100 -c 10 -p audio.wav -T "multipart/form-data; boundary=----WebKitFormBoundary" \
   http://api/v1/asr

# 使用 wrk
wrk -t4 -c10 -d30s --script=upload.lua http://api/v1/asr
```

### 预期性能

- **吞吐量**：接近多进程方案（80-90%）
- **延迟**：
  - 冷启动（首次）：30-60秒（加载模型）
  - 预热后：与多进程相当
  - 稳定运行：与多进程相当

## 进一步优化（可选）

### 1. 模型量化

减少模型大小和加载时间：

```python
# 在 load_sensevoice_model() 中添加
model = torch.quantization.quantize_dynamic(
    model, {torch.nn.Linear}, dtype=torch.qint8
)
```

**效果**：内存减少 60-75%，加载时间减少 50%

### 2. 服务分离

如果某些端点使用频率很低，可以拆分为独立服务：

```yaml
services:
  asr-service:  # 高频
    environment:
      - UVICORN_WORKERS=1
  
  vad-service:  # 低频
    deploy:
      replicas: 0  # 按需启动
```

## 总结

- ✅ **空闲内存减少 92%**（47GB → 3.5GB 或 100MB）
- ✅ **适合低频高并发批量处理**
- ✅ **使用预热机制避免冷启动**
- ✅ **保持高并发处理能力**
- ⚠️ **首次请求需要加载时间**（可通过预热解决）
- ⚠️ **需要测试实际吞吐量**（理论上接近多进程）
