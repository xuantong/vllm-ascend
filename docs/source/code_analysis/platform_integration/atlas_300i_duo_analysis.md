# Atlas 300I Duo 适配分析

本文档详细分析 vLLM Ascend 项目中 Atlas 300I Duo (ASCEND310P3) 的适配情况和模型装载流程。

## 概述

Atlas 300I Duo 是华为昇腾 310P 系列推理芯片，在 vLLM Ascend 中被标记为**实验性支持**。该设备基于 ASCEND310P3 芯片，需要特殊的优化策略以确保最佳性能。

## 硬件特征识别

### SOC 版本检测

项目通过多层检测机制识别 Atlas 300I Duo：

```python
# vllm_ascend/utils.py:65
def is_310p():
    global _IS_310P
    if _IS_310P is None:
        from vllm_ascend import _build_info
        _IS_310P = _build_info.__soc_version__.lower().startswith("ascend310p")
    return _IS_310P
```

**检测流程**：
1. **编译时检测**: 通过 `SOC_VERSION` 环境变量设置 (通常为 `ASCEND310P3`)
2. **构建信息生成**: setup.py 在构建时生成 `_build_info.py` 文件
3. **运行时判断**: `is_310p()` 函数根据构建信息判断当前平台

### 环境配置要求

在 Atlas 300I Duo 上构建需要：

```dockerfile
# Dockerfile.310p
export SOC_VERSION=ASCEND310P3
export COMPILE_CUSTOM_KERNELS=1  # 必须启用自定义内核
```

## 平台初始化流程

### 1. Worker 初始化

```python
# vllm_ascend/worker/worker_v1.py
def init_worker_ascend():
    # 注册自定义算子
    register_ascend_customop(vllm_config)

    # 初始化昇腾配置和 SOC 版本
    init_ascend_config(vllm_config)
    init_ascend_soc_version()
```

### 2. 模型运行器配置

```python
# vllm_ascend/worker/model_runner_v1.py
if is_310p():
    torch_npu.npu.set_compile_mode(jit_compile=False)  # 禁用 JIT 编译
    ACL_FORMAT = ACL_FORMAT_FRACTAL_NZ  # 使用 NZ 格式
else:
    ACL_FORMAT = ACL_FORMAT_FRACTAL_ND  # 使用 ND 格式
```

## 模型装载流程

### 1. 权重格式适配

Atlas 300I Duo 对矩阵格式有特殊要求：

```python
# vllm_ascend/torchair/models/torchair_pangu_moe.py
if is_310p() and "head" in name:
    # 在 300I Duo 平台上，ACL_FORMAT_FRACTAL_NZ 比 ACL_FORMAT_FRACTAL_ND
    # 在 matmul 操作中性能更优，手动转换格式
    param.data = torch_npu.npu_format_cast(param.data, ACL_FORMAT_FRACTAL_NZ)
```

### 2. 注意力机制适配

```python
# vllm_ascend/attention/attention_v1.py
if is_310p():
    # 310P 需要将 seq_lens_tensor 转移到设备上
    attn_metadata.seq_lens = attn_metadata.seq_lens.to(device=query.device)

    # 对 q, k, v 输出张量进行16对齐
    query = aligned_16(query)
    key = aligned_16(key)
    value = aligned_16(value)
```

### 3. KV Cache 布局

```python
# vllm_ascend/attention/attention_v1.py
def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
    if is_310p():
        # 310P 使用5维布局优化内存访问
        return (2, num_blocks, num_kv_heads * head_size // 16, block_size, 16)
    else:
        # 其他平台使用4维布局
        return (2, num_blocks, block_size, num_kv_heads * head_size)
```

## Atlas 300I Duo 特定优化

### 1. TorchAir 图配置优化

```python
# vllm_ascend/torchair/torchair_model_runner.py
config = torchair.CompilerConfig()

# 在 300I Duo 上禁用 tiling_schedule_optimize，因为存在已知问题
config.experimental_config.tiling_schedule_optimize = not is_310p()

# 启用其他优化选项
config.experimental_config.frozen_parameter = True
config.experimental_config.enable_view_optimize = True
```

**关键优化点**：
- **禁用平铺调度优化**: 300I Duo 上该功能有 bug
- **启用参数冻结**: 减少图执行时的地址刷新时间
- **启用视图优化**: 提高内存访问效率

### 2. 量化支持

```python
# vllm_ascend/quantization/w8a8.py
if is_310p():
    # 在 300I Duo 平台上，使用 nz 格式时需要额外的转置
    # 该转置在 torchair 中可以跳过
    output = torch_npu.npu_quant_matmul(x, weight, ...)
```

### 3. 分布式通信适配

```python
# vllm_ascend/torchair/torchair_model_runner.py
if is_310p():
    # 300I Duo 平台需要重新应用广播补丁
    # 因为 torchair 中的 patch_for_hcom 会覆盖原有补丁
    from vllm_ascend.patch.platform.patch_common.patch_distributed import \
        communication_adaptation_310p
    communication_adaptation_310p()
```

### 4. MoE 专家数量优化

```python
# vllm_ascend/torchair/models/torchair_pangu_moe.py
# 在 300I Duo 平台上，将 num_voted_experts 设置为 5 可以在
# 不太损失精度的情况下获得良好性能
num_voted_experts = 5 if is_310p() else 8
```

## 测试和验证

### E2E 测试覆盖

项目为 Atlas 300I Duo 提供专门的测试：

```python
# tests/e2e/310p/test_offline_inference_310p.py
MODELS = ["Qwen/Qwen3-0.6B", "Qwen/Qwen2.5-7B-Instruct"]

@pytest.mark.parametrize("model", MODELS)
def test_models(model: str, dtype: str, max_tokens: int):
    with VllmRunner(model,
                    tensor_parallel_size=1,
                    dtype=dtype,
                    max_model_len=2048,
                    enforce_eager=True,
                    compilation_config={
                        "custom_ops": ["none", "+rms_norm", "+rotary_embedding"]
                    }) as vllm_model:
        vllm_model.generate_greedy(example_prompts, max_tokens)
```

### CI/CD 支持

```yaml
# .github/workflows/vllm_ascend_test_310p.yaml
name: '310p e2e test'
runs-on: [linux-aarch64-310p-1, linux-aarch64-310p-4]
container:
  image: cann:8.2.rc1-310p-ubuntu22.04-py3.11
env:
  SOC_VERSION: ASCEND310P3
```

## 性能特征

### 内存优化
- **格式首选**: `ACL_FORMAT_FRACTAL_NZ` 优于 `ACL_FORMAT_FRACTAL_ND`
- **对齐要求**: 张量需要16字节对齐以获得最佳性能
- **KV Cache**: 使用5维布局优化内存访问模式

### 计算优化
- **禁用 JIT**: 避免动态编译开销
- **自定义内核**: 必须启用以获得最佳性能
- **专家选择**: MoE 模型使用较少的专家数量 (5 vs 8)

### 通信优化
- **广播适配**: 需要特殊的分布式通信补丁
- **内存格式**: 量化操作需要额外的格式转换

## 限制和已知问题

1. **实验性支持**: Atlas 300I Duo 仍处于实验阶段
2. **平铺调度**: `tiling_schedule_optimize` 功能被禁用
3. **自定义内核**: 必须启用，增加了构建复杂度
4. **格式转换**: 需要额外的内存格式转换开销

## 最佳实践

1. **环境配置**: 确保正确设置 `SOC_VERSION=ASCEND310P3`
2. **内核编译**: 必须启用 `COMPILE_CUSTOM_KERNELS=1`
3. **模型选择**: 优先使用已验证的模型（如 Qwen 系列）
4. **性能调优**: 根据具体模型调整专家并行和量化策略

---

*该分析基于 vLLM Ascend v0.10.2rc1 版本，随着项目发展可能会有更新*