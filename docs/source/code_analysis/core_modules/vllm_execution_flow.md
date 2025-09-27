# vLLM Ascend 完整执行流程分析

本文档详细分析 vLLM Ascend 在 Atlas 300I Duo 平台上的完整代码执行流程，重点关注关键路径和特定优化。

## 执行流程概览

```mermaid
graph TD
    A[用户调用 LLM()] --> B[插件注册与平台检测]
    B --> C[NPU Platform 初始化]
    C --> D[Worker 进程启动]
    D --> E[模型加载与权重初始化]
    E --> F[内存分配与 KV Cache 初始化]
    F --> G[推理执行循环]
    G --> H[Scheduler 调度]
    H --> I[ModelRunner 执行]
    I --> J[Attention 计算]
    J --> K[结果返回]

    subgraph "Atlas 300I Duo 特定处理"
        B1[SOC 版本检测]
        C1[平台配置适配]
        E1[权重格式转换]
        F1[KV Cache 5维布局]
        I1[TorchAir 图优化]
        J1[310P 注意力适配]
    end

    B --> B1
    C --> C1
    E --> E1
    F --> F1
    I --> I1
    J --> J1
```

## 详细执行流程

### 1. 系统启动与插件注册

#### 1.1 入口点注册
```python
# setup.py:392-396
entry_points={
    "vllm.platform_plugins": ["ascend = vllm_ascend:register"],
    "vllm.general_plugins": ["ascend_enhanced_model = vllm_ascend:register_model"],
}
```

**执行路径**: `vllm_ascend/__init__.py:19`
```python
def register():
    """Register the NPU platform."""
    return "vllm_ascend.platform.NPUPlatform"
```

#### 1.2 平台检测与初始化
**核心节点**: [平台检测与初始化流程](platform_detection_flow.md)

执行步骤：
1. **环境变量检测**: `SOC_VERSION=ASCEND310P3`
2. **构建信息生成**: `_build_info.py` 文件创建
3. **运行时判断**: `utils.py:65` - `is_310p()` 函数

```python
# vllm_ascend/utils.py:65-70
def is_310p():
    global _IS_310P
    if _IS_310P is None:
        from vllm_ascend import _build_info
        _IS_310P = _build_info.__soc_version__.lower().startswith("ascend310p")
    return _IS_310P
```

### 2. 平台初始化与配置

#### 2.1 NPUPlatform 配置
**位置**: `platform.py:121-144`

**Atlas 300I Duo 特定配置**:
```python
# platform.py:65-67
@classmethod
def pre_register_and_update(cls, parser=None):
    from vllm_ascend.utils import adapt_patch
    adapt_patch(is_global_patch=True)
```

#### 2.2 配置检查与更新
**核心节点**: [平台配置检查流程](platform_config_check.md)

关键检查项：
- V1 引擎强制使用
- 昇腾配置初始化
- 编译配置适配

### 3. Worker 进程初始化

#### 3.1 NPUWorker 创建
**位置**: `worker/worker_v1.py:67-90`

**初始化序列**:
1. **设备检测**: `_init_device()` - `worker_v1.py:163`
2. **分布式环境**: `_init_worker_distributed_environment()`
3. **随机种子**: `NPUPlatform.seed_everything()`
4. **模型运行器**: `NPUModelRunner(self.vllm_config, device)`

```python
# worker_v1.py:174-177
def init_device(self):
    device = self._init_device()
    self.model_runner = NPUModelRunner(self.vllm_config, device)
```

#### 3.2 昇腾特定初始化
**位置**: `worker_v1.py:88-94`

**Atlas 300I Duo 专用**:
```python
register_ascend_customop(vllm_config)
init_ascend_config(vllm_config)
init_ascend_soc_version()  # 关键：SOC 版本初始化
```

### 4. 模型加载与权重初始化

#### 4.1 模型加载流程
**核心节点**: [模型加载详细流程](model_loading_flow.md)

**位置**: `worker_v1.py:212-225`

```python
def load_model(self) -> None:
    if self.vllm_config.model_config.enable_sleep_mode:
        allocator = CaMemAllocator.get_instance()
        # Sleep mode 特殊处理

    with context:
        self.model_runner.load_model()
```

#### 4.2 Atlas 300I Duo 权重格式适配
**位置**: `torchair/models/torchair_pangu_moe.py`

```python
if is_310p() and "head" in name:
    # 300I Duo 平台上，ACL_FORMAT_FRACTAL_NZ 比 ACL_FORMAT_FRACTAL_ND 性能更优
    param.data = torch_npu.npu_format_cast(param.data, ACL_FORMAT_FRACTAL_NZ)
```

#### 4.3 模型运行器配置
**位置**: `model_runner_v1.py`

**Atlas 300I Duo 特定设置**:
```python
if is_310p():
    torch_npu.npu.set_compile_mode(jit_compile=False)  # 禁用 JIT
    ACL_FORMAT = ACL_FORMAT_FRACTAL_NZ                # 使用 NZ 格式
else:
    ACL_FORMAT = ACL_FORMAT_FRACTAL_ND                # 使用 ND 格式
```

### 5. 内存管理与 KV Cache 初始化

#### 5.1 内存分配
**核心节点**: [内存管理详细流程](memory_management_flow.md)

**位置**: `worker_v1.py:179-199`

```python
def determine_available_memory(self) -> int:
    NPUPlatform.clear_npu_memory()
    _, total_npu_memory = NPUPlatform.mem_get_info()
    self.model_runner.profile_run()
    free_npu_memory, _ = NPUPlatform.mem_get_info()
```

#### 5.2 KV Cache 布局 - Atlas 300I Duo 特殊处理
**位置**: `attention/attention_v1.py`

```python
def get_kv_cache_shape(num_blocks, block_size, num_kv_heads, head_size):
    if is_310p():
        # 310P 使用 5 维布局优化内存访问
        return (2, num_blocks, num_kv_heads * head_size // 16, block_size, 16)
    else:
        # 其他平台使用 4 维布局
        return (2, num_blocks, block_size, num_kv_heads * head_size)
```

#### 5.3 CaMem 分配器（Sleep Mode）
**位置**: `device_allocator/camem.py`

专用于 Sleep Mode 的内存管理，支持内存休眠和恢复。

### 6. 推理执行循环

#### 6.1 执行入口
**核心节点**: [推理执行详细流程](inference_execution_flow.md)

**位置**: `worker_v1.py:260-275`

```python
def execute_model(self, scheduler_output, intermediate_tensors):
    # 数据并行同步
    # 输入批次准备
    # 模型执行
    output = self.model_runner.execute_model(scheduler_output, intermediate_tensors)
```

#### 6.2 ModelRunner 执行
**位置**: `model_runner_v1.py`

**执行步骤**:
1. **输入准备**: `_prepare_inputs()`
2. **批次同步**: `_sync_metadata_across_dp()`
3. **模型前向**: `_execute_model_forward()`

#### 6.3 Atlas 300I Duo 特定优化
**位置**: `model_runner_v1.py`

**MoE 通信方法选择**:
```python
def _select_moe_comm_method(self, num_tokens, with_prefill):
    soc_version = get_ascend_soc_version()
    if soc_version in {AscendSocVersion.A2}:
        # A2 平台策略
    elif soc_version in {AscendSocVersion.A3}:
        # A3 平台策略
    else:
        raise ValueError(f"Unsupported soc_version: {soc_version}")
```

### 7. Attention 计算 - Atlas 300I Duo 适配

#### 7.1 注意力机制适配
**核心节点**: [Attention 计算流程](attention_computation_flow.md)

**位置**: `attention/attention_v1.py`

**Atlas 300I Duo 特殊处理**:
```python
if is_310p():
    # seq_lens_tensor 需要转移到设备上
    attn_metadata.seq_lens = attn_metadata.seq_lens.to(device=query.device)

    # 对 q, k, v 输出张量进行 16 对齐
    query = aligned_16(query)
    key = aligned_16(key)
    value = aligned_16(value)
```

#### 7.2 TorchAir 图优化
**位置**: `torchair/torchair_model_runner.py`

```python
config = torchair.CompilerConfig()
# 300I Duo 上禁用 tiling_schedule_optimize（已知 bug）
config.experimental_config.tiling_schedule_optimize = not is_310p()
config.experimental_config.frozen_parameter = True
```

### 8. 结果处理与返回

#### 8.1 输出处理
**位置**: `worker_v1.py`

执行完成后的输出处理和结果返回。

#### 8.2 异步处理支持
**位置**: `model_runner_v1.py:180-220`

支持异步输出拷贝以提高性能。

## Atlas 300I Duo 特定 TODO 清单

基于代码分析，以下是 Atlas 300I Duo 平台的关键待办事项：

### ✅ 已完成项目

1. **SOC 版本检测机制** - `utils.py:65`
2. **平台特定配置适配** - `platform.py`
3. **权重格式转换** - `torchair/models/`
4. **KV Cache 5维布局** - `attention/attention_v1.py`
5. **TorchAir 图优化配置** - `torchair/torchair_model_runner.py`

### ⚠️ 已知限制

1. **平铺调度优化禁用** - `tiling_schedule_optimize = False`
2. **JIT 编译禁用** - `jit_compile = False`
3. **MoE 专家数量限制** - `num_voted_experts = 5`

### 🔄 持续优化项目

1. **分布式通信补丁** - 需要特殊的广播适配
2. **量化操作适配** - 需要额外的格式转换
3. **内存对齐优化** - 16 字节对齐要求
4. **测试覆盖扩展** - E2E 测试持续完善

## 性能关键路径

### 关键性能节点

1. **[平台检测](platform_detection_flow.md)** - 启动时一次性开销
2. **[模型加载](model_loading_flow.md)** - 权重格式转换开销
3. **[内存管理](memory_management_flow.md)** - KV Cache 布局优化
4. **[推理执行](inference_execution_flow.md)** - 主要性能瓶颈
5. **[Attention 计算](attention_computation_flow.md)** - 计算密集型操作

### 优化建议

1. **内存预分配**: 减少运行时内存分配开销
2. **批次优化**: 充分利用 Atlas 300I Duo 的并行能力
3. **格式缓存**: 缓存转换后的权重格式
4. **图编译优化**: 平衡编译开销与执行性能

---

*该文档基于 vLLM Ascend v0.10.2rc1 版本分析，涵盖了 Atlas 300I Duo 平台的完整执行路径*