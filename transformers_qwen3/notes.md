**Qwen3 目录说明**
src/transformers/models/qwen3/__init__.py:1 提供 Qwen3 模块的惰性导入入口，延迟加载配置与建模代码，减小初始导入开销。
src/transformers/models/qwen3/configuration_qwen3.py:1 定义 Qwen3Config，集中描述模型的结构超参、RoPE 配置、并行计划等，可用于实例化或自定义模型。
src/transformers/models/qwen3/modular_qwen3.py:1 为人工维护的模块化源码，基于 Qwen2、Llama、Gemma 的组件组合出 Qwen3，并调整注意力与 MLP 细节；执行 make fixup 会据此生成完整建模文件。
src/transformers/models/qwen3/modeling_qwen3.py:1 是由模块化文件自动生成的 PyTorch 实现，包含所有前向逻辑与任务头；如需修改应编辑 modular_qwen3.py 后再重新生成。







**学习顺序建议**
先看 configuration_qwen3.py；从 Qwen3Config 入手理解所有超参含义、默认值和并行计划，心里有框架再看实现更轻松。
接着读 modular_qwen3.py；这是人工维护的源文件，整合了 Qwen2/Llama/Gemma 的组件。建议一边对照这些父类（qwen2,llama,gemma），一边追踪 Qwen3 改动在哪里。
当对 modular 版理解透彻后，再翻 modeling_qwen3.py，确认自动生成后的完整实现；关注它如何展开 modular 中的继承以及新增的前向逻辑。
最后回到 __init__.py，理解懒加载机制和模块导出结构；这部分读完就更清楚 transformer 在导入时的行为。