
#用来定义Qwen3配置对象, 做层类型校验
from ...configuration_utils import PretrainedConfig, layer_type_validation
#验证与rope相关的配置项
from ...modeling_rope_utils import rope_config_validation
#封装了日志设置, 用于后面控制日志
from ...utils import logging

#创建一个日志对象, 后续通过其打印调试信息或警告
logger = logging.get_logger(__name__)

# 定义Qwen3模型的配置对象, 继承PreTrainedConfig类, 以具备预训练模型通用的配置接口
# 类中会声明默认超参数
class Qwen3Config(PretrainedConfig):
    
    # 模型家族名, 用于反向找到对应的模型和分词器
    module_type = "Qwen3"
    # 推理阶段自动丢掉掉past_key_values(缓存的注意力 kv), 节省带宽和内存
    # 这里跟kvcache不是一回事, 这里代表 kv 缓存不会公开给最终的推理结果
    keys_to_ignore_at_inference = ["past_key_values"]
    
    # 张量并行时各权重的拆分方式
    # colwise表示列并行, rowwise表示行并行
    base_model_tp_plan = {
        #layers.* 的 * 是任意层索引，所以 plan 会作用于所有 Transformer 层
        
        # 这里没太搞懂
        "layers.*.self_attn.q_proj": "colwise",
        "layers.*.self_attn.k_proj": "colwise",
        "layers.*.self_attn.v_proj": "colwise",
        "layers.*.self_attn.o_proj": "rowwise",
        "layers.*.mlp.gate_proj": "colwise",
        "layers.*.mlp.down_proj": "rowwise",
        "layers.*.mlp.up_proj": "colwise",
    }
    
    # pipeline 并行的输入/输出签名
    base_model_pp_plan = {
        # 这里括号中的每个元素是一个元组, 元素的顺序是输入和输出的签名
        # 用于表明每段的数据接口, 以衔接各段
        "embed_tokens": (["input_ids"], ["input_embeds"]),
        "layers":(["hidden_states","attention_mask"], ["hidden_states"]),
        "norm":(["hidden_states"], ["hidden_states"])
    }
    
    def __init__(
        self,
        # 这里关于各个维度还是有点蒙圈, 后面再看下
        
        vocab_size=151936, # 词表大小
        hidden_size=4096, # 隐维度大小
        intermediate_size=22016, # 前馈层中间维度
        num_hidden_layers=32, # Transformer 块的数量
        num_attention_heads=32, # 总注意力头数
        num_key_value_heads=32, # KV头数
        head_dim=128, # 单个注意力头的维度
        hidden_act="silu", # 前馈层激活函数
        max_position_embeddings=32768, # rope编码的最大长度
        initializer_range=0.02, # 权重初始化标准差
        rms_norm_eps=1e-6, # RMSNorm 的 epsilon
        use_cache=True, # 推理时是否使用KV缓存
        
        tie_word_embeddings=False, # 是否共享输出/输入嵌入, 这里是使用两份独立权重
        # 查了一下, 小模型共用词表, 提升泛化能力;大模型参数足以学习更专门化的输入输出表示
        
        rope_theta=10000.0, # rope 角频率基准
        rope_scaling=None, # rope 动态扩展配置
        attention_bias=False, # 是否使用注意力偏置
        use_sliding_window=False, # 是否使用滑动窗口注意力
        # 虽然默认禁用滑窗, 但是下面的设置是为了兼容开启滑窗的场景
        sliding_window=4096, # 滑动窗口大小
        max_window_layers=28, # 滑窗层数上限
        layer_types=None, # 混合层类型
        attention_dropout=0.0, # 注意力的 dropout
        **kwargs, # 透传其他配置到基类
    ):
        
        self.vocab_size = vocab_size
        self.max_position_embeddings = max_position_embeddings
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.num_hidden_layers = num_hidden_layers
        self.num_attention_heads = num_attention_heads
        self.use_sliding_window = use_sliding_window
        self.sliding_window = sliding_window if self.use_sliding_window else None
        self.max_window_layers = max_window_layers

        # for backward compatibility
        if num_key_value_heads is None:
            num_key_value_heads = num_attention_heads

        self.num_key_value_heads = num_key_value_heads
        self.head_dim = head_dim
        self.hidden_act = hidden_act
        self.initializer_range = initializer_range
        self.rms_norm_eps = rms_norm_eps
        self.use_cache = use_cache
        self.rope_theta = rope_theta
        self.rope_scaling = rope_scaling
        self.attention_bias = attention_bias
        self.attention_dropout = attention_dropout
        # Validate the correctness of rotary position embeddings parameters
        # BC: if there is a 'type' field, move it to 'rope_type'.
        if self.rope_scaling is not None and "type" in self.rope_scaling:
            self.rope_scaling["rope_type"] = self.rope_scaling["type"]
        rope_config_validation(self)

        self.layer_types = layer_types
        if self.layer_types is None:
            self.layer_types = [
                "sliding_attention"
                if self.sliding_window is not None and i >= self.max_window_layers
                else "full_attention"
                for i in range(self.num_hidden_layers)
            ]
        layer_type_validation(self.layer_types, self.num_hidden_layers)

        super().__init__(
            tie_word_embeddings=tie_word_embeddings,
            **kwargs,
        )    
        
# 定义模块的公开接口
__all__ = ["Qwen3Config"]