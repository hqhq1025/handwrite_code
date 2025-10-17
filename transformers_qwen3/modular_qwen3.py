# 用于标注某个参数/属性 时刻调用对象
from collections.abc import Callable
from typing import Optional

import torch

# Transformers 提供的缓存抽象，统一管理 KV cache 和其它中间状态。
from ...cache_utils import Cache
# 封装 flash attention 前向的可选参数
from ...modeling_flash_attention_utils import FlashAttentionKwargs
#自回归语言模型的标准输出结构，包含 logits、past_key_values 等字段，Qwen3的 forward 会返回它
from ...modeling_outputs import CausalLMOutputWithPast
# 注册好的注意力实现集合
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS
# 描述流水线/分布式场景下输出的解包方式
from ...processing_utils import Unpack
# 用于限定/校验模型 forward 可以接受的关键字参数集合
from ...utils import TransformersKwargs, logging
# 重用 Gemma 模型的前馈层实现，Qwen3 的 MLP 直接基于它
from ..gemma.modeling_gemma import GemmaMLP
# 使用 LLaMA 的注意力模块作为基础，Qwen3 在此之上可能做少量改动或参数配置
from ..llama.modeling_llama import (
    LlamaAttention,
)
# 接复用 Qwen2 的核心实现，然后在 Qwen3 模型里加上必要的配置/微调
from ..qwen2.modeling_qwen2 import (
    Qwen2DecoderLayer,
    Qwen2ForCausalLM,
    Qwen2ForQuestionAnswering,
    Qwen2ForSequenceClassification,
    Qwen2ForTokenClassification,
    Qwen2Model,
    Qwen2PreTrainedModel,
    Qwen2RMSNorm,
    apply_rotary_pos_emb,
    eager_attention_forward,
)
#同目录下定义的配置类，供这些模型构造时加载默认超参
from .configuration_qwen3 import Qwen3Config

# 以当前模块命名的日志器, 走 Transformers 统一的日志系统
logger = logging.get_logger(__name__)
# 默认权重名称
_CHECKPOINT_FOR_DOC = "Qwen3/Qwen3-8B"

#以下两个是占位子类, 完全用父类实现, 但暴露成 Qwen3 的名字
class Qwen3RMSNorm(Qwen2RMSNorm):
    pass

class Qwen3MLP(GemmaMLP):
    pass

# 继承 LLaMA 的注意力实现, 并进行适配
class Qwen3Attention(LlamaAttention):
    
    def __init__(self, config: Qwen3Config, layer_idx:int):
        super().__init__(config, layer_idx)
        # 对q和k 做 rmsnorm, 维持尺度稳定, 使得 Softmax 更可控
        self.q_norm = Qwen3RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.k_norm = Qwen3RMSNorm(config.head_dim, eps=config.rms_norm_eps)
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None
        
    def forward(
        self,
        hidden_states:torch.Tensor, # 输入的隐状态张量, 形状(batch, seq_len, hidden_size)
        position_embeddings:tuple[torch.Tensor, torch.Tensor], # rope需要的正余弦位置编码, 元组(cos, sin)
        attention_mask: Optional[torch.Tensor], # 掩码张量, 用来屏蔽 padding 或者未来的 token
        past_key_values:Optional[Cache] = None, # 缓存对象, 用于推理时传入上一轮的 KV cache
        cache_position: Optional[torch.LongTensor] = None, # 指定缓存写入/读取的序号
        **kwargs: Unpack[FlashAttentionKwargs], # 额外的flash attention相关参数
    ) -> tuple[torch.Tensor, Optional[Cache]]: #返回值是一个元组(attn_output, attn_weights), 前者是经过注意力后的隐状态, 后者是可选的注意力权重
        
        # 取除了最后一维之外的所有维度, 这里是(batch, seq_len)
        input_shape = hidden_states.shape[:-1]
        # 把原先的 hidden_size 改成(-1, head_dim)
        hidden_shape = (*input_shape, -1, self.head_dim)
        
        # 隐状态乘以 Q 投影权重，得到形状 (batch, seq_len, hidden_size) 的查询向量；
        # .view(hidden_shape) 重排成 (batch, seq_len, num_heads, head_dim)；
        # self.q_norm(...) 对每个头的向量做RMSNorm；
        # 最后 .transpose(1, 2) 交换维度，得到 (batch, num_heads, seq_len, head_dim)
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1,2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1,2)
        # 对v 制作 proj 和 reshape, 不做 RMSNorm
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1,2)
        
        # 取出rope 所需的余弦/正弦张量
        cos, sin = position_embeddings
        # 调用函数对q,k 施加旋转位置编码, 注入位置信息
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)
        
        # 如果传入了上一轮缓存, 就把本步的 rope 相关信息传进去
        if past_key_values is not None:
            cache_kwargs = {
                "sin":sin, "cos":cos, "cache_position":cache_position, 
            }
            # 把当前算出的kv写进缓存中(按 layer_idx 分层保存)
            key_states, value_states = past_key_values.update(key_states, value_states,self.layer_idx, cache_kwargs)

        # 把注意力前向函数默认指向 eager...(常规逐步实现)
        attention_interface : Callable = eager_attention_forward
        # 如果不是 eager 模式, 就按照配置选择最合适的注意力内核
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]
        
        attn_output, attn_weights = attention_interface(
            self, # 传入当前注意力层
            query_states,
            key_states,
            value_states,
            attention_mask,
            # 在推理时强制 0, 训练时用配置的 attention_dropout
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,
            **kwargs,
        )
        
        # 把 (batch, num_heads, seq_len, head_dim) 摊回 (batch, seq_len,hidden_size)；
        # contiguous() 保证内存布局连续
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        # 输出线性层，把多头结果合成一份新的隐状态
        attn_output = self.o_proj(attn_output)
        
        # 隐状态交给后续 MLP，注意力权重按需提供（默认 None）
        return attn_output, attn_weights
        