# 这个文件其实是根据 modular_qwen3.py 自动生成的 很神奇

# 类型注释
from collections.abc import Callable
from typing import Optional,Union

# 张量和神经网络
import torch
from torch import nn

# 按字符串查找激活函数
from ...activations import ACT2FN
# 推理时管理kv缓存
from ...cache_utils import Cache, DynamicCache
# 集成后获得generate()等通用生成接口
from ...generation import GenerationMixin
# 在流式推理中可从 hub 下载高效内核替换 forward
from ...integrations import use_kernel_forward_from_hub
# 构造自回归和滑窗注意力的掩码
from ...masking_utils import create_causal_mask, create_sliding_window_causal_mask
# flash attention 前向的附加参数
from ...modeling_flash_attention_utils import FlashAttentionKwargs
# 提供通用的问答/分类头, 激活函数检查点, 用于节省内存
from ...modeling_layers import (
    GenericForQuestionAnswering,
    GenericForSequenceClassification,
    GenericForTokenClassification,
    GradientCheckpointingLayer,
)
# 标准化的返回结构
from ...modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast
# 管理rope 的初始化和动态扩展逻辑
from ...modeling_rope_utils import ROPE_INIT_FUNCTIONS, dynamic_rope_update
# 注意力实现注册表, 以及所有预训练模型的基类
from ...modeling_utils import ALL_ATTENTION_FUNCTIONS, PreTrainedModel
# 类型工具, 用于拆包kwargs
from ...processing_utils import Unpack
# 限制/记录模型forward允许的关键字参数, 用于做类型提示和校验
from ...utils import TransformersKwargs, auto_docstring, can_return_tuple
# 输入校验器, 检查输入的张量/参数
from ...utils.generic import check_model_inputs
# 配置类, 提供超参和设置
from .configuration_qwen3 import Qwen3Config

# 允许在推理时从 HuggingFace Hub 下载专用的高性能实现，自动替换这个类的 forward
@use_kernel_forward_from_hub("Qwen3")
class Qwen3Model(nn.Module):
    
    def __init__(self, hidden_size, eps: float = 1e-6) -> None:
        super().__init__()
        # 定义可训练参数 weight
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps
        
    # 对输入的hidden_states进行归一化处理
    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        input_dtype = hidden_states.dtype
        # 提升精度, 避免 norm 时溢出/失精
        hidden_states = hidden_states.to(torch.float32)
        # 做rmsnorm计算
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        # 转换为原有精度
        return self.weight * hidden_states.to(input_dtype)
    
    # PyTorch 模块的一个“显示钩子”，用来自定义打印层结构时显示的额外信息。
    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"
    

# 前馈层, 使用SwiGLU
class Qwen3MLP(nn.Module):
    
    def __init__(self, config):
        super().__init__()
        # 模型配置
        self.config = config
        # 隐藏层大小/前馈层输入维度
        self.hidden_size = config.hidden_size
        # 中间层大小/前馈层输出维度
        self.intermediate_size = config.intermediate_size
        # 门控分支, 输出门控信号
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # 内容分支
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        # 投射回主维度
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)
        # 从配置中获取指定的激活参数'silu'
        self.act = ACT2FN[config.hidden_act]
        
    def forward(self,x):
        # 门控信号经过激活函数后, 与内容信号相乘, 最后投射回主维度
        down_proj = self.down_proj(self.act(self.gate_proj(x)) * self.up_proj(x))
        return down_proj
    
    # 用于把每个 head 的奇偶维度做复数形式的旋转
    # 这里还是有点没太懂
def rotate_half(x):
    
    # x1取前半部分
    x1 = x[..., : x.shape[-1] // 2]
    # x2取后半部分
    x2 = x[..., x.shape[-1] // 2 :]
    
    # 把每对 (x1, x2) 映射到 (-x2, x1)，也就是标准的 90° 旋转
    return torch.cat((-x2, x1), dim=-1)

# 将 rope 融合进 qk 之中
# q,k形状是(batch_size, num_heads, seq_len, head_dim)
# cos,sin 形状是(batch_size, seq_len, head_dim)
def apply_rotary_pos_emb(q, k, cos, sin, position_ids=None, unsqueeze_dim=1):
    # 为了让cos/sin 和 q/k 做逐元素惩罚, 需要通过 unsqueeze 在 num_heads 那一维补一个尺寸 1
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    # cos 部分保留原始分量, sin 部分做旋转, 然后相加
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    
    return q_embed, k_embed

# 用于把 KV 张量复制到完整的注意力头数, 支持多查询注意力场景(MQA/MHA)
# 这里还是有点懵, 跟MQA相关, 后面再看看
# 这里的维度变化似乎没太搞懂
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    
    # 表示每个KV头需要重复多少次才能覆盖全部注意力头
    if n_rep == 1:
        return hidden_states
    
    # 大于1次时, 先插入额外维度, 每个KV头被重复n次, 但是共享内存
    hidden_states = hidden_states[:, :, None, :, :].expand(batch, num_key_value_heads, n_rep, slen, head_dim)
    
    # 把二三维合成, 返回
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)

# 朴素版多头注意力实现
def eager_attention_forward(
    module: nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Optional[torch.Tensor],
    scaling: float,
    dropout: float = 0.0,
    **kwargs: Unpack[TransformersKwargs],
):
    # 
    key_states = repeat_kv(key, module.num_key_value_groups)
    value_states = repeat_kv(value, module.num_key_value_groups)
    
    # 计算得到注意力logits, 乘以缩放系数(1/sqrt(head_dim))
    attn_weights = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    
    # 如果传入掩码
    if attention_mask is not None:
        # 将注意力掩码裁剪到当前 k_len
        causal_mask = attention_mask[:, :, :, : key_states.shape[-2]]
        # 将mask叠加到logits, 屏蔽不允许关注的位置
        attn_weights = attn_weights + causal_mask

    # 沿最后维对 logits 做 Softmax, 得到注意力概率
    attn_weights = nn.functional.softmax(attn_weights, dim=-1, dtype=torch.float32).to(query.dtype)
    # 训练时按设定概率丢弃部分权重, 实现注意力dropout; 推理时不丢弃
    attn_weights = nn.functional.dropout(attn_weights, p=dropout, training=module.training)
    # 将注意力按照概率作为权重汇聚 V
    attn_output = torch.matmul(attn_weights, value_states)
    # 把维度整理成(batch, q_len, heads, head_dim)
    attn_output = attn_output.transpose(1, 2).contiguous()

    # 返回注意力输出和注意力权重
    return attn_output, attn_weights


class Qwen3Attention(nn.Module):
    
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        # 保存配置和层索引
        self.config = config
        self.layer_idx = layer_idx
        # 如果没有给出head_dim, 则计算出head_dim
        self.head_dim = getattr(config, "head_dim", config.hidden_size // config.num_attention_heads)
        # 用于多查询注意力
        self.num_key_value_groups = config.num_attention_heads // config.num_key_value_heads
        # 计算注意力logits 时用于缩放的系数
        self.scaling = self.head_dim**-0.5
        # dropout的超参数
        self.attention_dropout = config.attention_dropout
        # 用于表示这个注意力层是否是"因果型"的
        self.is_causal = True

        # 构建线性层, 完成多投注意力的投影
        self.q_proj = nn.Linear(
            config.hidden_size, config.num_attention_heads * self.head_dim, bias=config.attention_bias
        )
        self.k_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.v_proj = nn.Linear(
            config.hidden_size, config.num_key_value_heads * self.head_dim, bias=config.attention_bias
        )
        self.o_proj = nn.Linear(
            config.num_attention_heads * self.head_dim, config.hidden_size, bias=config.attention_bias
        )
        
        # 对 qk 做 rmsnorm
        self.q_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # unlike olmo, only on the head dim!
        self.k_norm = Qwen3RMSNorm(self.head_dim, eps=config.rms_norm_eps)  # thus post q_norm does not need reshape
        # 如果这个层是滑窗型, 则使用滑窗注意力
        self.sliding_window = config.sliding_window if config.layer_types[layer_idx] == "sliding_attention" else None

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_embeddings: tuple[torch.Tensor, torch.Tensor],
        attention_mask: Optional[torch.Tensor],
        past_key_values: Optional[Cache] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[FlashAttentionKwargs],
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        # 记录输入除 hidden_states 以外的所有维度
        input_shape = hidden_states.shape[:-1]
        # 为 QKV 的 reshape 适配维度, 把最后一维拆成 头数*head_dim
        hidden_shape = (*input_shape, -1, self.head_dim)
        # 通过线性层得到投影, 然后 reshape, 然后 RMSNorm, 最后 transpose
        # 其中 qk 做 norm, v 不做
        query_states = self.q_norm(self.q_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        key_states = self.k_norm(self.k_proj(hidden_states).view(hidden_shape)).transpose(1, 2)
        value_states = self.v_proj(hidden_states).view(hidden_shape).transpose(1, 2)

        # 拆出 RoPE 的余弦/正弦张量
        cos, sin = position_embeddings
        # 给 QK 注入旋转位置编码
        query_states, key_states = apply_rotary_pos_emb(query_states, key_states, cos, sin)

        # 增量推理时更新 KV 缓存
        if past_key_values is not None:
            cache_kwargs = {"sin": sin, "cos": cos, "cache_position": cache_position}
            key_states, value_states = past_key_values.update(key_states, value_states, self.layer_idx, cache_kwargs)

        # 默认使用朴素注意力
        attention_interface: Callable = eager_attention_forward
        if self.config._attn_implementation != "eager":
            attention_interface = ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]

        # 执行注意力前向, 传入相关参数
        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            sliding_window=self.sliding_window,  
            **kwargs,
        )

        # 把注意力输出从 (batch, seq_len,num_heads, head_dim) 
        # 摊回 (batch, seq_len, hidden_size)
        attn_output = attn_output.reshape(*input_shape, -1).contiguous()
        # 通过线性输出层整合所有头
        attn_output = self.o_proj(attn_output)
        
        return attn_output, attn_weights
    
    
# 这是Decoder层, 原生支持梯度检查点功能
class Qwen3DecoderLayer(GradientCheckpointingLayer):
    def __init__(self, config: Qwen3Config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        # 实例化之前定义的注意力子层
        self.self_attn = Qwen3Attention(config=config, layer_idx=layer_idx)
        # 构造前馈网络
        self.mlp = Qwen3MLP(config)
        # 早注意力前和 MLP 前使用 RMSNorm
        self.input_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = Qwen3RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        # 记录这一层的标签
        self.attention_type = config.layer_types[layer_idx]

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        use_cache: Optional[bool] = False,
        cache_position: Optional[torch.LongTensor] = None,
        position_embeddings: Optional[tuple[torch.Tensor, torch.Tensor]] = None,  
        **kwargs: Unpack[TransformersKwargs],
    ) -> torch.Tensor:
        # 将输入保存为 residual, 用于残差相加
        residual = hidden_states
        # 在注意力前做归一化
        hidden_states = self.input_layernorm(hidden_states)

        # 通过注意力机制, 更新 hidden_states
        hidden_states, _ = self.self_attn(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            use_cache=use_cache,
            cache_position=cache_position,
            position_embeddings=position_embeddings,
            **kwargs,
        )
        # 残差连接
        hidden_states = residual + hidden_states

        # 有一个残差, 不过这个是mlp的
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        hidden_states = residual + hidden_states
        
        return hidden_states

# 补齐模型文档中的 docstring
@auto_docstring
# 定义Qwen3模型的公共基类
class Qwen3PreTrainedModel(PreTrainedModel):
    config: Qwen3Config
    # 保存/加载权重时，顶层模块会挂在 state_dict 的 "model.*" 名字空间下
    base_model_prefix = "model"
    # 声明模型内建梯度检查点功能
    supports_gradient_checkpointing = True
    # 用于分布式分片时的黑名单，避免拆解完整的 decoder 层
    _no_split_modules = ["Qwen3DecoderLayer"]
    # 不自动把缓存张量搬设备
    _skip_keys_device_placement = ["past_key_values"]
    # 支持 flash attention 内核
    _supports_flash_attn = True
    # 支持 PyTorch 的 SDPA 实现
    _supports_sdpa = True
    # 支持 Flex Attention
    _supports_flex_attn = True
    # 允许 torch.compile 等工具对整个模型进行优化编译
    _can_compile_fullgraph = True
    # 在设置 model.config._attn_implementation 时可切换不同注意力后端
    _supports_attention_backend = True
    # 在计算隐藏状态或注意力输出时应该挂钩到哪些模块
    _can_record_outputs = {
        "hidden_states": Qwen3DecoderLayer,
        "attentions": Qwen3Attention,
    }
    
    # 装饰器在 forward 前进行统一的输入校验
    @check_model_inputs()
    @auto_docstring
    # Qwen3 主模型的前向接口
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs: Unpack[TransformersKwargs],
    ) -> BaseModelOutputWithPast:
        # 只允许传入inputs_ids或者inputs_embeds 
        if (input_ids is None) ^ (inputs_embeds is not None):
            raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

        # 如果传入ids，则转换为embeds
        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        # 开启缓存, 但是没传现有缓存时, 初始化动态 KV Cache
        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        # 根据已有缓存长度生成本次 token 对应的绝对位置序号
        if cache_position is None:
            past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
            cache_position = torch.arange(
                past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
            )

        # 没给position_ids, 默认用 cache_position
        if position_ids is None:
            position_ids = cache_position.unsqueeze(0)

        # 如果传入的mask不是构建好的dict, 则自己生成掩码
        if not isinstance(causal_mask_mapping := attention_mask, dict):
            # 打包掩码需要的信息
            mask_kwargs = {
                "config": self.config,
                "input_embeds": inputs_embeds,
                "attention_mask": attention_mask,
                "cache_position": cache_position,
                "past_key_values": past_key_values,
                "position_ids": position_ids,
            }
            # 构建标准的因果掩码
            causal_mask_mapping = {
                "full_attention": create_causal_mask(**mask_kwargs),
            }
            # 如果包含滑窗, 则再准备滑窗掩码
            if self.has_sliding_layers:
                causal_mask_mapping["sliding_attention"] = create_sliding_window_causal_mask(**mask_kwargs)

        # 初始化层输入
        hidden_states = inputs_embeds
        # 算出要给所有层复用的RoPE正余弦张量
        position_embeddings = self.rotary_emb(hidden_states, position_ids)

        # 按层执行前向
        for decoder_layer in self.layers[: self.config.num_hidden_layers]:
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=causal_mask_mapping[decoder_layer.attention_type],
                position_ids=position_ids,
                past_key_values=past_key_values,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
                **kwargs,
            )

        # 做输出层归一化
        hidden_states = self.norm(hidden_states)
        
        # 返回结构化结果
        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values if use_cache else None,
        )


@auto_docstring
# 定义语言模型头, 几句呗预训练模型通用能力, 又支持 .generate()
class Qwen3ForCausalLM(Qwen3PreTrainedModel, GenerationMixin):
    # 记录与输入嵌入共享权重的键
    _tied_weights_keys = ["lm_head.weight"]
    # 张量并行方案
    _tp_plan = {"lm_head": "colwise_rep"}
    # 流水线并行接口
    _pp_plan = {"lm_head": (["hidden_states"], ["logits"])}

    def __init__(self, config):
        super().__init__(config)
        # 主干模型
        self.model = Qwen3Model(config)
        # 保存词表
        self.vocab_size = config.vocab_size
        # 语言模型输出层
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # 调用基类的权重初始化/微调
        self.post_init()

    # 装饰器确保用于请求tuple输出时返回预期格式
    @can_return_tuple
    # 继续自动文档填充
    @auto_docstring
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Cache] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs: Unpack[TransformersKwargs],
    ) -> CausalLMOutputWithPast:
        # 调用 Qwen3Model 生成最后隐状态
        outputs: BaseModelOutputWithPast = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            cache_position=cache_position,
            **kwargs,
        )

        # 从最后一层的输出中获取隐藏状态
        hidden_states = outputs.last_hidden_state
        # 根据logits_to_keep 参数决定要保留的时间步
        # 最常见的场景是, 只在自回归生成中保留最新一个 token 的概率分布, 其余历史位置的 logits 无人在意
        slice_indices = slice(-logits_to_keep, None) if isinstance(logits_to_keep, int) else logits_to_keep
        # 把选择的隐状态投影到词表空间, 得到 (batch, K, vocab_size) 的 logits
        logits = self.lm_head(hidden_states[:, slice_indices, :])
        # 默认不计算 loss
        loss = None
        # 若给了标签, 则调用基类的loss_func计算loss
        if labels is not None:
            loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

        # 返回结果
        return CausalLMOutputWithPast(
            loss=loss, 
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
            attentions=outputs.attentions,
        )
        
# 以下三类继承通用的任务头和 Qwen3 基类
# 表示仅需复用通用头部即可, 无需额外代码
class Qwen3ForSequenceClassification(GenericForSequenceClassification, Qwen3PreTrainedModel):
    pass


class Qwen3ForTokenClassification(GenericForTokenClassification, Qwen3PreTrainedModel):
    pass

# 这里是为了向后兼容旧版权重命名
class Qwen3ForQuestionAnswering(GenericForQuestionAnswering, Qwen3PreTrainedModel):
    base_model_prefix = "transformer"  # For BC, where `transformer` was used instead of `model`

# 定义模块公开的符号集合, 导出相关基类, 供用户从 from...qwen3 import ...时使用
__all__ = [
    "Qwen3ForCausalLM",
    "Qwen3ForQuestionAnswering",
    "Qwen3PreTrainedModel",
    "Qwen3Model",
    "Qwen3ForSequenceClassification",
    "Qwen3ForTokenClassification",
]