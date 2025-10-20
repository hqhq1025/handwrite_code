# 用于记录调试信息/警告
import logging
# 导入类型注解需要的泛型工具, 用于在函数签名中说明参数和返回值结构
from typing import Any, Dict, Iterable, List, Optional, Tuple

# 导入 torch 和 nn
import torch
from torch import nn

# 引入流水线并行与张量并行的查询函数
from sglang.srt.distributed import (
    get_pp_group,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)
# 为每个解码层包装好跨设备的张量分布/归并逻辑，并在自注意力与 MLP 前后调用相应的通信模式
from sglang.srt.layers.communicator import LayerCommunicator, LayerScatterModes
# 导入注意力专用的张量并行信息获取函数
from sglang.srt.layers.dp_attention import get_attention_tp_rank, get_attention_tp_size
# 使用项目自带的 RMSNorm
from sglang.srt.layers.layernorm import RMSNorm
# 引入张量并行线性层, 前者生成qkv权重分片, 后者执行并行的输出投影
from sglang.srt.layers.linear import QKVParallelLinear, RowParallelLinear
# 封装了输出阶段的温度缩放、top-k/p、mask 等操作, 在计算 logits 会用到
from sglang.srt.layers.logits_processor import LogitsProcessor
# 支持隐藏态池化（这里默认取最后一个 token 并归一化）
from sglang.srt.layers.pooler import Pooler, PoolingType
# 导入量化配置基类
from sglang.srt.layers.quantization.base_config import QuantizationConfig
# 引入核心注意力算子 RadixAttention，它实现 radix-based 的高性能注意力前向/缓存逻辑
from sglang.srt.layers.radix_attention import RadixAttention
# 提供旋转位置编码实例
from sglang.srt.layers.rotary_embedding import get_rope
# 流水线并行阶段的占位模块; 帮助权重加载时解析参数属于哪一层
from sglang.srt.layers.utils import PPMissingLayer, get_layer_id
# 词表并行输出层，实现与张量并行相兼容的 logits 映射
from sglang.srt.layers.vocab_parallel_embedding import ParallelLMHead
# 判断当前是否在 CUDA 图捕获阶段
from sglang.srt.model_executor.cuda_graph_runner import get_is_capture_mode
# 引入 ForwardBatch（承载当前 batch 的 GPU 状态、KV 缓存等）和 PPProxyTensors（在流水线 stage 之间传递隐藏态/residual 的容器）
from sglang.srt.model_executor.forward_batch_info import ForwardBatch, PPProxyTensors

from sglang.srt.model_loader.weight_utils import (
    default_weight_loader, # 常规参数拷贝函数
    maybe_remap_kv_scale_name, # 处理 KV 缓存 scale 在不同权重格式下的命名差异，这些在 load_weights 中用于对齐 checkpoint 参数
)

# 直接复用 Qwen2MLP，并在本文件中别名为 Qwen3MLP，因为 Qwen3 的前馈结构与 Qwen2 一致，所以无需重新实现
from sglang.srt.models.qwen2 import Qwen2MLP as Qwen3MLP
# 导入 Qwen2Model，Qwen3Model 会继承它，只替换解码层类型等细节
from sglang.srt.models.qwen2 import Qwen2Model

from sglang.srt.utils import (
    add_prefix, # 用于构建权重名称前缀
    get_cmo_stream, # 在 NPU 上用于权重预取并同步
    # 判断当前设备类型，条件地启用备用流或 CMO
    is_cuda, 
    is_npu,
    wait_cmo_stream,
)

# 暂时把 config 设为 none, 实际配置类在加载权重时动态传入
Qwen3Config = None

# 初始化模块专用日志记录器
logger = logging.getLogger(__name__)

# 判断缓存设备类型
_is_cuda = is_cuda()
_is_npu = is_npu()

# 负责单层注意力的全部子模块
class Qwen3Attention(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int,
        num_kv_heads: int,
        layer_id: int = 0,
        rope_theta: float = 1000000,
        rope_scaling: Optional[Dict[str, Any]] = None,
        head_dim: Optional[int] = None,
        max_position_embeddings: int = 32768,
        quant_config: Optional[QuantizationConfig] = None,
        rms_norm_eps: float = None,
        attention_bias: bool = False,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        self.hidden_size = hidden_size
        # 去的当前张量并行组的规模(同一层参数被切分成多少块)
        self.tp_size = get_tensor_model_parallel_world_size()
        # 总的注意力头数
        self.total_num_heads = num_heads
        # 注意力钻用的张量并行rank和大小
        # rank是指当前卡在组里的编号, 决定持有哪一份权重/张亮
        attn_tp_rank = get_attention_tp_rank()
        # size指注意力张量并行组里一共有多少参与的节点(这一组被拆成几块)
        attn_tp_size = get_attention_tp_size()

        # 判断总头数是否能被attn_tp_size整除, 否则无法均匀分配到各个rank
        assert self.total_num_heads % attn_tp_size == 0
        # 计算每个rank的注意力头数
        self.num_heads = self.total_num_heads // attn_tp_size
        # 保存总的kv头数
        self.total_num_kv_heads = num_kv_heads
        
        # 若 KV 头总数不少于 attn_tp 规模, 则可以把他们切成多份分给不同的 rank
        if self.total_num_kv_heads >= attn_tp_size:
            # 这里看看能不能整除
            assert self.total_num_kv_heads % attn_tp_size == 0
        else:
            # 反之, 需要在多个rank上复制相同的KV(避免某些 rank 没有数据)
            assert attn_tp_size % self.total_num_kv_heads == 0
            
        # 计算当前rank实际持有的KV头数. 足够多就均分, 不足就保证至少是 1
        self.num_kv_heads = max(1, self.total_num_kv_heads // attn_tp_size)
        # 设置单头维度
        self.head_dim = head_dim or hidden_size // self.total_num_heads
        # 计算QKV的维度
        self.q_size = self.num_heads * self.head_dim
        self.kv_size = self.num_kv_heads * self.head_dim
        # 设置缩放系数
        self.scaling = self.head_dim**-0.5
        # 保存rope 的基数
        self.rope_theta = rope_theta
        # 缓存的最大位置长度
        self.max_position_embeddings = max_position_embeddings
        # 记录在整体张量并行组中的编号
        self.tp_rank = get_tensor_model_parallel_rank()

        # 对 kq 做 RMSNorm
        self.q_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)
        self.k_norm = RMSNorm(self.head_dim, eps=rms_norm_eps)

        # 构造处理qkv的并行线性层
        # 输入hidden_size 输出num_heads * head_dim
        self.qkv_proj = QKVParallelLinear(
            hidden_size,
            self.head_dim,
            self.total_num_heads,
            self.total_num_kv_heads,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            prefix=add_prefix("qkv_proj", prefix), # 用于权重命名
        )
        # 构造输出投影
        # 从 num_heads * head_dim 映射回 hidden_size
        self.o_proj = RowParallelLinear(
            self.total_num_heads * self.head_dim,
            hidden_size,
            bias=attention_bias,
            quant_config=quant_config,
            tp_rank=attn_tp_rank,
            tp_size=attn_tp_size,
            reduce_results=False, # 表示该层不会立即做 all_reduce, 而是留给外部决定何时聚合结果
            prefix=add_prefix("o_proj", prefix),
        )

        # 生成rope实例
        self.rotary_emb = get_rope(
            self.head_dim,
            rotary_dim=self.head_dim,
            max_position=max_position_embeddings,
            base=rope_theta,
            rope_scaling=rope_scaling,
        )
        # 实例化radix attention. 这是高性能注意力内核
        self.attn = RadixAttention(
            self.num_heads,
            self.head_dim,
            self.scaling,
            num_kv_heads=self.num_kv_heads,
            layer_id=layer_id,
            prefix=add_prefix("attn", prefix),
        )
        # 保存 alt_stream，这是可选的备用 CUDA stream
        self.alt_stream = alt_stream
        
    # 对qk做rmsnorm
    def _apply_qk_norm(
        self, q: torch.Tensor, k: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 如果传入 alt_stream, 尝试让qk的norm重叠运行
        if self.alt_stream is not None and get_is_capture_mode():
            current_stream = torch.cuda.current_stream()
            self.alt_stream.wait_stream(current_stream)
            q_by_head = q.reshape(-1, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            with torch.cuda.stream(self.alt_stream):
                k_by_head = k.reshape(-1, self.head_dim)
                k_by_head = self.k_norm(k_by_head)
            current_stream.wait_stream(self.alt_stream)
        # 若没有备用流/不在捕获模式, 则顺序进行reshape 和 norm
        else:
            q_by_head = q.reshape(-1, self.head_dim)
            q_by_head = self.q_norm(q_by_head)
            k_by_head = k.reshape(-1, self.head_dim)
            k_by_head = self.k_norm(k_by_head)
        q = q_by_head.view(q.shape)
        k = k_by_head.view(k.shape)
        return q, k
    
    # 自注意力的前向过程
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        forward_batch: ForwardBatch,
    ) -> torch.Tensor:
        # 对隐状态做一次fused线性变化, 得到拼接的qkv张量.
        # 返回的第二个值是bias, 这里不需要直接丢弃
        qkv, _ = self.qkv_proj(hidden_states)
        # 把qkv张量拆成q, k, v
        q, k, v = qkv.split([self.q_size, self.kv_size, self.kv_size], dim=-1)
        # 进行归一化
        q, k = self._apply_qk_norm(q, k)
        # 嵌入旋转编码
        q, k = self.rotary_emb(positions, q, k)
        # 执行注意力计算
        attn_output = self.attn(q, k, v, forward_batch)
        # 用张量并行的输出线性层把注意力输出映射回隐藏维度
        output, _ = self.o_proj(attn_output)
        return output
    
# 构建 Qwen3 解码器中的单个 Transformer 层
class Qwen3DecoderLayer(nn.Module):
    def __init__(
        self,
        config: Qwen3Config,
        layer_id: int = 0,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
        alt_stream: Optional[torch.cuda.Stream] = None,
    ) -> None:
        super().__init__()
        # 从Qwen3Config中获取超参数
        self.hidden_size = config.hidden_size
        rope_theta = getattr(config, "rope_theta", 1000000)
        rope_scaling = getattr(config, "rope_scaling", None)
        max_position_embeddings = getattr(config, "max_position_embeddings", 32768)
        head_dim = getattr(config, "head_dim", None)
        # 构建自注意力层
        self.self_attn = Qwen3Attention(
            hidden_size=self.hidden_size,
            num_heads=config.num_attention_heads,
            num_kv_heads=config.num_key_value_heads,
            layer_id=layer_id,
            rope_theta=rope_theta,
            rope_scaling=rope_scaling,
            head_dim=head_dim,
            max_position_embeddings=max_position_embeddings,
            quant_config=quant_config,
            rms_norm_eps=config.rms_norm_eps,
            attention_bias=config.attention_bias,
            prefix=add_prefix("self_attn", prefix),
            alt_stream=alt_stream,
        )
        # 构建mlp
        self.mlp = Qwen3MLP(
            hidden_size=self.hidden_size,
            intermediate_size=config.intermediate_size,
            hidden_act=config.hidden_act,
            quant_config=quant_config,
            prefix=add_prefix("mlp", prefix),
        )
        self.input_layernorm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, eps=config.rms_norm_eps
        )

        # 创建本层的通信/散布模式元数据
        self.layer_scatter_modes = LayerScatterModes.init_new(
            layer_id=layer_id,
            num_layers=config.num_hidden_layers,
            is_layer_sparse=False,
            is_previous_layer_sparse=False,
        )
        # 基于散布模式, 封装具体的通信逻辑
        self.layer_communicator = LayerCommunicator(
            layer_scatter_modes=self.layer_scatter_modes,
            input_layernorm=self.input_layernorm,
            post_attention_layernorm=self.post_attention_layernorm,
        )
    # Decoder层的前向传播
    def forward(
        self,
        positions: torch.Tensor,
        hidden_states: torch.Tensor,
        # 这是执行阶段的批次描述, 会带有运行时信息
        forward_batch: ForwardBatch,
        residual: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # 根据forward_batch, 决定如何整理输入
        hidden_states, residual = self.layer_communicator.prepare_attn(
            hidden_states, residual, forward_batch
        )
        # 若当前批次有token, 则调用执行注意力计算
        if hidden_states.shape[0] != 0:
            hidden_states = self.self_attn(
                positions=positions,
                hidden_states=hidden_states,
                forward_batch=forward_batch,
            )

        # 在进入mlp前, 将隐藏态和残差按需求整理好
        hidden_states, residual = self.layer_communicator.prepare_mlp(
            hidden_states,
            residual,
            forward_batch,
            cache=(
                [self.mlp.gate_up_proj.weight, self.mlp.down_proj.weight]
                if _is_npu
                else None
            ),
        )
        hidden_states = self.mlp(hidden_states)
        # 对npu的特殊处理
        if _is_npu and get_cmo_stream():
            wait_cmo_stream()
        hidden_states, residual = self.layer_communicator.postprocess_layer(
            hidden_states, residual, forward_batch
        )
        return hidden_states, residual
    
# 在构建完整的Qwen3模型时, 沿用Qwen2的主体框架, 仅替换解码层类型
class Qwen3Model(Qwen2Model):
    def __init__(
        self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        alt_stream = torch.cuda.Stream() if _is_cuda else None
        super().__init__(
            config=config,
            quant_config=quant_config,
            prefix=prefix,
            decoder_layer_type=Qwen3DecoderLayer,
            alt_stream=alt_stream,
        )
        
# 构建Qwen3的CausalLM, 让模型可以用于自回归语言建模
class Qwen3ForCausalLM(nn.Module):
    # 列出适合bitsandbytes量化的权重
    # 用 8bit/4bit 实现替换掉，以减少显存占用、提升吞吐
    default_bitsandbytes_target_modules = [
        ".gate_proj.",
        ".down_proj.",
        ".up_proj.",
        ".q_proj.",
        ".k_proj.",
        ".v_proj.",
        ".o_proj.",
    ]
    # 堆叠权重
    # 把 q_proj/k_proj/v_proj 合并成一个大矩阵 qkv_proj 存储
    bitsandbytes_stacked_params_mapping = {
        # shard_name, weight_name, index
        "q_proj": ("qkv_proj", 0),
        "k_proj": ("qkv_proj", 1),
        "v_proj": ("qkv_proj", 2),
        "gate_proj": ("gate_up_proj", 0),
        "up_proj": ("gate_up_proj", 1),
    }

    def __init__(
        self,
        config: Qwen3Config,
        quant_config: Optional[QuantizationConfig] = None,
        prefix: str = "",
    ) -> None:
        super().__init__()
        # 把当前pipeline的通信组取出
        self.pp_group = get_pp_group()
        # 保存当前配置和量化配置
        self.config = config
        self.quant_config = quant_config
        # 构建模型
        self.model = Qwen3Model(
            config, quant_config=quant_config, prefix=add_prefix("model", prefix)
        )

        # 如果是本进程最后一个 stage
        if self.pp_group.is_last_rank:
            # 若pipeline只有一个stage, 并给开启了tie_word_embeddings, 则直接使用embed_tokens
            # 让输入输出共享一套参数, 实现 tying
            if self.pp_group.world_size == 1 and config.tie_word_embeddings:
                self.lm_head = self.model.embed_tokens
            
            # 否则就在最后 stage 上创建一个 ParallelLMHead，兼容张量并行或非 tying 的情况
            else:
                self.lm_head = ParallelLMHead(
                    config.vocab_size,
                    config.hidden_size,
                    quant_config=quant_config,
                    prefix=add_prefix("lm_head", prefix),
                )
        else:
            # 如果不是最后一个 stage，就放一个 PPMissingLayer() 占位
            # 上游 stage 虽然会构建 Qwen3ForCausalLM，但不会真正在那里执行 logits 计算
            self.lm_head = PPMissingLayer()

        if self.pp_group.world_size > 1 and config.tie_word_embeddings:
            # 若是第一个stake, 拥有实际的emb token, 则把这个矩阵通过send传给最后一个stage
            if self.pp_group.is_first_rank:
                self.pp_group.send(
                    self.model.embed_tokens.weight, dst=self.pp_group.last_rank
                )
            # 其他 stage 中只有最后一个会真正走 else 分支
            # 其从第一个stage接收emb矩阵, 覆盖到自己lm head上
            else:
                emb_token_weight = self.pp_group.recv(
                    size=(config.vocab_size, config.hidden_size),
                    dtype=next(self.model.parameters()).dtype,
                    src=self.pp_group.first_rank,
                )
                self.lm_head.weight.copy_(emb_token_weight)

        # 按配置对logits进行处理
        self.logits_processor = LogitsProcessor(config)
        # 在需要提取序列时使用. 默认取最后一个token的hidden states, 并做归一化
        self.pooler = Pooler(pooling_type=PoolingType.LAST, normalize=True)

        # 给 EAGLE3 等高级特性预留的开关
        self.capture_aux_hidden_states = False
        
    # 把内部的Qwen3Model的emb层暴露, 方便外部使用
    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.get_input_embeddings()


    @torch.no_grad()
    def forward(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        input_embeds: torch.Tensor = None,
        get_embedding: bool = False,
        pp_proxy_tensors: Optional[PPProxyTensors] = None,
    ) -> torch.Tensor:
        # 将输入和位置信息输入模型, 得到隐藏态
        hidden_states = self.model(
            input_ids,
            positions,
            forward_batch,
            input_embeds,
            pp_proxy_tensors=pp_proxy_tensors,
        )

        # 这里是配合那个EAGLE3的
        aux_hidden_states = None
        if self.capture_aux_hidden_states:
            hidden_states, aux_hidden_states = hidden_states


        # 如果是最后一个stage
        if self.pp_group.is_last_rank:
            # 若果不需要获取 embedding, 则进行 logits 处理
            if not get_embedding:
                # 用lm head把隐藏态转换成vocab logits
                return self.logits_processor(
                    input_ids,
                    hidden_states,
                    self.lm_head,
                    forward_batch,
                    aux_hidden_states,
                )
            # 否则, 直接用pooler对隐藏态做池化
            else:
                return self.pooler(hidden_states, forward_batch)
            
        # 直接把隐藏态传给下一个 stage
        else:
            return hidden_states
        
    # 这是prefill前向的函数
    @torch.no_grad()
    def forward_split_prefill(
        self,
        input_ids: torch.Tensor,
        positions: torch.Tensor,
        forward_batch: ForwardBatch,
        split_interval: Tuple[int, int],  # [start, end) 0-based
        input_embeds: torch.Tensor = None,
    ):
        # 表示当前调用负责 start到 end-1 个解码层
        # 可以把长序列的预填充切成几段
        start, end = split_interval
        
        # 表示本次分段从底层开始算, 需准备 emb
        if start == 0:
            # 若外部没给emb, 则用模型自己的emb
            if input_embeds is None:
                forward_batch.hidden_states = self.model.embed_tokens(input_ids)
            # 否则, 使用外部输入的 emb
            else:
                forward_batch.hidden_states = input_embeds
        
        # 循环遍历解码层
        for i in range(start, end):
            # 获取当前的层
            layer = self.model.layers[i]
            # 传入当前的隐藏态和残差
            forward_batch.hidden_states, forward_batch.residual = layer(
                positions,
                forward_batch.hidden_states,
                forward_batch,
                forward_batch.residual,
            )

        # 当end等于解码层总数, 说明已经跑到了最后一层
        if end == self.model.config.num_hidden_layers:
            # 先做输出层归一化, 并处理残差
            hidden_states, _ = self.model.norm(
                forward_batch.hidden_states, forward_batch.residual
            )
            forward_batch.hidden_states = hidden_states
            # 用lm head把隐藏态转换成vocab logits
            result = self.logits_processor(
                input_ids, forward_batch.hidden_states, self.lm_head, forward_batch
            )
        else:
            result = None

        return result
    
    # 将模型中的start_layer和end_layer属性暴露出来
    @property
    def start_layer(self):
        return self.model.start_layer

    @property
    def end_layer(self):
        return self.model.end_layer


    # 把外部加载的权重逐条加载进模型
    def load_weights(self, weights: Iterable[Tuple[str, torch.Tensor]]):
        # 描述哪些权重是堆叠存储的
        stacked_params_mapping = [
            # (param_name, shard_name, shard_id)
            ("qkv_proj", "q_proj", "q"),
            ("qkv_proj", "k_proj", "k"),
            ("qkv_proj", "v_proj", "v"),
            ("gate_up_proj", "gate_proj", 0),
            ("gate_up_proj", "up_proj", 1),
        ]

        # 建立一个参数字典, 方便后续定位参数
        params_dict = dict(self.named_parameters())
        for name, loaded_weight in weights:
            # 如果是emb模型, 则先加上model.前缀, 匹配当前模型命名
            if "Embedding" in self.config.name_or_path:
                name = add_prefix(name, "model")
            # 从权重名中解析层号
            layer_id = get_layer_id(name)
            if (
                layer_id is not None # 层号解析成功
                and hasattr(self.model, "start_layer") # 模型有start_layer和end_layer属性
                and (
                    # 不在当前进程负责范围
                    layer_id < self.model.start_layer  
                    or layer_id >= self.model.end_layer
                )
            ):
                continue # 跳过

            # 跳过无需加载的条目
            if "rotary_emb.inv_freq" in name or "projector" in name:
                continue
            if "rotary_emb.cos_cached" in name or "rotary_emb.sin_cached" in name:
                # Models trained using ColossalAI may include these tensors in
                # the checkpoint. Skip them.
                continue
            
            # 共享权重, 并且当前是lm head层
            if self.config.tie_word_embeddings and "lm_head.weight" in name:
                # 若是多stage, 并且是最后一段
                if self.pp_group.world_size > 1 and self.pp_group.is_last_rank:
                    # 获取 emb
                    embed_token_weights = next(
                        filter(lambda x: x[0] == "model.embed_tokens.weight", weights)
                    )[1]
                    # 把 emb转换成lm head
                    loaded_weight = embed_token_weights
                else:
                    continue
                
            # 有些 checkpoint 会带上视觉分支的权重，但当前模型里没这个模块，这种情况下直接跳过
            if name.startswith("model.vision_tower") and name not in params_dict:
                continue
            # 某些 KV 量化或缩放参数在不同版本里命名有差异
            if "scale" in name:
                # 查当前模型的参数表 params_dict，必要时把名字改成模型实际用的那套
                name = maybe_remap_kv_scale_name(name, params_dict)
                # 这条权重在当前模型中根本没有对应项，于是直接跳过
                if name is None:
                    continue
                
            for param_name, weight_name, shard_id in stacked_params_mapping:
                # 如果当前权重名里不包含 weight_name（例如 .q_proj.），就继续下一条映射
                if weight_name not in name:
                    continue
                # 找到了就把名字里的 weight_name 替换成堆叠矩阵 param_name（如 qkv_proj），并得到堆叠的索引 shard_id
                name = name.replace(weight_name, param_name)
                # GPTQ 等模型可能多存了 bias，如果模型里没有这个 bias，就跳过
                if name.endswith(".bias") and name not in params_dict:
                    continue
                # 取出真正的参数
                param = params_dict[name]
                # 调用自身的权重加载函数, 将切片写入
                weight_loader = param.weight_loader
                weight_loader(param, loaded_weight, shard_id)
                break
            
            # break 之后
            else:
                #  先处理 GPTQ 这类模型可能额外带的 bias
                if name.endswith(".bias") and name not in params_dict:
                    continue
                
                # 检查 name 是否存在于 params_dict
                if name in params_dict.keys():
                    param = params_dict[name]
                    weight_loader = getattr(
                        param, "weight_loader", default_weight_loader
                    )
                    weight_loader(param, loaded_weight)
                else:
                    logger.warning(f"Parameter {name} not found in params_dict")

    # 返回当前的emb和lm head
    def get_embed_and_head(self):
        return self.model.embed_tokens.weight, self.lm_head.weight

    # 删除已有权重, 将外部传入的对象重新挂到相应位置
    def set_embed_and_head(self, embed, head):
        del self.model.embed_tokens.weight
        del self.lm_head.weight
        self.model.embed_tokens.weight = embed
        self.lm_head.weight = head
        torch.cuda.empty_cache()
        torch.cuda.synchronize()

    # 加载 KV 量化缩放参数
    def load_kv_cache_scales(self, quantization_param_path: str) -> None:
        self.model.load_kv_cache_scales(quantization_param_path)

    # 兼容 EAGLE3 的特性，让模型在前向过程中抓取特定层的隐藏态
    def set_eagle3_layers_to_capture(self, layer_ids: Optional[List[int]] = None):
        if not self.pp_group.is_last_rank:
            return

        self.capture_aux_hidden_states = True
        if layer_ids is None:
            num_layers = self.config.num_hidden_layers
            self.model.layers_to_capture = [
                2,
                num_layers // 2,
                num_layers - 3,
            ]  # Specific layers for EAGLE3 support
        else:
            self.model.layers_to_capture = [val + 1 for val in layer_ids]

# 将Qwen3ForCausalLM标记为对外暴露的入口
EntryClass = Qwen3ForCausalLM
