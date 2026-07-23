from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass
from typing import Any, Literal

import torch
import torch.nn as nn
import torch.nn.functional as F


Architecture = Literal["dense", "switch", "standard_moe", "boichitro_moe"]


@dataclass
class BoichitroConfig:
    vocab_size: int
    architecture: Architecture = "dense"
    max_seq_len: int = 1024
    n_layers: int = 16
    d_model: int = 512
    n_heads: int = 8
    n_kv_heads: int = 2
    dense_ffn_dim: int = 2304
    dropout: float = 0.0
    rope_theta: float = 10_000.0
    rms_norm_eps: float = 1e-6
    qk_norm: bool = True
    tie_embeddings: bool = True
    pad_token_id: int = 0

    dense_prefix_layers: int = 4
    n_routed_experts: int = 8
    expert_dim: int = 768
    top_k: int = 2
    shared_expert: bool = True
    router_z_loss_weight: float = 1e-4
    balance_loss_weight: float = 0.0
    dynamic_bias_update_rate: float = 1e-3
    banked_upcycle_fraction: float = 0.02
    banked_upcycle_release_fraction: float = 0.02
    banked_pairing_penalty: float = 0.25
    use_grouped_gemm: bool = True

    n_dialects: int = 13
    n_tasks: int = 3
    n_sources: int = 1
    dialect_loss_weight: float = 0.10
    source_adversary_weight: float = 0.01
    classification_loss_weight: float = 1.0
    contrastive_loss_weight: float = 0.0
    contrastive_temperature: float = 0.07
    mtp_loss_weight: float = 0.05
    use_mtp: bool = True
    use_classification_head: bool = True
    use_dialect_aux_head: bool = True
    use_source_adversary: bool = False
    use_task_conditioning: bool = False
    use_lexical_routing_prior: bool = False
    randomize_lexical_prior: bool = False
    lexical_prior_permutation_seed: int = 8675309
    bidirectional_attention: bool = False

    def validate(self) -> None:
        if self.d_model % self.n_heads:
            raise ValueError("d_model must be divisible by n_heads")
        if self.n_heads % self.n_kv_heads:
            raise ValueError("n_heads must be divisible by n_kv_heads")
        if self.top_k > self.n_routed_experts:
            raise ValueError("top_k cannot exceed n_routed_experts")
        if self.architecture == "switch" and self.top_k != 1:
            raise ValueError("Switch uses top_k=1")
        if self.architecture == "dense" and self.dense_prefix_layers > self.n_layers:
            raise ValueError("dense_prefix_layers exceeds n_layers")
        if self.contrastive_loss_weight < 0:
            raise ValueError("contrastive_loss_weight must be non-negative")
        if self.contrastive_temperature <= 0:
            raise ValueError("contrastive_temperature must be positive")
        if not 0.0 <= self.banked_upcycle_fraction <= 1.0:
            raise ValueError("banked_upcycle_fraction must be in [0, 1]")
        if not self.banked_upcycle_fraction <= self.banked_upcycle_release_fraction <= 1.0:
            raise ValueError(
                "banked_upcycle_release_fraction must be in "
                "[banked_upcycle_fraction, 1]"
            )
        if self.banked_pairing_penalty < 0.0:
            raise ValueError("banked_pairing_penalty must be non-negative")

    @classmethod
    def from_dict(cls, values: dict[str, Any]) -> "BoichitroConfig":
        config = cls(**values)
        config.validate()
        return config

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx: Any, value: torch.Tensor, scale: float) -> torch.Tensor:
        ctx.scale = float(scale)
        return value.view_as(value)

    @staticmethod
    def backward(ctx: Any, gradient: torch.Tensor) -> tuple[torch.Tensor, None]:
        return -ctx.scale * gradient, None


class RMSNorm(nn.Module):
    """RMSNorm with an FP32 master scale and activation-matched compute scale."""

    def __init__(self, width: int, eps: float) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return F.rms_norm(value, (value.size(-1),), self.weight.to(value.dtype), self.eps)


def rotate_half(value: torch.Tensor) -> torch.Tensor:
    first, second = value.chunk(2, dim=-1)
    return torch.cat((-second, first), dim=-1)


class RotaryEmbedding(nn.Module):
    def __init__(self, head_dim: int, max_seq_len: int, theta: float) -> None:
        super().__init__()
        if head_dim % 2:
            raise ValueError("RoPE head dimension must be even")
        inverse_frequency = 1.0 / (
            theta ** (torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        )
        positions = torch.arange(max_seq_len, dtype=torch.float32)
        frequencies = torch.outer(positions, inverse_frequency)
        embedding = torch.cat((frequencies, frequencies), dim=-1)
        self.register_buffer("cos", embedding.cos()[None, None, :, :], persistent=False)
        self.register_buffer("sin", embedding.sin()[None, None, :, :], persistent=False)

    def forward(
        self, query: torch.Tensor, key: torch.Tensor, position_offset: int = 0
    ) -> tuple[torch.Tensor, torch.Tensor]:
        length = query.size(-2)
        cosine = self.cos[:, :, position_offset : position_offset + length].to(query.dtype)
        sine = self.sin[:, :, position_offset : position_offset + length].to(query.dtype)
        return query * cosine + rotate_half(query) * sine, key * cosine + rotate_half(key) * sine


class GroupedQueryAttention(nn.Module):
    def __init__(self, config: BoichitroConfig) -> None:
        super().__init__()
        self.n_heads = config.n_heads
        self.n_kv_heads = config.n_kv_heads
        self.head_dim = config.d_model // config.n_heads
        self.dropout = config.dropout
        self.bidirectional = config.bidirectional_attention
        self.q_proj = nn.Linear(config.d_model, config.n_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(config.d_model, config.n_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(config.n_heads * self.head_dim, config.d_model, bias=False)
        self.q_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else nn.Identity()
        self.k_norm = RMSNorm(self.head_dim, config.rms_norm_eps) if config.qk_norm else nn.Identity()
        self.rope = RotaryEmbedding(self.head_dim, config.max_seq_len, config.rope_theta)

    def forward(
        self,
        value: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, tuple[torch.Tensor, torch.Tensor] | None]:
        batch, length, _ = value.shape
        query = self.q_proj(value).view(batch, length, self.n_heads, self.head_dim).transpose(1, 2)
        key = self.k_proj(value).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        val = self.v_proj(value).view(batch, length, self.n_kv_heads, self.head_dim).transpose(1, 2)
        query = self.q_norm(query)
        key = self.k_norm(key)
        past_length = 0 if past_key_value is None else past_key_value[0].size(-2)
        query, key = self.rope(query, key, position_offset=past_length)
        if past_key_value is not None:
            key = torch.cat((past_key_value[0], key), dim=-2)
            val = torch.cat((past_key_value[1], val), dim=-2)

        mask: torch.Tensor | None = None
        causal = not self.bidirectional
        if self.bidirectional and attention_mask is not None:
            mask = attention_mask[:, None, None, :].to(torch.bool)
        elif use_cache and attention_mask is not None:
            # CUDA SDPA does not allow explicit padding masks together with
            # is_causal=True. Build the offset causal mask for the one prompt
            # pass; cached single-token steps then attend to all valid history.
            key_positions = torch.arange(key.size(-2), device=value.device)
            query_positions = past_length + torch.arange(length, device=value.device)
            causal_mask = key_positions[None, :] <= query_positions[:, None]
            mask = causal_mask[None, None] & attention_mask[:, None, None, :].to(torch.bool)
            causal = False
        attended = F.scaled_dot_product_attention(
            query,
            key,
            val,
            attn_mask=mask,
            dropout_p=self.dropout if self.training else 0.0,
            is_causal=causal,
            enable_gqa=self.n_heads != self.n_kv_heads,
        )
        attended = attended.transpose(1, 2).contiguous().view(batch, length, -1)
        present = (key, val) if use_cache else None
        return self.o_proj(attended), present


class SwiGLU(nn.Module):
    def __init__(self, d_model: int, hidden_dim: int) -> None:
        super().__init__()
        self.gate_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.up_proj = nn.Linear(d_model, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, d_model, bias=False)

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        return self.down_proj(F.silu(self.gate_proj(value)) * self.up_proj(value))


@dataclass
class RouterOutput:
    hidden: torch.Tensor
    counts: torch.Tensor
    probability_mass: torch.Tensor
    entropy: torch.Tensor
    z_loss: torch.Tensor
    balance_loss: torch.Tensor
    selected_experts: torch.Tensor | None = None


class SparseMoE(nn.Module):
    def __init__(self, config: BoichitroConfig, layer_index: int) -> None:
        super().__init__()
        self.config = config
        self.layer_index = layer_index
        self.n_experts = config.n_routed_experts
        self.top_k = config.top_k
        self.router = nn.Linear(config.d_model, self.n_experts, bias=False)
        self.experts = nn.ModuleList(
            [SwiGLU(config.d_model, config.expert_dim) for _ in range(self.n_experts)]
        )
        self.shared_expert = (
            SwiGLU(config.d_model, config.expert_dim) if config.shared_expert else None
        )
        self.register_buffer("dynamic_bias", torch.zeros(self.n_experts, dtype=torch.float32))
        self.register_buffer("last_counts", torch.zeros(self.n_experts, dtype=torch.long), persistent=False)

        proposed = config.architecture == "boichitro_moe"
        relative = layer_index - config.dense_prefix_layers
        self.lexical_conditioned = proposed and config.use_lexical_routing_prior and relative < 4
        self.task_conditioned = proposed and config.use_task_conditioning and relative >= 8
        self.task_router_bias = (
            nn.Embedding(config.n_tasks, self.n_experts) if self.task_conditioned else None
        )
        self.dialect_router_map = (
            nn.Parameter(torch.zeros(config.n_dialects, self.n_experts))
            if self.lexical_conditioned
            else None
        )
        if self.lexical_conditioned:
            generator = torch.Generator().manual_seed(
                config.lexical_prior_permutation_seed + layer_index
            )
            permutation = torch.randperm(config.n_dialects, generator=generator)
        else:
            permutation = torch.arange(config.n_dialects)
        self.register_buffer("dialect_permutation", permutation, persistent=False)

    def _topk(
        self,
        probabilities: torch.Tensor,
        selection_scores: torch.Tensor,
        bank_constraint_strength: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        strength = min(1.0, max(0.0, float(bank_constraint_strength)))
        if strength >= 1.0 and self.top_k == 2 and self.n_experts % 2 == 0:
            split = self.n_experts // 2
            first = selection_scores[:, :split].argmax(dim=-1, keepdim=True)
            second = selection_scores[:, split:].argmax(dim=-1, keepdim=True) + split
            indices = torch.cat((first, second), dim=-1)
            weights = probabilities.gather(-1, indices)
        elif strength > 0.0 and self.top_k == 2 and self.n_experts % 2 == 0:
            # Anneal from exact cross-bank pairing to unrestricted top-2. The
            # first expert is selected normally. A decaying penalty on experts
            # from the same bank makes individual tokens migrate at different
            # score margins instead of changing every route at one boundary.
            split = self.n_experts // 2
            first = selection_scores.argmax(dim=-1, keepdim=True)
            expert_ids = torch.arange(
                self.n_experts, device=selection_scores.device
            ).unsqueeze(0)
            first_is_upper = first.ge(split)
            same_bank = expert_ids.ge(split).eq(first_is_upper)
            second_scores = selection_scores - same_bank.to(selection_scores.dtype) * (
                self.config.banked_pairing_penalty * strength
            )
            second_scores = second_scores.scatter(1, first, -torch.inf)
            second = second_scores.argmax(dim=-1, keepdim=True)
            indices = torch.cat((first, second), dim=-1)
            weights = probabilities.gather(-1, indices)
        else:
            _, indices = torch.topk(selection_scores, k=self.top_k, dim=-1)
            weights = probabilities.gather(-1, indices)
        if self.top_k == 1:
            # Forward-exact straight-through gate. This preserves a cloned
            # dense FFN at upcycling (gate value is exactly one) while keeping
            # the language-model gradient to the selected router probability.
            weights = weights / weights.detach().clamp_min(1e-9)
        else:
            weights = weights / weights.sum(dim=-1, keepdim=True).clamp_min(1e-9)
        return indices, weights

    def forward(
        self,
        value: torch.Tensor,
        *,
        token_mask: torch.Tensor | None = None,
        task_ids: torch.Tensor | None,
        dialect_probabilities: torch.Tensor | None,
        lexical_prior_scale: float,
        bank_constraint_strength: float,
        capture_assignments: bool,
    ) -> RouterOutput:
        batch, length, width = value.shape
        flat = value.reshape(-1, width)
        logits = F.linear(flat.float(), self.router.weight.float())
        if self.task_router_bias is not None and task_ids is not None:
            task_bias = self.task_router_bias(task_ids).float()
            logits = logits + task_bias[:, None, :].expand(batch, length, -1).reshape(-1, self.n_experts)
        if (
            self.dialect_router_map is not None
            and dialect_probabilities is not None
            and lexical_prior_scale > 0.0
        ):
            dialect_evidence = dialect_probabilities.float()
            if self.config.randomize_lexical_prior:
                dialect_evidence = dialect_evidence.index_select(
                    -1, self.dialect_permutation
                )
            prior = dialect_evidence @ self.dialect_router_map.float()
            logits = logits + lexical_prior_scale * prior.reshape(-1, self.n_experts)

        probabilities = torch.softmax(logits, dim=-1)
        # DeepSeek-style loss-free bias changes selection, not mixture weights.
        selection_scores = probabilities + self.dynamic_bias.to(probabilities.device)
        indices, weights = self._topk(
            probabilities, selection_scores, bank_constraint_strength
        )
        if self.config.use_grouped_gemm and flat.is_cuda and flat.dtype == torch.bfloat16:
            routed = self._grouped_dispatch(flat, indices, weights)
        else:
            routed = torch.zeros_like(flat)
            for expert_index, expert in enumerate(self.experts):
                token_indices, slots = torch.where(indices == expert_index)
                if token_indices.numel() == 0:
                    continue
                expert_values = expert(flat.index_select(0, token_indices))
                expert_weights = weights[token_indices, slots].to(expert_values.dtype).unsqueeze(-1)
                routed.index_add_(0, token_indices, expert_values * expert_weights)
        if self.shared_expert is not None:
            routed = routed + self.shared_expert(flat)

        if token_mask is None:
            valid = torch.ones(flat.size(0), dtype=torch.bool, device=flat.device)
        else:
            valid = token_mask.reshape(-1).to(device=flat.device, dtype=torch.bool)
        valid_indices = indices[valid]
        valid_probabilities = probabilities[valid]
        counts = torch.bincount(valid_indices.reshape(-1), minlength=self.n_experts)
        if self.training:
            # Accumulate across all microbatches owned by one optimizer step.
            # Evaluation forwards must never perturb subsequent router updates.
            self.last_counts.add_(counts.detach().to(self.last_counts.device))
        probability_mass = valid_probabilities.mean(dim=0)
        load_fraction = counts.float() / max(1, valid_indices.numel())
        balance_loss = self.n_experts * torch.sum(probability_mass * load_fraction)
        entropy = -(
            valid_probabilities * valid_probabilities.clamp_min(1e-9).log()
        ).sum(dim=-1).mean()
        z_loss = torch.logsumexp(logits[valid], dim=-1).square().mean()
        return RouterOutput(
            hidden=routed.view(batch, length, width),
            counts=counts,
            probability_mass=probability_mass,
            entropy=entropy,
            z_loss=z_loss,
            balance_loss=balance_loss,
            selected_experts=indices.view(batch, length, self.top_k) if capture_assignments else None,
        )

    def _grouped_dispatch(
        self,
        flat: torch.Tensor,
        indices: torch.Tensor,
        weights: torch.Tensor,
    ) -> torch.Tensor:
        """Dropless expert dispatch using PyTorch's grouped BF16 GEMM kernel."""

        token_indices = (
            torch.arange(flat.size(0), device=flat.device)
            .unsqueeze(1)
            .expand(-1, self.top_k)
            .reshape(-1)
        )
        expert_indices = indices.reshape(-1)
        mixture_weights = weights.reshape(-1)
        order = torch.argsort(expert_indices, stable=True)
        sorted_tokens = token_indices.index_select(0, order)
        sorted_experts = expert_indices.index_select(0, order)
        sorted_input = flat.index_select(0, sorted_tokens).contiguous()
        counts = torch.bincount(sorted_experts, minlength=self.n_experts)
        offsets = counts.cumsum(0).to(torch.int32)
        dtype = flat.dtype

        gate_weights = torch.stack(
            [expert.gate_proj.weight.transpose(0, 1) for expert in self.experts]
        ).to(dtype)
        up_weights = torch.stack(
            [expert.up_proj.weight.transpose(0, 1) for expert in self.experts]
        ).to(dtype)
        down_weights = torch.stack(
            [expert.down_proj.weight.transpose(0, 1) for expert in self.experts]
        ).to(dtype)
        gate = torch._grouped_mm(sorted_input, gate_weights, offsets)
        up = torch._grouped_mm(sorted_input, up_weights, offsets)
        activated = F.silu(gate) * up
        expert_output = torch._grouped_mm(activated, down_weights, offsets)
        sorted_mixture = mixture_weights.index_select(0, order).to(dtype).unsqueeze(-1)
        routed = torch.zeros_like(flat)
        routed.index_add_(0, sorted_tokens, expert_output * sorted_mixture)
        return routed

    @torch.no_grad()
    def update_dynamic_bias(self) -> None:
        counts = self.last_counts.float()
        if counts.sum() == 0:
            return
        target = counts.mean()
        direction = torch.sign(target - counts)
        self.dynamic_bias.add_(self.config.dynamic_bias_update_rate * direction)
        self.dynamic_bias.sub_(self.dynamic_bias.mean())
        self.last_counts.zero_()

    def total_expert_parameters(self) -> int:
        modules = list(self.experts)
        if self.shared_expert is not None:
            modules.append(self.shared_expert)
        return sum(parameter.numel() for module in modules for parameter in module.parameters())

    def active_expert_parameters(self) -> int:
        one = sum(parameter.numel() for parameter in self.experts[0].parameters())
        shared = (
            sum(parameter.numel() for parameter in self.shared_expert.parameters())
            if self.shared_expert is not None
            else 0
        )
        return shared + self.top_k * one


class DecoderBlock(nn.Module):
    def __init__(self, config: BoichitroConfig, layer_index: int, use_moe: bool) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.attention = GroupedQueryAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.ffn: SwiGLU | SparseMoE
        self.ffn = SparseMoE(config, layer_index) if use_moe else SwiGLU(config.d_model, config.dense_ffn_dim)
        self.dropout = config.dropout

    def forward(
        self,
        value: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None,
        task_ids: torch.Tensor | None,
        dialect_probabilities: torch.Tensor | None,
        lexical_prior_scale: float,
        bank_constraint_strength: float,
        capture_assignments: bool,
        past_key_value: tuple[torch.Tensor, torch.Tensor] | None = None,
        use_cache: bool = False,
    ) -> tuple[
        torch.Tensor,
        RouterOutput | None,
        tuple[torch.Tensor, torch.Tensor] | None,
    ]:
        attended, present_key_value = self.attention(
            self.attention_norm(value),
            attention_mask,
            past_key_value=past_key_value,
            use_cache=use_cache,
        )
        value = value + F.dropout(
            attended,
            p=self.dropout,
            training=self.training,
        )
        normalized = self.ffn_norm(value)
        router_output = None
        if isinstance(self.ffn, SparseMoE):
            token_mask = (
                attention_mask[:, -value.size(1) :]
                if attention_mask is not None
                else None
            )
            router_output = self.ffn(
                normalized,
                token_mask=token_mask,
                task_ids=task_ids,
                dialect_probabilities=dialect_probabilities,
                lexical_prior_scale=lexical_prior_scale,
                bank_constraint_strength=bank_constraint_strength,
                capture_assignments=capture_assignments,
            )
            feedforward = router_output.hidden
        else:
            feedforward = self.ffn(normalized)
        value = value + F.dropout(feedforward, p=self.dropout, training=self.training)
        return value, router_output, present_key_value


def last_token_pool(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        return hidden[:, -1]
    indices = torch.arange(attention_mask.size(1), device=hidden.device).unsqueeze(0)
    positions = (indices * attention_mask.long()).max(dim=-1).values
    return hidden[torch.arange(hidden.size(0), device=hidden.device), positions]


def causal_running_mean(hidden: torch.Tensor, attention_mask: torch.Tensor | None) -> torch.Tensor:
    if attention_mask is None:
        mask = torch.ones(hidden.shape[:2], device=hidden.device, dtype=hidden.dtype)
    else:
        mask = attention_mask.to(hidden.dtype)
    cumulative = (hidden * mask.unsqueeze(-1)).cumsum(dim=1)
    denominator = mask.cumsum(dim=1).clamp_min(1.0).unsqueeze(-1)
    return cumulative / denominator


def masked_cross_entropy_per_example(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid = labels.ne(-100)
    safe_labels = labels.masked_fill(~valid, 0)
    losses = F.cross_entropy(logits.float(), safe_labels, reduction="none")
    return losses * valid.to(losses.dtype)


def masked_mean(values: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    valid = labels.ne(-100)
    return values.sum() / valid.sum().clamp_min(1)


def supervised_contrastive_per_example(
    representations: torch.Tensor,
    labels: torch.Tensor,
    source_labels: torch.Tensor,
    temperature: float,
) -> torch.Tensor:
    """Use only same-dialect, different-source positives.

    Dialects represented by one source in a batch receive zero contrastive
    loss. Same-dialect/same-source pairs are neutral rather than false
    negatives; different-dialect examples form the negative set.
    """

    result = representations.new_zeros(representations.size(0), dtype=torch.float32)
    valid = labels.ge(0) & source_labels.ge(0)
    if int(valid.sum()) < 2:
        return result
    selected = F.normalize(representations[valid].float(), dim=-1)
    selected_labels = labels[valid]
    selected_sources = source_labels[valid]
    similarity = selected @ selected.transpose(0, 1) / temperature
    identity = torch.eye(len(selected), device=selected.device, dtype=torch.bool)
    positives = (
        selected_labels[:, None].eq(selected_labels[None, :])
        & selected_sources[:, None].ne(selected_sources[None, :])
        & ~identity
    )
    has_positive = positives.any(dim=1)
    if not bool(has_positive.any()):
        return result
    neutral = (
        selected_labels[:, None].eq(selected_labels[None, :])
        & selected_sources[:, None].eq(selected_sources[None, :])
        & ~identity
    )
    denominator_logits = similarity.masked_fill(identity | neutral, -torch.inf)
    log_probabilities = similarity - torch.logsumexp(denominator_logits, dim=1, keepdim=True)
    selected_losses = -(
        log_probabilities.masked_fill(~positives, 0.0).sum(dim=1)
        / positives.sum(dim=1).clamp_min(1)
    )
    selected_losses = selected_losses.masked_fill(~has_positive, 0.0)
    result[valid] = selected_losses
    return result


class BoichitroForMultiTask(nn.Module):
    def __init__(self, config: BoichitroConfig) -> None:
        super().__init__()
        config.validate()
        self.config = config
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model, padding_idx=config.pad_token_id)
        self.layers = nn.ModuleList()
        for layer_index in range(config.n_layers):
            use_moe = config.architecture != "dense" and layer_index >= config.dense_prefix_layers
            self.layers.append(DecoderBlock(config, layer_index, use_moe))
        self.final_norm = RMSNorm(config.d_model, config.rms_norm_eps)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        if config.tie_embeddings:
            self.lm_head.weight = self.token_embedding.weight

        self.dialect_head = (
            nn.Linear(config.d_model, config.n_dialects) if config.use_dialect_aux_head else None
        )
        self.routing_dialect_head = (
            nn.Linear(config.d_model, config.n_dialects)
            if config.architecture == "boichitro_moe" and config.use_lexical_routing_prior
            else None
        )
        self.classification_head = (
            nn.Linear(config.d_model, config.n_dialects) if config.use_classification_head else None
        )
        self.source_head = (
            nn.Linear(config.d_model, config.n_sources) if config.use_source_adversary else None
        )
        self.mtp_projection = (
            nn.Linear(config.d_model, config.d_model, bias=False) if config.use_mtp else None
        )
        self.training_progress = 1.0
        self.apply(self._initialize)

    def _initialize(self, module: nn.Module) -> None:
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.padding_idx is not None:
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def set_training_progress(self, progress: float) -> None:
        self.training_progress = float(min(1.0, max(0.0, progress)))

    def _language_model_loss_per_example(
        self,
        logits: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        token_loss = F.cross_entropy(
            logits[:, :-1].float().reshape(-1, logits.size(-1)),
            labels[:, 1:].reshape(-1),
            ignore_index=-100,
            reduction="none",
        ).view(labels.size(0), -1)
        valid = labels[:, 1:].ne(-100)
        per_example = (token_loss * valid).sum(dim=-1) / valid.sum(dim=-1).clamp_min(1)
        return per_example

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
        task_ids: torch.Tensor | None = None,
        dialect_labels: torch.Tensor | None = None,
        source_labels: torch.Tensor | None = None,
        classification_labels: torch.Tensor | None = None,
        example_weights: torch.Tensor | None = None,
        adversary_scale: float = 1.0,
        capture_routing: bool = False,
        return_lm_logits: bool | None = None,
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]] | None = None,
        use_cache: bool = False,
    ) -> dict[str, Any]:
        if input_ids.size(1) > self.config.max_seq_len:
            raise ValueError("Input exceeds configured sequence length")
        hidden = self.token_embedding(input_ids)
        if hidden.is_cuda and torch.is_autocast_enabled("cuda"):
            hidden = hidden.to(torch.get_autocast_dtype("cuda"))
        router_outputs: list[RouterOutput] = []
        dialect_probabilities: torch.Tensor | None = None
        routing_dialect_logits: torch.Tensor | None = None
        lexical_prior_scale = max(0.0, 1.0 - self.training_progress)
        bank_constraint_strength = self.upcycle_bank_constraint_strength()
        present_key_values: list[tuple[torch.Tensor, torch.Tensor]] = []
        if past_key_values is not None and len(past_key_values) != len(self.layers):
            raise ValueError("past_key_values must contain one entry per decoder layer")
        current_attention_mask = (
            attention_mask[:, -input_ids.size(1) :]
            if attention_mask is not None
            else None
        )

        for layer_index, layer in enumerate(self.layers):
            if (
                self.config.architecture == "boichitro_moe"
                and self.config.use_lexical_routing_prior
                and layer_index == self.config.dense_prefix_layers
                and self.routing_dialect_head is not None
            ):
                running = causal_running_mean(hidden, current_attention_mask)
                routing_sequence_logits = self.routing_dialect_head(running)
                dialect_probabilities = torch.softmax(routing_sequence_logits.float(), dim=-1)
                routing_dialect_logits = last_token_pool(
                    routing_sequence_logits, current_attention_mask
                )
            hidden, router_output, present_key_value = layer(
                hidden,
                attention_mask=attention_mask,
                task_ids=task_ids,
                dialect_probabilities=dialect_probabilities,
                lexical_prior_scale=lexical_prior_scale,
                bank_constraint_strength=bank_constraint_strength,
                capture_assignments=capture_routing,
                past_key_value=(
                    past_key_values[layer_index] if past_key_values is not None else None
                ),
                use_cache=use_cache,
            )
            if present_key_value is not None:
                present_key_values.append(present_key_value)
            if router_output is not None:
                router_outputs.append(router_output)
        hidden = self.final_norm(hidden)
        has_lm_labels = labels is not None and bool(labels.ne(-100).any())
        if return_lm_logits is None:
            return_lm_logits = labels is None or has_lm_labels
        logits = self.lm_head(hidden) if return_lm_logits or has_lm_labels else None
        pooled = last_token_pool(hidden, current_attention_mask)
        zero = hidden.new_zeros(())
        losses: dict[str, torch.Tensor] = {}
        batch_size = input_ids.size(0)
        zero_vector = hidden.new_zeros(batch_size, dtype=torch.float32)
        lm_per_example = (
            self._language_model_loss_per_example(logits, labels)
            if has_lm_labels and logits is not None
            else zero_vector
        )
        losses["language_model"] = lm_per_example.mean()
        if self.mtp_projection is not None and has_lm_labels and labels.size(1) >= 3:
            mtp_logits = F.linear(self.mtp_projection(hidden[:, :-2]), self.lm_head.weight)
            mtp_labels = labels[:, 2:]
            mtp_token = F.cross_entropy(
                mtp_logits.float().reshape(-1, mtp_logits.size(-1)),
                mtp_labels.masked_fill(mtp_labels.eq(-100), 0).reshape(-1),
                reduction="none",
            ).view(mtp_labels.shape)
            mtp_valid = mtp_labels.ne(-100)
            mtp_per_example = (mtp_token * mtp_valid).sum(dim=-1) / mtp_valid.sum(
                dim=-1
            ).clamp_min(1)
            losses["mtp"] = mtp_per_example.mean()
        else:
            mtp_per_example = zero_vector
            losses["mtp"] = zero

        dialect_logits = self.dialect_head(pooled) if self.dialect_head is not None else None
        dialect_supervision = []
        if dialect_logits is not None and dialect_labels is not None:
            dialect_supervision.append(
                masked_cross_entropy_per_example(dialect_logits, dialect_labels)
            )
        if routing_dialect_logits is not None and dialect_labels is not None:
            dialect_supervision.append(
                masked_cross_entropy_per_example(routing_dialect_logits, dialect_labels)
            )
        if dialect_supervision and dialect_labels is not None:
            dialect_per_example = torch.stack(dialect_supervision).mean(dim=0)
            losses["dialect"] = masked_mean(dialect_per_example, dialect_labels)
        else:
            dialect_per_example = zero_vector
            losses["dialect"] = zero

        classification_logits = (
            self.classification_head(pooled) if self.classification_head is not None else None
        )
        if classification_logits is not None and classification_labels is not None:
            classification_per_example = masked_cross_entropy_per_example(
                classification_logits, classification_labels
            )
            losses["classification"] = masked_mean(
                classification_per_example, classification_labels
            )
        else:
            classification_per_example = zero_vector
            losses["classification"] = zero

        if (
            self.config.contrastive_loss_weight > 0
            and dialect_labels is not None
            and source_labels is not None
        ):
            contrastive_per_example = supervised_contrastive_per_example(
                pooled,
                dialect_labels,
                source_labels,
                self.config.contrastive_temperature,
            )
            contrastive_valid = (
                dialect_labels.ge(0)
                & source_labels.ge(0)
                & contrastive_per_example.ne(0)
            )
            losses["contrastive"] = (
                contrastive_per_example[contrastive_valid].mean()
                if bool(contrastive_valid.any())
                else zero
            )
        else:
            contrastive_per_example = zero_vector
            losses["contrastive"] = zero

        source_logits = None
        if self.source_head is not None:
            reversed_hidden = GradientReversal.apply(pooled, adversary_scale)
            source_logits = self.source_head(reversed_hidden)
        if source_logits is not None and source_labels is not None:
            source_per_example = masked_cross_entropy_per_example(source_logits, source_labels)
            losses["source"] = masked_mean(source_per_example, source_labels)
        else:
            source_per_example = zero_vector
            losses["source"] = zero

        if router_outputs:
            losses["router_z"] = torch.stack([item.z_loss for item in router_outputs]).mean()
            losses["balance"] = torch.stack([item.balance_loss for item in router_outputs]).mean()
        else:
            losses["router_z"] = zero
            losses["balance"] = zero

        per_example = (
            lm_per_example
            + self.config.mtp_loss_weight * mtp_per_example
            + self.config.dialect_loss_weight * dialect_per_example
            + self.config.classification_loss_weight * classification_per_example
            + self.config.contrastive_loss_weight * contrastive_per_example
            + self.config.source_adversary_weight * source_per_example
        )
        if example_weights is not None:
            weights = example_weights.to(per_example.dtype)
            supervised_total = (per_example * weights).sum() / weights.sum().clamp_min(1e-9)
        else:
            supervised_total = per_example.mean()
        total = (
            supervised_total
            + self.config.router_z_loss_weight * losses["router_z"]
            + self.config.balance_loss_weight * losses["balance"]
        )
        return {
            "loss": total,
            "per_example_loss": per_example,
            "losses": losses,
            "logits": logits,
            "hidden": hidden,
            "classification_logits": classification_logits,
            "dialect_logits": dialect_logits,
            "routing_dialect_logits": routing_dialect_logits,
            "source_logits": source_logits,
            "routing": router_outputs,
            "past_key_values": present_key_values if use_cache else None,
        }

    @torch.no_grad()
    def update_router_biases(self) -> None:
        for module in self.modules():
            if isinstance(module, SparseMoE):
                module.update_dynamic_bias()

    def upcycle_bank_constraint_strength(self) -> float:
        """Return the deterministic cross-bank routing curriculum strength."""

        if self.config.architecture not in ("standard_moe", "boichitro_moe"):
            return 0.0
        start = self.config.banked_upcycle_fraction
        release = self.config.banked_upcycle_release_fraction
        progress = self.training_progress
        if start > 0.0 and progress <= start:
            return 1.0
        if release <= start or progress >= release:
            return 0.0
        return (release - progress) / (release - start)

    def parameter_report(self) -> dict[str, int | float]:
        total = sum(parameter.numel() for parameter in self.parameters())
        inactive = 0
        for module in self.modules():
            if isinstance(module, SparseMoE):
                inactive += module.total_expert_parameters() - module.active_expert_parameters()
        active = total - inactive
        return {
            "total_parameters": total,
            "active_parameters_per_token": active,
            "inactive_expert_parameters_per_token": inactive,
            "active_fraction": active / total,
        }


@torch.no_grad()
def upcycle_dense_to_moe(
    dense: BoichitroForMultiTask,
    moe: BoichitroForMultiTask,
    *,
    noise_std: float = 1e-5,
    router_init_std: float = 0.0,
) -> dict[str, Any]:
    """Copy a dense checkpoint into either Switch or shared top-2 experts.

    Switch receives identical full-width expert clones. The shared top-2 model
    partitions the dense FFN into S/A/B banks and reconstructs it exactly while
    the router is still bank-constrained.
    """

    if dense.config.architecture != "dense" or moe.config.architecture == "dense":
        raise ValueError("Expected a dense source and MoE destination")
    if dense.config.d_model != moe.config.d_model or dense.config.n_layers != moe.config.n_layers:
        raise ValueError("Dense and MoE backbone dimensions differ")
    switch_clone = moe.config.architecture == "switch"
    if switch_clone:
        if moe.config.shared_expert or moe.config.top_k != 1:
            raise ValueError("Switch upcycling requires top_k=1 and no shared expert")
        if dense.config.dense_ffn_dim != moe.config.expert_dim:
            raise ValueError("Switch experts must match the full dense FFN width")
    elif dense.config.dense_ffn_dim != 3 * moe.config.expert_dim:
        raise ValueError("Shared top-2 upcycling requires dense_ffn_dim == 3 * expert_dim")

    destination = moe.state_dict()
    source = dense.state_dict()
    copied: list[str] = []
    for name, tensor in source.items():
        if ".ffn." in name or name not in destination or destination[name].shape != tensor.shape:
            continue
        destination[name].copy_(tensor)
        copied.append(name)

    for index in range(moe.config.dense_prefix_layers):
        destination_prefix = moe.layers[index].ffn
        source_ffn = dense.layers[index].ffn
        assert isinstance(destination_prefix, SwiGLU) and isinstance(source_ffn, SwiGLU)
        destination_prefix.load_state_dict(source_ffn.state_dict())

    for index in range(moe.config.dense_prefix_layers, moe.config.n_layers):
        source_ffn = dense.layers[index].ffn
        destination_moe = moe.layers[index].ffn
        assert isinstance(source_ffn, SwiGLU) and isinstance(destination_moe, SparseMoE)
        if switch_clone:
            for expert in destination_moe.experts:
                expert.load_state_dict(source_ffn.state_dict())
                if noise_std:
                    expert.gate_proj.weight.add_(
                        torch.randn_like(expert.gate_proj.weight) * noise_std
                    )
                    expert.up_proj.weight.add_(
                        torch.randn_like(expert.up_proj.weight) * noise_std
                    )
                    expert.down_proj.weight.add_(
                        torch.randn_like(expert.down_proj.weight) * noise_std
                    )
            if router_init_std > 0.0:
                nn.init.normal_(
                    destination_moe.router.weight,
                    mean=0.0,
                    std=router_init_std,
                )
            else:
                nn.init.zeros_(destination_moe.router.weight)
            destination_moe.dynamic_bias.zero_()
            continue
        width = moe.config.expert_dim
        chunks = []
        for chunk_index in range(3):
            start, stop = chunk_index * width, (chunk_index + 1) * width
            chunks.append(
                {
                    "gate": source_ffn.gate_proj.weight[start:stop],
                    "up": source_ffn.up_proj.weight[start:stop],
                    "down": source_ffn.down_proj.weight[:, start:stop],
                }
            )
        if destination_moe.shared_expert is not None:
            destination_moe.shared_expert.gate_proj.weight.copy_(chunks[0]["gate"])
            destination_moe.shared_expert.up_proj.weight.copy_(chunks[0]["up"])
            destination_moe.shared_expert.down_proj.weight.copy_(chunks[0]["down"])
        for expert_index, expert in enumerate(destination_moe.experts):
            bank = 1 if expert_index < len(destination_moe.experts) // 2 else 2
            expert.gate_proj.weight.copy_(chunks[bank]["gate"])
            expert.up_proj.weight.copy_(chunks[bank]["up"])
            # Top-2 router weights sum to one; x2 reconstructs both routed chunks.
            down = 2.0 * chunks[bank]["down"] if destination_moe.top_k == 2 else chunks[bank]["down"]
            expert.down_proj.weight.copy_(down)
            if noise_std:
                expert.gate_proj.weight.add_(torch.randn_like(expert.gate_proj.weight) * noise_std)
                expert.up_proj.weight.add_(torch.randn_like(expert.up_proj.weight) * noise_std)
                expert.down_proj.weight.add_(torch.randn_like(expert.down_proj.weight) * noise_std)
        nn.init.zeros_(destination_moe.router.weight)
        destination_moe.dynamic_bias.zero_()

    return {
        "copied_non_ffn_tensors": len(copied),
        "upcycled_layers": moe.config.n_layers - moe.config.dense_prefix_layers,
        "noise_std": noise_std,
        "router_init_std": router_init_std,
        "scheme": (
            "full_width_identical_switch_expert_clones"
            if switch_clone
            else "shared_chunk_S_plus_top2_banked_chunks_A_B_with_2x_down_projection"
        ),
    }
