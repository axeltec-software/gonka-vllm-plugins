# SPDX-License-Identifier: Apache-2.0
"""Pure mixing-policy functions, ported VERBATIM from Ilya Slavutin's in-tree
port (axeltec-software/vllm @ poc-decode-0.25, vllm/poc/mixed_decode.py).
Kept pure/unit-testable exactly as authored; only the import home changed.
"""

# Bound on consecutive chat-prefill defers before a decoding PoC is forced an
# exclusive step (fairness valve — keeps PoC from starving under chat churn).
POC_DEFER_LIMIT = 4


def poc_is_pure_path(poc_params) -> bool:
    """True for prefill-only PoC (max_tokens == 0), which has no decode loop. All
    decode — generation and validation — runs step-driven. Pure (unit-testable)."""
    return poc_params.max_tokens == 0


def decode_only_mixing_gate(
    *,
    mixed_cudagraph: bool,
    poc_decode_pending: bool,
    poc_will_prefill: bool,
    chat_will_prefill: bool,
    consecutive_defers: int,
    defer_limit: int = POC_DEFER_LIMIT,
) -> tuple[bool, bool, int]:
    """Decide (defer_chat, defer_poc, consecutive_defers) so chat and PoC share a
    forward only when both decode; prefills run isolated. Mutually exclusive defers.
    Pure (unit-testable). With mixed_cudagraph=False reduces to the original
    behaviour (defer_chat=poc_decode_pending, defer_poc=False). The valve bounds
    consecutive chat-prefill defers so chat churn can't starve a decoding PoC.
    """
    defer_chat = poc_decode_pending or (mixed_cudagraph and poc_will_prefill)
    defer_poc = mixed_cudagraph and (not defer_chat) and chat_will_prefill
    if defer_poc:
        consecutive_defers += 1
        if consecutive_defers > defer_limit:
            # Give the decoding PoC one exclusive (pure-decode, graphable) step.
            defer_poc, defer_chat, consecutive_defers = False, True, 0
    else:
        consecutive_defers = 0
    return defer_chat, defer_poc, consecutive_defers


def poc_step_num_tokens(poc_params, num_computed_tokens: int) -> int:
    """Tokens to schedule for a PoC request this step: mixed decode generation
    prefills seq_len once then 1 token/step; the pure / prefill-only path is a
    single seq_len step. Pure (unit-testable)."""
    if not poc_is_pure_path(poc_params):
        return poc_params.seq_len if num_computed_tokens == 0 else 1
    return poc_params.seq_len


def poc_share_budget(poc_share: float, token_budget: int) -> int:
    """PoC's slice of a step's compute (token) budget. poc_share=0 -> PoC blocked
    this step; 1.0 -> PoC may use the whole budget. Pure (unit-testable)."""
    return int(poc_share * token_budget)


def resolve_poc_max_batch_size(configured: int, max_num_seqs: int) -> int:
    """The per-step PoC nonce cap. `configured` == 0 means AUTO -> use the engine's
    own concurrency limit (max_num_seqs), so PoC fills the batch like inference does
    instead of being pinned to a fixed constant that throttles it on bigger machines.
    Any explicit >0 value is honored verbatim. Pure (unit-testable)."""
    return max_num_seqs if configured == 0 else configured
