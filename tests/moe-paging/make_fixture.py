import array
import random
import struct
import sys
from math import prod
from pathlib import Path

rng = random.Random(681)
width, hidden, experts, layers, vocab = 512, 1024, 32, 2, 128
path = Path(sys.argv[1])
pack = lambda fmt, *values: struct.pack("<" + fmt, *values)
string = lambda s: pack("Q", len(s.encode())) + s.encode()
pad = lambda n: (-n) % 32

metadata = {
    "general.architecture": "qwen3moe",
    "general.name": "Deterministic Qwen3MoE test fixture",
    "tokenizer.ggml.model": "none",
    "qwen3moe.vocab_size": vocab,
    "qwen3moe.context_length": 128,
    "qwen3moe.embedding_length": width,
    "qwen3moe.feed_forward_length": hidden,
    "qwen3moe.expert_feed_forward_length": hidden,
    "qwen3moe.block_count": layers,
    "qwen3moe.attention.head_count": 8,
    "qwen3moe.attention.head_count_kv": 4,
    "qwen3moe.attention.layer_norm_rms_epsilon": 1e-5,
    "qwen3moe.rope.dimension_count": 64,
    "qwen3moe.expert_count": experts,
    "qwen3moe.expert_used_count": 2,
}
q8_blocks = [pack("e", 0.0004) + rng.randbytes(32) for _ in range(8192)]
tensors = []

def add(name, dims, norm=False):
    tensors.append((name + ".weight", dims, norm))

add("token_embd", (width, vocab))
add("output_norm", (width,), True)
for i in range(layers):
    prefix = f"blk.{i}."
    for name, dims in {
        "attn_norm": (width,), "attn_q_norm": (64,), "attn_k_norm": (64,),
        "ffn_norm": (width,), "attn_q": (width, width), "attn_k": (width, width // 2),
        "attn_v": (width, width // 2), "attn_output": (width, width),
        "ffn_gate_inp": (width, experts), "ffn_gate_exps": (width, hidden, experts),
        "ffn_up_exps": (width, hidden, experts), "ffn_down_exps": (hidden, width, experts),
    }.items():
        add(prefix + name, dims, name.endswith("norm"))

with path.open("wb") as f:
    f.write(b"GGUF" + pack("IQQ", 3, len(tensors), len(metadata)))
    for key, value in metadata.items():
        f.write(string(key))
        if isinstance(value, str):
            f.write(pack("I", 8) + string(value))
        elif isinstance(value, float):
            f.write(pack("If", 6, value))
        else:
            f.write(pack("II", 4, value))
    offset = 0
    for name, dims, norm in tensors:
        kind = 8 if len(dims) == 3 else 0
        size = prod(dims) // 32 * 34 if kind else prod(dims) * 4
        f.write(string(name) + pack("I" + "Q" * len(dims) + "IQ", len(dims), *dims, kind, offset))
        offset += size + pad(size)
    f.write(bytes(pad(f.tell())))
    for name, dims, norm in tensors:
        count = prod(dims)
        if len(dims) == 3:
            blocks_per_expert = prod(dims[:2]) // 32
            for _ in range(experts):
                start = rng.randrange(len(q8_blocks))
                blocks = q8_blocks[start:] + q8_blocks[:start]
                row = b"".join(blocks)
                full, tail = divmod(blocks_per_expert, len(blocks))
                f.write(row * full + row[:tail * 34])
            size = count // 32 * 34
        else:
            data = array.array("f", (1.0 if norm else rng.uniform(-0.07, 0.07) for _ in range(count)))
            if sys.byteorder != "little":
                data.byteswap()
            f.write(data.tobytes())
            size = count * 4
        f.write(bytes(pad(size)))
