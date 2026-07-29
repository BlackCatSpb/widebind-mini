import gguf, torch, os, json, shutil, re
from collections import OrderedDict
from safetensors.torch import save_file
from gguf import dequantize, GGMLQuantizationType

model_dir = r'C:\Users\black\OneDrive\Desktop\EVA-Ai\models\ruadapt_qwen3_4b_openvino_ModelB'
out_dir = r'C:\Users\black\OneDrive\Desktop\qwen3_ruadapt_hf'
os.makedirs(out_dir, exist_ok=True)

reader = gguf.GGUFReader(os.path.join(model_dir, 'qwen3_ruadapr.gguf'))

state_dict = OrderedDict()
for t in reader.tensors:
    name = t.name
    np_arr = dequantize(t.data, GGMLQuantizationType(t.tensor_type)).copy()
    arr = torch.from_numpy(np_arr).reshape(tuple(t.shape))

    m = re.match(r'blk\.(\d+)\.(.+)', name)
    if name == 'output_norm.weight':
        state_dict['model.norm.weight'] = arr
    elif name == 'token_embd.weight':
        state_dict['model.embed_tokens.weight'] = arr.t().contiguous()
    elif m:
        layer_idx = int(m.group(1))
        layer_key = m.group(2)
        hf_name = {
            'attn_norm.weight': f'model.layers.{layer_idx}.input_layernorm.weight',
            'ffn_norm.weight': f'model.layers.{layer_idx}.post_attention_layernorm.weight',
            'attn_q.weight': f'model.layers.{layer_idx}.self_attn.q_proj.weight',
            'attn_k.weight': f'model.layers.{layer_idx}.self_attn.k_proj.weight',
            'attn_v.weight': f'model.layers.{layer_idx}.self_attn.v_proj.weight',
            'attn_output.weight': f'model.layers.{layer_idx}.self_attn.o_proj.weight',
            'attn_q_norm.weight': f'model.layers.{layer_idx}.self_attn.q_norm.weight',
            'attn_k_norm.weight': f'model.layers.{layer_idx}.self_attn.k_norm.weight',
            'ffn_gate.weight': f'model.layers.{layer_idx}.mlp.gate_proj.weight',
            'ffn_up.weight': f'model.layers.{layer_idx}.mlp.up_proj.weight',
            'ffn_down.weight': f'model.layers.{layer_idx}.mlp.down_proj.weight',
        }.get(layer_key)
        if hf_name:
            state_dict[hf_name] = arr

state_dict['lm_head.weight'] = state_dict['model.embed_tokens.weight'].clone()

vocab_size = state_dict['model.embed_tokens.weight'].shape[0]
cfg = {
    'architectures': ['Qwen3ForCausalLM'],
    'model_type': 'qwen3',
    'hidden_size': 2560,
    'num_hidden_layers': 36,
    'num_attention_heads': 32,
    'num_key_value_heads': 8,
    'head_dim': 128,
    'intermediate_size': 9728,
    'vocab_size': vocab_size,
    'max_position_embeddings': 40960,
    'rms_norm_eps': 1e-6,
    'rope_theta': 1000000.0,
    'hidden_act': 'silu',
    'tie_word_embeddings': True,
    'attention_bias': False,
    'attention_dropout': 0.0,
    'bos_token_id': 151643,
    'eos_token_id': 151645,
    'use_cache': True,
}
with open(os.path.join(out_dir, 'config.json'), 'w') as f:
    json.dump(cfg, f, indent=2)

save_file(state_dict, os.path.join(out_dir, 'model.safetensors'))

for fn in ['tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json',
           'added_tokens.json', 'merges.txt', 'vocab.json', 'chat_template.jinja']:
    src = os.path.join(model_dir, fn)
    if os.path.exists(src):
        shutil.copy2(src, os.path.join(out_dir, fn))

total_gb = sum(v.numel() * v.element_size() for v in state_dict.values()) / 1e9
print(f'Saved HF model to {out_dir}')
print(f'  Vocab: {vocab_size}')
print(f'  Tensors: {len(state_dict)}')
print(f'  Size: {total_gb:.2f}GB')
