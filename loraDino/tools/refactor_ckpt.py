
import torch

"""这块是将它提供的ckpt文件中的key进行重命名，以适配加入了lora层后的参数引入"""

# Missing
# key(s) in state_dict:
# "backbone.model.blocks.0.attn.qkv.qkv.weight", "backbone.model.blocks.0.attn.qkv.qkv.bias", "backbone.model.blocks.0.attn.qkv.linear_a_q.weight", "backbone.model.blocks.0.attn.qkv.linear_b_q.weight", "backbone.model.blocks.0.attn.qkv.linear_a_v.weight", "backbone.model.blocks.0.attn.qkv.linear_b_v.weight", "backbone.model.blocks.1.attn.qkv.qkv.weight", "backbone.model.blocks.1.attn.qkv.qkv.bias", "backbone.model.blocks.1.attn.qkv.linear_a_q.weight", "backbone.model.blocks.1.attn.qkv.linear_b_q.weight", "backbone.model.blocks.1.attn.qkv.linear_a_v.weight", "backbone.model.blocks.1.attn.qkv.linear_b_v.weight", "backbone.model.blocks.2.attn.qkv.qkv.weight", "backbone.model.blocks.2.attn.qkv.qkv.bias", "backbone.model.blocks.2.attn.qkv.linear_a_q.weight", "backbone.model.blocks.2.attn.qkv.linear_b_q.weight", "backbone.model.blocks.2.attn.qkv.linear_a_v.weight", "backbone.model.blocks.2.attn.qkv.linear_b_v.weight", "backbone.model.blocks.3.attn.qkv.qkv.weight", "backbone.model.blocks.3.attn.qkv.qkv.bias", "backbone.model.blocks.3.attn.qkv.linear_a_q.weight", "backbone.model.blocks.3.attn.qkv.linear_b_q.weight", "backbone.model.blocks.3.attn.qkv.linear_a_v.weight", "backbone.model.blocks.3.attn.qkv.linear_b_v.weight", "backbone.model.blocks.4.attn.qkv.qkv.weight", "backbone.model.blocks.4.attn.qkv.qkv.bias", "backbone.model.blocks.4.attn.qkv.linear_a_q.weight", "backbone.model.blocks.4.attn.qkv.linear_b_q.weight", "backbone.model.blocks.4.attn.qkv.linear_a_v.weight", "backbone.model.blocks.4.attn.qkv.linear_b_v.weight", "backbone.model.blocks.5.attn.qkv.qkv.weight", "backbone.model.blocks.5.attn.qkv.qkv.bias", "backbone.model.blocks.5.attn.qkv.linear_a_q.weight", "backbone.model.blocks.5.attn.qkv.linear_b_q.weight", "backbone.model.blocks.5.attn.qkv.linear_a_v.weight", "backbone.model.blocks.5.attn.qkv.linear_b_v.weight", "backbone.model.blocks.6.attn.qkv.qkv.weight", "backbone.model.blocks.6.attn.qkv.qkv.bias", "backbone.model.blocks.6.attn.qkv.linear_a_q.weight", "backbone.model.blocks.6.attn.qkv.linear_b_q.weight", "backbone.model.blocks.6.attn.qkv.linear_a_v.weight", "backbone.model.blocks.6.attn.qkv.linear_b_v.weight", "backbone.model.blocks.7.attn.qkv.qkv.weight", "backbone.model.blocks.7.attn.qkv.qkv.bias", "backbone.model.blocks.7.attn.qkv.linear_a_q.weight", "backbone.model.blocks.7.attn.qkv.linear_b_q.weight", "backbone.model.blocks.7.attn.qkv.linear_a_v.weight", "backbone.model.blocks.7.attn.qkv.linear_b_v.weight", "backbone.model.blocks.8.attn.qkv.qkv.weight", "backbone.model.blocks.8.attn.qkv.qkv.bias", "backbone.model.blocks.8.attn.qkv.linear_a_q.weight", "backbone.model.blocks.8.attn.qkv.linear_b_q.weight", "backbone.model.blocks.8.attn.qkv.linear_a_v.weight", "backbone.model.blocks.8.attn.qkv.linear_b_v.weight", "backbone.model.blocks.9.attn.qkv.qkv.weight", "backbone.model.blocks.9.attn.qkv.qkv.bias", "backbone.model.blocks.9.attn.qkv.linear_a_q.weight", "backbone.model.blocks.9.attn.qkv.linear_b_q.weight", "backbone.model.blocks.9.attn.qkv.linear_a_v.weight", "backbone.model.blocks.9.attn.qkv.linear_b_v.weight", "backbone.model.blocks.10.attn.qkv.qkv.weight", "backbone.model.blocks.10.attn.qkv.qkv.bias", "backbone.model.blocks.10.attn.qkv.linear_a_q.weight", "backbone.model.blocks.10.attn.qkv.linear_b_q.weight", "backbone.model.blocks.10.attn.qkv.linear_a_v.weight", "backbone.model.blocks.10.attn.qkv.linear_b_v.weight", "backbone.model.blocks.11.attn.qkv.qkv.weight", "backbone.model.blocks.11.attn.qkv.qkv.bias", "backbone.model.blocks.11.attn.qkv.linear_a_q.weight", "backbone.model.blocks.11.attn.qkv.linear_b_q.weight", "backbone.model.blocks.11.attn.qkv.linear_a_v.weight", "backbone.model.blocks.11.attn.qkv.linear_b_v.weight".
#
# Unexpected
# key(s) in state_dict:
# "backbone.model.blocks.0.attn.qkv.weight", "backbone.model.blocks.0.attn.qkv.bias", "backbone.model.blocks.1.attn.qkv.weight", "backbone.model.blocks.1.attn.qkv.bias", "backbone.model.blocks.2.attn.qkv.weight", "backbone.model.blocks.2.attn.qkv.bias", "backbone.model.blocks.3.attn.qkv.weight", "backbone.model.blocks.3.attn.qkv.bias", "backbone.model.blocks.4.attn.qkv.weight", "backbone.model.blocks.4.attn.qkv.bias", "backbone.model.blocks.5.attn.qkv.weight", "backbone.model.blocks.5.attn.qkv.bias", "backbone.model.blocks.6.attn.qkv.weight", "backbone.model.blocks.6.attn.qkv.bias", "backbone.model.blocks.7.attn.qkv.weight", "backbone.model.blocks.7.attn.qkv.bias", "backbone.model.blocks.8.attn.qkv.weight", "backbone.model.blocks.8.attn.qkv.bias", "backbone.model.blocks.9.attn.qkv.weight", "backbone.model.blocks.9.attn.qkv.bias", "backbone.model.blocks.10.attn.qkv.weight", "backbone.model.blocks.10.attn.qkv.bias", "backbone.model.blocks.11.attn.qkv.weight", "backbone.model.blocks.11.attn.qkv.bias".

checkpoint = torch.load(
    f"/home/jack/wvn/self_supervised_segmentation/models/stego_cocostuff27_vit_base_5_cluster_linear_fine_tuning.ckpt",
    map_location='cuda:0')

# checkpoint_check = torch.load(
#     f"/home/jack/wvn/self_supervised_segmentation/models/zzy_refactor.ckpt",
#     map_location='cuda:0')

# 获取 state_dict（包含模型权重）
state_dict = checkpoint['state_dict']


rename_mapping = {
    "backbone.model.blocks.0.attn.qkv.weight": "backbone.model.blocks.0.attn.qkv.qkv.weight",
    "backbone.model.blocks.0.attn.qkv.bias": "backbone.model.blocks.0.attn.qkv.qkv.bias",
    "backbone.model.blocks.1.attn.qkv.weight": "backbone.model.blocks.1.attn.qkv.qkv.weight",
    "backbone.model.blocks.1.attn.qkv.bias": "backbone.model.blocks.1.attn.qkv.qkv.bias",
    "backbone.model.blocks.2.attn.qkv.weight": "backbone.model.blocks.2.attn.qkv.qkv.weight",
    "backbone.model.blocks.2.attn.qkv.bias": "backbone.model.blocks.2.attn.qkv.qkv.bias",
    "backbone.model.blocks.3.attn.qkv.weight": "backbone.model.blocks.3.attn.qkv.qkv.weight",
    "backbone.model.blocks.3.attn.qkv.bias": "backbone.model.blocks.3.attn.qkv.qkv.bias",
    "backbone.model.blocks.4.attn.qkv.weight": "backbone.model.blocks.4.attn.qkv.qkv.weight",
    "backbone.model.blocks.4.attn.qkv.bias": "backbone.model.blocks.4.attn.qkv.qkv.bias",
    "backbone.model.blocks.5.attn.qkv.weight": "backbone.model.blocks.5.attn.qkv.qkv.weight",
    "backbone.model.blocks.5.attn.qkv.bias": "backbone.model.blocks.5.attn.qkv.qkv.bias",
    "backbone.model.blocks.6.attn.qkv.weight": "backbone.model.blocks.6.attn.qkv.qkv.weight",
    "backbone.model.blocks.6.attn.qkv.bias": "backbone.model.blocks.6.attn.qkv.qkv.bias",
    "backbone.model.blocks.7.attn.qkv.weight": "backbone.model.blocks.7.attn.qkv.qkv.weight",
    "backbone.model.blocks.7.attn.qkv.bias": "backbone.model.blocks.7.attn.qkv.qkv.bias",
    "backbone.model.blocks.8.attn.qkv.weight": "backbone.model.blocks.8.attn.qkv.qkv.weight",
    "backbone.model.blocks.8.attn.qkv.bias": "backbone.model.blocks.8.attn.qkv.qkv.bias",
    "backbone.model.blocks.9.attn.qkv.weight": "backbone.model.blocks.9.attn.qkv.qkv.weight",
    "backbone.model.blocks.9.attn.qkv.bias": "backbone.model.blocks.9.attn.qkv.qkv.bias",
    "backbone.model.blocks.10.attn.qkv.weight": "backbone.model.blocks.10.attn.qkv.qkv.weight",
    "backbone.model.blocks.10.attn.qkv.bias": "backbone.model.blocks.10.attn.qkv.qkv.bias",
    "backbone.model.blocks.11.attn.qkv.weight": "backbone.model.blocks.11.attn.qkv.qkv.weight",
    "backbone.model.blocks.11.attn.qkv.bias": "backbone.model.blocks.11.attn.qkv.qkv.bias",
    }

new_state_dict = {}
for key, value in state_dict.items():
    # 如果当前键在重命名映射中，进行重命名
    if key in rename_mapping:
        new_state_dict[rename_mapping[key]] = value
    else:
        new_state_dict[key] = value  # 保留原键

# 3. 将重命名后的 state_dict 写入原始 checkpoint 中
checkpoint['state_dict'] = new_state_dict

save_path = f"/home/jack/wvn/self_supervised_segmentation/models/zzy_refactor.ckpt"
torch.save(checkpoint, save_path)
print(f"Checkpoint with renamed keys saved to {save_path}")


# Missing
# key(s) in state_dict:
# "backbone.model.blocks.0.attn.qkv.linear_a_q.weight",
# "backbone.model.blocks.0.attn.qkv.linear_b_q.weight",
# "backbone.model.blocks.0.attn.qkv.linear_a_v.weight",
# "backbone.model.blocks.0.attn.qkv.linear_b_v.weight",
# "backbone.model.blocks.1.attn.qkv.linear_a_q.weight",
# "backbone.model.blocks.1.attn.qkv.linear_b_q.weight",
# "backbone.model.blocks.1.attn.qkv.linear_a_v.weight",
# "backbone.model.blocks.1.attn.qkv.linear_b_v.weight",
# "backbone.model.blocks.2.attn.qkv.linear_a_q.weight",
# "backbone.model.blocks.2.attn.qkv.linear_b_q.weight",
# "backbone.model.blocks.2.attn.qkv.linear_a_v.weight",
# "backbone.model.blocks.2.attn.qkv.linear_b_v.weight",
# "backbone.model.blocks.3.attn.qkv.linear_a_q.weight",
# "backbone.model.blocks.3.attn.qkv.linear_b_q.weight",
# "backbone.model.blocks.3.attn.qkv.linear_a_v.weight",
# "backbone.model.blocks.3.attn.qkv.linear_b_v.weight",
# "backbone.model.blocks.4.attn.qkv.linear_a_q.weight",
# "backbone.model.blocks.4.attn.qkv.linear_b_q.weight",
# "backbone.model.blocks.4.attn.qkv.linear_a_v.weight",
# "backbone.model.blocks.4.attn.qkv.linear_b_v.weight",
# "backbone.model.blocks.5.attn.qkv.linear_a_q.weight",
# "backbone.model.blocks.5.attn.qkv.linear_b_q.weight",
# "backbone.model.blocks.5.attn.qkv.linear_a_v.weight",
# "backbone.model.blocks.5.attn.qkv.linear_b_v.weight",
# "backbone.model.blocks.6.attn.qkv.linear_a_q.weight",
# "backbone.model.blocks.6.attn.qkv.linear_b_q.weight",
# "backbone.model.blocks.6.attn.qkv.linear_a_v.weight",
# "backbone.model.blocks.6.attn.qkv.linear_b_v.weight",
# "backbone.model.blocks.7.attn.qkv.linear_a_q.weight",
# "backbone.model.blocks.7.attn.qkv.linear_b_q.weight",
# "backbone.model.blocks.7.attn.qkv.linear_a_v.weight",
# "backbone.model.blocks.7.attn.qkv.linear_b_v.weight",
# "backbone.model.blocks.8.attn.qkv.linear_a_q.weight",
# "backbone.model.blocks.8.attn.qkv.linear_b_q.weight",
# "backbone.model.blocks.8.attn.qkv.linear_a_v.weight",
# "backbone.model.blocks.8.attn.qkv.linear_b_v.weight",
# "backbone.model.blocks.9.attn.qkv.linear_a_q.weight",
# "backbone.model.blocks.9.attn.qkv.linear_b_q.weight",
# "backbone.model.blocks.9.attn.qkv.linear_a_v.weight",
# "backbone.model.blocks.9.attn.qkv.linear_b_v.weight",
# "backbone.model.blocks.10.attn.qkv.linear_a_q.weight",
# "backbone.model.blocks.10.attn.qkv.linear_b_q.weight",
# "backbone.model.blocks.10.attn.qkv.linear_a_v.weight",
# "backbone.model.blocks.10.attn.qkv.linear_b_v.weight",
# "backbone.model.blocks.11.attn.qkv.linear_a_q.weight",
# "backbone.model.blocks.11.attn.qkv.linear_b_q.weight",
# "backbone.model.blocks.11.attn.qkv.linear_a_v.weight",
# "backbone.model.blocks.11.attn.qkv.linear_b_v.weight".
