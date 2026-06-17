import os

import torch

from mae_st.util.logging import master_print as print


DEFAULT_2D_CKPT_DIR = "/scratch/individuals/renao/code/Data_preprocessing/ckpt"

MODEL_TO_2D_PRETRAIN = {
    "mae_vit_base_patch16": ("CONCH", "Couch.bin", "visual.trunk."),
    "mae_vit_large_patch16": ("UNI", "uni.bin", ""),
    "mae_vit_huge_patch14": ("H-optimus-1", "H-optimus-1.pth", ""),
}


def get_2d_pretrain_spec(model_name, ckpt_dir=DEFAULT_2D_CKPT_DIR):
    if model_name not in MODEL_TO_2D_PRETRAIN:
        raise ValueError(
            "No default 2D pretrain mapping for {}. Available: {}".format(
                model_name, sorted(MODEL_TO_2D_PRETRAIN)
            )
        )
    source, filename, prefix = MODEL_TO_2D_PRETRAIN[model_name]
    return source, os.path.join(ckpt_dir, filename), prefix


def _unwrap_state_dict(checkpoint):
    if not isinstance(checkpoint, dict):
        return checkpoint
    for key in ("state_dict", "model", "model_state"):
        if key in checkpoint and isinstance(checkpoint[key], dict):
            return checkpoint[key]
    return checkpoint


def _strip_prefix(name, prefix):
    if prefix and name.startswith(prefix):
        return name[len(prefix) :]
    return name


def _is_skip_key(name):
    skip_tokens = (
        "pos_embed",
        "cls_token",
        "reg_token",
        "register",
        "head.",
        "fc_norm",
        "mask_token",
        "decoder",
        "text.",
        "visual.proj",
        "logit_scale",
    )
    return any(token in name for token in skip_tokens)


def _inflate_patch_embed(weight_2d, target_shape):
    depth = target_shape[2]
    return weight_2d.unsqueeze(2).repeat(1, 1, depth, 1, 1) / depth


def load_2d_pretrained_weights(
    model,
    model_name,
    ckpt_dir=DEFAULT_2D_CKPT_DIR,
    ckpt_path="",
    source="auto",
):
    """Load matching 2D ViT encoder weights into a 3D MAE encoder."""
    if source == "auto":
        source, default_path, prefix = get_2d_pretrain_spec(model_name, ckpt_dir)
        ckpt_path = ckpt_path or default_path
    else:
        prefix = MODEL_TO_2D_PRETRAIN.get(model_name, ("", "", ""))[2]
        if not ckpt_path:
            raise ValueError("--pretrained_2d_ckpt is required when source is not auto")

    if not os.path.exists(ckpt_path):
        raise FileNotFoundError("2D pretrained checkpoint not found: {}".format(ckpt_path))

    print("=> Loading 2D {} checkpoint from {}".format(source, ckpt_path))
    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    state_dict_2d = _unwrap_state_dict(checkpoint)
    model_state = model.state_dict()
    load_state = {}
    inflated = []
    copied = []
    skipped_shape = []

    for raw_name, param in state_dict_2d.items():
        name = _strip_prefix(raw_name, prefix)
        if prefix and raw_name == name:
            continue
        if _is_skip_key(name):
            continue
        if name not in model_state:
            continue

        target_shape = model_state[name].shape
        if name == "patch_embed.proj.weight" and param.ndim == 4:
            if param.shape[:2] == target_shape[:2] and param.shape[2:] == target_shape[3:]:
                load_state[name] = _inflate_patch_embed(param, target_shape)
                inflated.append(name)
            else:
                skipped_shape.append((name, tuple(param.shape), tuple(target_shape)))
        elif tuple(param.shape) == tuple(target_shape):
            load_state[name] = param
            copied.append(name)
        else:
            skipped_shape.append((name, tuple(param.shape), tuple(target_shape)))

    msg = model.load_state_dict(load_state, strict=False)
    print(
        "=> 2D pretrain loaded: copied={} inflated_patch_embed={} missing={} unexpected={}".format(
            len(copied), len(inflated), len(msg.missing_keys), len(msg.unexpected_keys)
        )
    )
    if inflated:
        print("=> Inflated: {}".format(", ".join(inflated)))
    if skipped_shape:
        print("=> Shape-skipped keys: {}".format(len(skipped_shape)))
        for name, src_shape, dst_shape in skipped_shape[:10]:
            print("   {}: {} -> {}".format(name, src_shape, dst_shape))
    return msg
