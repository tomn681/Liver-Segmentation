# -*- coding: utf-8 -*-
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
from functools import partial
from pathlib import Path
import urllib.request
import torch

from .modeling import (
    ImageEncoderViT,
    MaskDecoder,
    PromptEncoder,
    Sam,
    TwoWayTransformer,
)


def build_sam_vit_h(checkpoint=None,img_size=512,in_chans = 3):
    return _build_sam(
        image_size=img_size,
        in_chans = in_chans,
        encoder_embed_dim=1280,
        encoder_depth=32,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[7, 15, 23, 31],
        checkpoint=checkpoint,
    )


build_sam = build_sam_vit_h


def build_sam_vit_l(checkpoint=None,img_size=512,in_chans = 3):
    return _build_sam(
        image_size=img_size,
        in_chans = in_chans,
        encoder_embed_dim=1024,
        encoder_depth=24,
        encoder_num_heads=16,
        encoder_global_attn_indexes=[5, 11, 17, 23],
        checkpoint=checkpoint,
    )


def build_sam_vit_b(checkpoint=None,img_size=512,in_chans = 3,):
    return _build_sam(
        image_size=img_size,
        in_chans = in_chans,
        encoder_embed_dim=768,
        encoder_depth=12,
        encoder_num_heads=12,
        encoder_global_attn_indexes=[2, 5, 8, 11],
        checkpoint=checkpoint,
    )


sam_model_registry = {
    "default": build_sam_vit_h,
    "vit_h": build_sam_vit_h,
    "vit_l": build_sam_vit_l,
    "vit_b": build_sam_vit_b,
}


def _build_sam(
    image_size,
    in_chans,
    encoder_embed_dim,
    encoder_depth,
    encoder_num_heads,
    encoder_global_attn_indexes,
    checkpoint=None,
):
    prompt_embed_dim = 256
    image_size = image_size
    vit_patch_size = 16
    image_embedding_size = image_size // vit_patch_size
    sam = Sam(
        image_encoder=ImageEncoderViT(
            depth=encoder_depth,
            embed_dim=encoder_embed_dim,
            img_size=image_size,
            in_chans = in_chans,
            mlp_ratio=4,
            norm_layer=partial(torch.nn.LayerNorm, eps=1e-6),
            num_heads=encoder_num_heads,
            patch_size=vit_patch_size,
            qkv_bias=True,
            use_rel_pos=True,
            global_attn_indexes=encoder_global_attn_indexes,
            window_size=14,
            out_chans=prompt_embed_dim,
        ),
        prompt_encoder=PromptEncoder(
            embed_dim=prompt_embed_dim,
            image_embedding_size=(image_embedding_size, image_embedding_size),
            input_image_size=(image_size, image_size),
            mask_in_chans=16,
        ),
        mask_decoder=MaskDecoder(
            num_multimask_outputs=3,
            transformer=TwoWayTransformer(
                depth=2,
                embedding_dim=prompt_embed_dim,
                mlp_dim=2048,
                num_heads=8,
            ),
            transformer_dim=prompt_embed_dim,
            iou_head_depth=3,
            iou_head_hidden_dim=256,
        ),
        pixel_mean=[123.675, 116.28, 103.53],
        pixel_std=[58.395, 57.12, 57.375],
    )
    sam.eval()
    checkpoint = Path(checkpoint)
    if checkpoint.name == "sam_vit_b_01ec64.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_b_01ec64.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == "y":
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-B checkpoint...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_b_01ec64.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")
    elif checkpoint.name == "sam_vit_h_4b8939.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_h_4b8939.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == "y":
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-H checkpoint...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_h_4b8939.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")
    elif checkpoint.name == "sam_vit_l_0b3195.pth" and not checkpoint.exists():
        cmd = input("Download sam_vit_l_0b3195.pth from facebook AI? [y]/n: ")
        if len(cmd) == 0 or cmd.lower() == "y":
            checkpoint.parent.mkdir(parents=True, exist_ok=True)
            print("Downloading SAM ViT-L checkpoint...")
            urllib.request.urlretrieve(
                "https://dl.fbaipublicfiles.com/segment_anything/sam_vit_l_0b3195.pth",
                checkpoint,
            )
            print(checkpoint.name, " is downloaded!")

    if checkpoint is not None:
        with open(checkpoint, "rb") as f:
            loaded_state_dict = torch.load(f)

        state_dict = loaded_state_dict
        model_state_dict = sam.state_dict()
        is_changed = False
        for k in state_dict:
            if k in model_state_dict:
                if state_dict[k].shape != model_state_dict[k].shape:
                    print(f"Skip loading parameter: {k}, "
                                f"required shape: {model_state_dict[k].shape}, "
                                f"loaded shape: {state_dict[k].shape}")
                    state_dict[k] = model_state_dict[k]
                    is_changed = True
            else:
                print(f"Dropping parameter {k}")
                is_changed = True
        sam.load_state_dict(loaded_state_dict, strict=False)
    return sam
