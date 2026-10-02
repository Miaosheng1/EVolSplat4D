from unidepth.models.backbones import _make_dinov2_model

def dinov2_vits14(config, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-S/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    vit = _make_dinov2_model(
        arch_name="vit_small",
        pretrained=config["pretrained"],
        output_idx=config.get("output_idx", [3, 6, 9, 12]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        num_register_tokens=config.get("num_register_tokens", 0),
        use_norm=config.get("use_norm", False),
        export=config.get("export", False),
        interpolate_offset=config.get("interpolate_offset", 0.0),
        **kwargs,
    )
    return vit


def dinov2_vitb14(config, pretrained: bool = True, **kwargs):
    """
    DINOv2 ViT-B/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    vit = _make_dinov2_model(
        arch_name="vit_base",
        pretrained=config["pretrained"],
        output_idx=config.get("output_idx", [3, 6, 9, 12]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        num_register_tokens=config.get("num_register_tokens", 0),
        use_norm=config.get("use_norm", False),
        export=config.get("export", False),
        interpolate_offset=config.get("interpolate_offset", 0.0),
        **kwargs,
    )
    return vit


def dinov2_vitl14(config, pretrained: str = "", **kwargs):
    """
    DINOv2 ViT-L/14 model (optionally) pretrained on the LVD-142M dataset.
    """
    vit = _make_dinov2_model(
        arch_name="vit_large",
        pretrained=config["pretrained"],
        output_idx=config.get("output_idx", [5, 12, 18, 24]),
        checkpoint=config.get("use_checkpoint", False),
        drop_path_rate=config.get("drop_path", 0.0),
        num_register_tokens=config.get("num_register_tokens", 0),
        use_norm=config.get("use_norm", False),
        export=config.get("export", False),
        interpolate_offset=config.get("interpolate_offset", 0.0),
        **kwargs,
    )
    return vit
