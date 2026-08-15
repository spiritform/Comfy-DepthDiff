import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import folder_paths


def _to_luma(image: torch.Tensor) -> torch.Tensor:
    if image.shape[-1] == 1:
        return image[..., 0]
    r, g, b = image[..., 0], image[..., 1], image[..., 2]
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _gaussian_blur(mask: torch.Tensor, radius: int) -> torch.Tensor:
    if radius <= 0:
        return mask
    sigma = max(radius / 2.0, 0.1)
    ksize = radius * 2 + 1
    coords = torch.arange(ksize, dtype=mask.dtype, device=mask.device) - radius
    kernel_1d = torch.exp(-(coords ** 2) / (2 * sigma ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = kernel_1d[:, None] * kernel_1d[None, :]
    kernel_2d = kernel_2d.view(1, 1, ksize, ksize)
    x = mask.unsqueeze(1)
    x = F.pad(x, (radius, radius, radius, radius), mode="reflect")
    x = F.conv2d(x, kernel_2d)
    return x.squeeze(1)


_DEPTH_CKPTS = [
    "depth_anything_v2_vitl_fp32.safetensors",
    "depth_anything_v2_vitl_fp16.safetensors",
    "depth_anything_v2_vitb_fp32.safetensors",
    "depth_anything_v2_vitb_fp16.safetensors",
    "depth_anything_v2_vits_fp32.safetensors",
    "depth_anything_v2_vits_fp16.safetensors",
    "depth_anything_v2_vitg_fp32.safetensors",
]


def _run_depth_anything_v2_kijai(image: torch.Tensor, ckpt_name: str) -> torch.Tensor:
    try:
        from nodes import NODE_CLASS_MAPPINGS
    except ImportError:
        NODE_CLASS_MAPPINGS = {}

    Loader = NODE_CLASS_MAPPINGS.get("DownloadAndLoadDepthAnythingV2Model")
    Inference = NODE_CLASS_MAPPINGS.get("DepthAnything_V2")
    if Loader is None or Inference is None:
        raise RuntimeError(
            "depth_mode requires Kijai's comfyui-depthanythingv2 pack. "
            "Install via Manager (search 'DepthAnythingV2')."
        )

    (da_model,) = Loader().loadmodel(model=ckpt_name)
    (depth_image,) = Inference().process(da_model=da_model, images=image)
    return depth_image


def _make_diff_diffusion_fn(multiplier: float):
    def forward(sigma, denoise_mask, extra_options):
        model = extra_options["model"]
        step_sigmas = extra_options["sigmas"]
        sigma_to = model.inner_model.model_sampling.sigma_min
        if step_sigmas[-1] > sigma_to:
            sigma_to = step_sigmas[-1]
        sigma_from = step_sigmas[0]

        ts_from = model.inner_model.model_sampling.timestep(sigma_from)
        ts_to = model.inner_model.model_sampling.timestep(sigma_to)
        current_ts = model.inner_model.model_sampling.timestep(sigma[0])

        threshold = (current_ts - ts_to) / (ts_from - ts_to)
        return (denoise_mask * multiplier >= threshold).to(denoise_mask.dtype)
    return forward


class DepthDiff:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model": ("MODEL",),
                "image": ("IMAGE",),
                "vae": ("VAE",),
                "depth_mode": ("BOOLEAN", {"default": False}),
                "depth_ckpt": (_DEPTH_CKPTS, {"default": "depth_anything_v2_vitl_fp32.safetensors"}),
                "depth_max_size": ("INT", {"default": 1024, "min": 256, "max": 4096, "step": 64}),
                "invert": ("BOOLEAN", {"default": True}),
                "input_black": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 255.0, "step": 1.0}),
                "input_white": ("FLOAT", {"default": 255.0, "min": 0.0, "max": 255.0, "step": 1.0}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.01}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "blur_radius": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "diff_diffusion_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("MODEL", "LATENT", "MASK")
    RETURN_NAMES = ("model", "latent", "mask")
    FUNCTION = "build"
    CATEGORY = "conditioning/depthdiff"
    OUTPUT_NODE = True

    def build(self, model, image, vae, depth_mode, depth_ckpt, depth_max_size,
              invert, input_black, input_white, gamma,
              brightness, contrast, blur_radius, strength,
              diff_diffusion_multiplier, mask=None):
        latent = {"samples": vae.encode(image[:, :, :, :3])}

        mask_source = image
        if depth_mode:
            orig_h, orig_w = image.shape[1], image.shape[2]
            long_side = max(orig_h, orig_w)
            if long_side > depth_max_size:
                scale = depth_max_size / long_side
                new_h = max(1, int(round(orig_h * scale)))
                new_w = max(1, int(round(orig_w * scale)))
                small = F.interpolate(
                    image.permute(0, 3, 1, 2), size=(new_h, new_w),
                    mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1).contiguous()
                depth_small = _run_depth_anything_v2_kijai(small, depth_ckpt)
                mask_source = F.interpolate(
                    depth_small.permute(0, 3, 1, 2), size=(orig_h, orig_w),
                    mode="bilinear", align_corners=False
                ).permute(0, 2, 3, 1).contiguous()
            else:
                mask_source = _run_depth_anything_v2_kijai(image, depth_ckpt)

        m = _to_luma(mask_source).clamp(0.0, 1.0)

        if invert:
            m = 1.0 - m

        bp, wp = float(input_black) / 255.0, float(input_white) / 255.0
        if wp <= bp:
            wp = bp + 1e-4
        m = ((m - bp) / (wp - bp)).clamp(0.0, 1.0)

        if gamma != 1.0:
            m = m.pow(1.0 / max(gamma, 1e-4))

        if contrast != 1.0:
            m = ((m - 0.5) * contrast + 0.5)
        if brightness != 0.0:
            m = m + brightness
        m = m.clamp(0.0, 1.0)

        m = _gaussian_blur(m, int(blur_radius))
        m = (m * float(strength)).clamp(0.0, 1.0)

        if mask is not None:
            gate = mask
            if gate.dim() == 2:
                gate = gate.unsqueeze(0)
            gate = gate.to(m.device, m.dtype)
            if gate.shape[-2:] != m.shape[-2:]:
                gate = F.interpolate(
                    gate.unsqueeze(1), size=m.shape[-2:], mode="bilinear", align_corners=False
                ).squeeze(1)
            if gate.shape[0] != m.shape[0]:
                if gate.shape[0] == 1:
                    gate = gate.expand(m.shape[0], -1, -1)
                else:
                    gate = gate[:m.shape[0]]
            m = (m * gate).clamp(0.0, 1.0)

        preview = m.unsqueeze(-1).repeat(1, 1, 1, 3)

        model_out = model.clone()
        model_out.set_model_denoise_mask_function(
            _make_diff_diffusion_fn(float(diff_diffusion_multiplier))
        )

        latent_out = latent.copy()
        latent_out["noise_mask"] = m.reshape(-1, 1, m.shape[-2], m.shape[-1])

        temp_dir = folder_paths.get_temp_directory()
        os.makedirs(temp_dir, exist_ok=True)
        ui_images = []
        for i in range(preview.shape[0]):
            arr = (preview[i].detach().cpu().clamp(0, 1).numpy() * 255.0).astype(np.uint8)
            img = Image.fromarray(arr)
            fname = f"luma_mask_{random.randint(0, 0xFFFFFFFF):08x}_{i:03d}.png"
            img.save(os.path.join(temp_dir, fname), compress_level=1)
            ui_images.append({"filename": fname, "subfolder": "", "type": "temp"})

        return {"ui": {"images": ui_images}, "result": (model_out, latent_out, m)}


NODE_CLASS_MAPPINGS = {
    "DepthDiff": DepthDiff,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthDiff": "DepthDiff",
}
