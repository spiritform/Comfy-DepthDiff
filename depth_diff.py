import os
import random

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

import folder_paths


def _to_luma(image: torch.Tensor) -> torch.Tensor:
    # image: (B, H, W, C) in [0,1]
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
    "depth_anything_v2_vitl.pth",
    "depth_anything_v2_vitb.pth",
    "depth_anything_v2_vits.pth",
    "depth_anything_v2_vitg.pth",
]


def _run_depth_anything_v2(image: torch.Tensor, ckpt_name: str, resolution: int) -> torch.Tensor:
    try:
        import comfy.model_management as model_management
        from custom_controlnet_aux.depth_anything_v2 import DepthAnythingV2Detector
        from custom_controlnet_aux.util import HWC3
    except ImportError as e:
        raise RuntimeError(
            "depth_mode=True requires comfyui_controlnet_aux (and its DepthAnythingV2 weights) to be installed."
        ) from e

    model = DepthAnythingV2Detector.from_pretrained(filename=ckpt_name).to(model_management.get_torch_device())
    try:
        out_batch = []
        for i in range(image.shape[0]):
            arr = (image[i].detach().cpu().numpy() * 255.0).clip(0, 255).astype("uint8")
            arr = HWC3(arr)
            depth = model(arr, output_type="np", detect_resolution=resolution, max_depth=1)
            depth = HWC3(depth)
            out_batch.append(torch.from_numpy(depth.astype("float32") / 255.0))
        return torch.stack(out_batch, dim=0)
    finally:
        del model


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
                "depth_ckpt": (_DEPTH_CKPTS, {"default": "depth_anything_v2_vitl.pth"}),
                "depth_resolution": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
                "invert": ("BOOLEAN", {"default": True}),
                "black_point": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "white_point": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "gamma": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.05}),
                "brightness": ("FLOAT", {"default": 0.0, "min": -1.0, "max": 1.0, "step": 0.01}),
                "contrast": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 4.0, "step": 0.05}),
                "blur_radius": ("INT", {"default": 0, "min": 0, "max": 128, "step": 1}),
                "strength": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 1.0, "step": 0.01}),
                "diff_diffusion_multiplier": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.05}),
            },
        }

    RETURN_TYPES = ("MODEL", "LATENT", "MASK", "IMAGE")
    RETURN_NAMES = ("model", "latent", "mask", "preview")
    FUNCTION = "build"
    CATEGORY = "conditioning/depthdiff"
    OUTPUT_NODE = True

    def build(self, model, image, vae, depth_mode, depth_ckpt, depth_resolution,
              invert, black_point, white_point, gamma,
              brightness, contrast, blur_radius, strength,
              diff_diffusion_multiplier):
        latent = {"samples": vae.encode(image[:, :, :, :3])}

        mask_source = image
        if depth_mode:
            mask_source = _run_depth_anything_v2(image, depth_ckpt, int(depth_resolution))

        m = _to_luma(mask_source).clamp(0.0, 1.0)

        if invert:
            m = 1.0 - m

        bp, wp = float(black_point), float(white_point)
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

        return {"ui": {"images": ui_images}, "result": (model_out, latent_out, m, preview)}


NODE_CLASS_MAPPINGS = {
    "DepthDiff": DepthDiff,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "DepthDiff": "DepthDiff",
}
