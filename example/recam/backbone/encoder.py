
import torch

from example.recam.backbone.wan_prompter import WanPrompter
from example.recam.backbone.wan_video_text_encoder import WanTextEncoder
from example.recam.backbone.wan_video_vae import WanVideoVAE
from example.recam.backbone.wan_video_vae2_2 import WanVideoVAE_2_2
from example.recam.config.config import EncoderConfig

class WanEncoder:
    """Wan video encoder that turns inputs into model-ready representations."""
    
    def __init__(self, cfg: EncoderConfig, device: torch.device, dtype=torch.bfloat16) -> None:
        self.cfg = cfg
        self.device = device
        self.dtype = dtype

        self.tiler_kwargs = dict(
            tiled=self.cfg.vae.use_tile,
            tile_size=(34, 34),
            tile_stride=(18, 16),
        )
        if self.cfg.vae.type == "WanVideoVAE_2_1":
            self.vae = WanVideoVAE().to(device=self.device, dtype=self.dtype).eval().requires_grad_(False)
            self.compression = (4,8,8)
        else:
            self.vae = WanVideoVAE_2_2().to(device=self.device, dtype=self.dtype).eval().requires_grad_(False)
            self.compression = (4,16,16)
        self.vae.model.load_state_dict(torch.load(self.cfg.vae.path, map_location='cpu', weights_only=False), strict=True)
        if self.cfg.vae.torch_compile:
            self.vae = torch.compile(self.vae)

        with torch.device('meta'):
            self.text_encoder = WanTextEncoder().to(dtype=self.dtype)
        self.text_encoder.to_empty(device=self.device)
        self.text_encoder.load_state_dict(torch.load(self.cfg.text_encoder.path, map_location='cpu', weights_only=False), strict=True)
        self.text_encoder.eval().requires_grad_(False).to(device=self.device)
        self.prompter = WanPrompter()
        self.prompter.fetch_models(self.text_encoder)
        self.prompter.fetch_tokenizer(self.cfg.text_encoder.tokenizer_path)
        if self.cfg.text_encoder.torch_compile:
            self.text_encoder = torch.compile(self.text_encoder)

    def encode_prompt(self, prompt: list[str]) -> torch.Tensor:
        with torch.no_grad():
            context = self.prompter.encode_prompt(prompt, device=self.device)
            return context

    def encode_video(self, video: torch.tensor) -> torch.Tensor:
        with torch.no_grad():
            video = video.to(self.device, self.dtype)
            latent = self.vae.encode(video, device=self.device, **self.tiler_kwargs)
            return latent