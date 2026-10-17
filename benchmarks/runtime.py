"""Model setup, configuration and validation helpers outside the timed request."""
from datetime import timedelta
import os
from pathlib import Path
import random


def forward_layout(config):
    branches = ['conditional', 'unconditional'] if abs(config['guidance_weight'] - 1.0) > 1e-6 else ['conditional']
    return [dict(forward_index=step*len(branches)+branch, scheduler_step_index=step, branch=name)
            for step in range(config['num_steps']) for branch, name in enumerate(branches)]


def reset_random(torch, np, seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def configure_caches(directory, rank):
    paths = {}
    for key, name in [('TRITON_CACHE_DIR', 'triton'), ('TORCHINDUCTOR_CACHE_DIR', 'inductor'),
                      ('CUDA_CACHE_PATH', 'cuda')]:
        path = directory / 'cache' / f'rank{rank}' / name
        path.mkdir(parents=True, exist_ok=True)
        os.environ[key] = paths[key] = str(path)
    return paths


def load_pipeline(plan, suite):
    import torch
    import torch.distributed as dist
    # Fail before loading weights if the preferred native dependency cannot load.
    import flash_attn  # noqa: F401
    from kandinsky.utils import get_video_pipeline, get_distributed_pipeline
    from kandinsky.models.attention import SelfAttentionEngine
    local_rank = int(os.environ['LOCAL_RANK'])
    torch.cuda.set_device(local_rank)
    device = torch.device('cuda', local_rank)
    dist.init_process_group('nccl', timeout=timedelta(hours=2), device_id=device)
    pipe = get_video_pipeline(device_map=device, mode='t2v',
        conf_path=str(suite / 'model.yaml'), cache_dir=str(suite / 'cache/assets'),
        offload=False, magcache=False, quantized_qwen=False, text_token_padding=False,
        attention_engine=plan['generation']['attention_engine'])
    pipe = get_distributed_pipeline(pipe, tp_size=dist.get_world_size(), mode='t2v')
    pipe.dit.eval()
    backend = SelfAttentionEngine(plan['generation']['attention_engine']).get_attention()
    return pipe, dict(module=backend.__module__, name=backend.__name__)


class VAEObservation:
    """Record the native memory-dependent tiling decision; never change tile/stride."""
    def __init__(self, vae):
        self.vae = vae
        self.original_tiling = vae.get_dec_optimal_tiling
        self.original_decode = vae.decode
        self.tiles = []
        self.validation = []
        vae.get_dec_optimal_tiling = self.tiling

    def tiling(self, shape):
        tile, stride = self.original_tiling(shape)
        self.tiles.append(dict(latent_shape=list(shape), tile=list(tile), stride=list(stride)))
        return tile, stride

    def decode(self, latents, *args, **kwargs):
        output = self.original_decode(latents, *args, **kwargs)
        self.validation.extend([latents, output.sample])
        return output

    def prepare(self, warmup):
        self.tiles.clear()
        self.validation.clear()
        # Floating tensors only survive through E2E for excluded warmups.
        # Formal requests use the native decode method and tensor lifetimes.
        self.vae.decode = self.decode if warmup else self.original_decode

    def restore(self):
        self.vae.get_dec_optimal_tiling = self.original_tiling
        self.vae.decode = self.original_decode
        self.validation.clear()


def video_metadata(path):
    import av
    with av.open(str(path)) as container:
        stream = container.streams.video[0]
        return dict(frames=sum(1 for _ in container.decode(video=0)), fps=float(stream.average_rate),
                    width=stream.width, height=stream.height, codec=stream.codec_context.name)
