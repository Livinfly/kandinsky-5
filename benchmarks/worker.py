"""One torchrun worker per GPU; explicit E2E and full DiT boundaries."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import sys
import time
import traceback

from common import ROOT, SCHEMA, aggregate, read_json, schedule, sha256, write_json
from runtime import configure_caches, forward_layout, load_pipeline, reset_random, VAEObservation, video_metadata
from timing import TimedDiT


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--plan', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    plan = read_json(args.plan)
    config = plan['generation']
    rank = int(os.environ['RANK'])
    cache_paths = configure_caches(args.output, rank)  # Before torch / compiled imports.
    sys.path.insert(0, str(ROOT))
    import torch
    import torch.distributed as dist
    import numpy as np

    records, observer = [], None
    phase = 'load'

    def dump(name, data):
        if rank == 0:
            write_json(args.output / name, data)

    try:
        dump('progress.json', dict(status='loading'))
        started = time.perf_counter()
        pipe, backend = load_pipeline(plan, args.plan.parent)
        inner = pipe.dit
        layout = forward_layout(config)
        # e2e-only suites allocate no DiT events or wrapper.
        timer = TimedDiT(inner, torch, len(layout)) if 'dit' in plan['modes'] else None
        observer = VAEObservation(pipe.vae)
        torch.cuda.synchronize()
        dist.barrier()
        load_s = time.perf_counter() - started
        props = torch.cuda.get_device_properties(torch.cuda.current_device())
        devices = [None] * dist.get_world_size()
        dist.all_gather_object(devices, dict(rank=rank, name=props.name,
            uuid=str(getattr(props, 'uuid', 'unknown')), capability=[props.major, props.minor],
            total_memory_gib=props.total_memory / 2**30))
        packages = {name: importlib.metadata.version(name) for name in
            ['torch','torchvision','transformers','diffusers','numpy','av','triton','omegaconf','peft','flash-attn']}
        effective = dict(timing_schema_version=SCHEMA, generation=config, world_size=dist.get_world_size(),
            devices=devices, packages=packages, python=sys.version, cuda=torch.version.cuda,
            cudnn=torch.backends.cudnn.version(), resolved_attention=backend,
            actual_dit_forwards=len(layout), forward_layout=layout,
            load_seconds_excluded=load_s, cpu_threads=torch.get_num_threads(),
            rank0_cache_paths=cache_paths, offload=False, magcache=False, quantized_qwen=False,
            text_token_padding=False, fps=24, crf=5,
            parameter_storage='checkpoint mixed BF16/FP32' if dist.get_world_size()==1 else 'FSDP2 FP32 sharded storage, BF16 forward/reduction',
            environment={k: os.environ.get(k) for k in ['CUDA_VISIBLE_DEVICES','OMP_NUM_THREADS',
                'PYTORCH_CUDA_ALLOC_CONF','NCCL_ALGO','NCCL_PROTO']})
        dump('effective_config.json', effective)
        requests = {k: config[k] for k in ['time_length','width','height','seed','num_steps',
                    'guidance_weight','scheduler_scale','negative_caption','expand_prompts']}
        requests['text'] = config['prompt']
        expected_shape = [1, 3, 4*(config['time_length']*24//4)+1, config['height'], config['width']]

        for mode, warmup, repeat in schedule(plan['modes'], plan['warmup'], plan['repeats']):
            phase = 'warmup' if warmup else 'repeat'
            tag = f'{mode}_{phase}_{repeat}'
            video_path = args.output / f'{tag}.mp4'
            dump('progress.json', dict(status='running', tag=tag))
            observer.prepare(warmup)
            pipe.dit = timer if mode == 'dit' else inner
            if mode == 'dit':
                timer.count = 0
            reset_random(torch, np, config['seed'])
            torch.cuda.synchronize()
            dist.barrier()
            torch.cuda.reset_peak_memory_stats()
            response_s = None
            with torch.no_grad():
                # E2E START: native pipeline includes conversion, MP4 write and close.
                start = time.perf_counter()
                images = pipe(**requests, save_path=str(video_path))
                if rank == 0:
                    response_s = time.perf_counter() - start  # E2E END: MP4 saved.
                # No added pipeline-end CUDA sync or per-step sync in either mode.
                dist.barrier()
                distributed_s = time.perf_counter() - start

            # All checks, event queries and serialization below are OUTSIDE E2E.
            pipe.dit = inner
            torch.cuda.synchronize()
            steps = timer.elapsed_ms() if mode == 'dit' else None
            if warmup:
                if len(observer.validation) != 2 or not all(torch.isfinite(t).all().item() for t in observer.validation):
                    raise ValueError('Non-finite floating latent or VAE output in warmup')
                if list(observer.validation[1].shape) != expected_shape:
                    raise ValueError('Unexpected floating VAE output shape')
            local = dict(rank=rank, response_e2e_s=response_s, distributed_e2e_s=distributed_s,
                dit_step_ms=steps, vae_tiling=list(observer.tiles), floating_finite_checked=warmup,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30)
            per_rank = [None]*dist.get_world_size()
            dist.all_gather_object(per_rank, local)
            record = aggregate(per_rank, mode=mode, tag=tag, warmup=warmup, repeat=repeat)
            if rank == 0:
                if list(images.shape) != expected_shape or images.dtype != torch.uint8:
                    raise ValueError(f'Invalid output: {images.shape}, {images.dtype}')
                record.update(output_shape=list(images.shape), video_file=video_path.name)
                if plan['validate_video']:
                    metadata = video_metadata(video_path)
                    if (metadata['frames'] != expected_shape[2] or metadata['fps'] != 24
                            or metadata['width'] != config['width'] or metadata['height'] != config['height']):
                        raise ValueError(f'Invalid MP4: {metadata}')
                    record.update(video_metadata=metadata, video_sha256=sha256(video_path))
                records.append(record)
                dump('metrics.json', dict(timing_schema_version=SCHEMA, status='running', records=records))
                print(json.dumps({k: record[k] for k in ['tag','response_e2e_s','distributed_e2e_s']}), flush=True)
            observer.validation.clear()
            del images
        dump('metrics.json', dict(timing_schema_version=SCHEMA, status='complete', records=records))
        dump('progress.json', dict(status='complete'))
    except Exception as error:
        write_json(args.output / f'error_rank{rank}.json', dict(phase=phase,
            status='memory_insufficient' if isinstance(error, torch.cuda.OutOfMemoryError) else 'failed',
            traceback=traceback.format_exc()))
        raise
    finally:
        if observer is not None:
            observer.restore()
        if dist.is_initialized():
            dist.destroy_process_group()


if __name__ == '__main__':
    main()
