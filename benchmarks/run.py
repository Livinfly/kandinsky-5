"""Portable single-node benchmark matrix; waits for selected GPUs to become idle."""
import argparse
from contextlib import ExitStack
import datetime
import math
import os
from pathlib import Path
import shlex
import shutil
import signal
import subprocess
import sys

from lifecycle import stop_worker
from state import write_status
from common import HERE, ROOT, SCHEMA, measurements_complete, read_json, sha256, write_json
from gpu_wait import inventory, select_devices, wait_for_idle, wait_for_suite


def portable_model_config(config):
    """Exclude machine-local checkpoint paths from model configuration comparisons."""
    import copy
    config = copy.deepcopy(config)
    model = config['model']
    model['checkpoint_path'] = '<dit>'
    model['vae']['checkpoint_path'] = '<vae-root>'
    for key in ['qwen', 'clip']:
        model['text_embedder'][key]['checkpoint_path'] = f'<{key}>'
    return config


def model_manifest(paths):
    configs, weights = {}, {}
    for component, root in paths.items():
        files = [root] if root.is_file() else root.rglob('*')
        for path in files:
            if not path.is_file() or '.cache' in path.parts:
                continue
            name = component + '/' + (path.name if root.is_file() else str(path.relative_to(root)))
            if path.name in {'config.json', 'preprocessor_config.json', 'tokenizer_config.json'}:
                configs[name] = dict(sha256=sha256(path), config=read_json(path))
            elif path.suffix in {'.safetensors','.bin','.pt'}:
                weights[name] = path.stat().st_size
    return dict(configs=configs, weight_file_sizes=weights,
                note='Configuration hashes and weight sizes; weight payloads are not hashed.')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, default=HERE/'baseline.json')
    parser.add_argument('--model-config', type=Path, default=ROOT/'configs/k5_pro_t2v_10s_sft_sd.yaml')
    for key, help_text in [('dit','DiT safetensors file'),('qwen','Qwen model directory'),
                           ('clip','CLIP model directory'),('vae','Root containing vae/ subfolder')]:
        parser.add_argument(f'--{key}-path', type=Path, required=True, help=help_text)
    prompt = parser.add_mutually_exclusive_group()
    prompt.add_argument('--prompt')
    prompt.add_argument('--prompt-file', type=Path)
    parser.add_argument('--gpus', type=int, nargs='+', default=[1])
    parser.add_argument('--modes', choices=['dit','e2e'], nargs='+', default=['dit'])
    parser.add_argument('--warmup', type=int, default=1)
    parser.add_argument('--repeats', type=int, default=1)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--background', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    parser.add_argument('--validate-video', action='store_true')
    parser.add_argument('--baseline', type=Path, help='Opt in to historical comparison')
    parser.add_argument('--wait-for', type=Path, help='Also wait for another suite status.json')
    parser.add_argument('--sparse-attention', action='store_true',
                        help='Scope v2: attention input communication through result return/aggregation and output readiness; see sparse_attention.py')
    args = parser.parse_args()
    if args.sparse_attention:
        os.environ['VIDEO_BENCH_SPARSE_ATTENTION'] = '1'
        if 'dit' not in args.modes:
            parser.error('--sparse-attention requires dit mode')
    if args.warmup < 1 or args.repeats < 1 or min(args.gpus) < 1:
        parser.error('Require positive GPU counts, >=1 warmup and >=1 measured request')
    if len(set(args.gpus)) != len(args.gpus) or len(set(args.modes)) != len(args.modes):
        parser.error('Duplicate GPU counts/modes are not allowed')
    config = read_json(args.config)
    if config.keys() != read_json(HERE/'baseline.json').keys():
        parser.error('Use the keys in baseline.json')
    if any(type(config[k]) is not int or config[k] < 1 for k in ['width','height','num_steps','time_length']):
        parser.error('Dimensions, duration and steps must be positive integers')
    if (type(config['seed']) is not int or not 0 <= config['seed'] < 2**32
            or type(config['expand_prompts']) is not bool or config['scheduler_scale'] <= 0):
        parser.error('Invalid seed, expand_prompts or scheduler_scale')
    if not all(math.isfinite(config[k]) for k in ['guidance_weight','scheduler_scale']):
        parser.error('guidance_weight and scheduler_scale must be finite')
    if args.prompt_file:
        config['prompt'] = args.prompt_file.read_text().strip()
    elif args.prompt is not None:
        config['prompt'] = args.prompt
    stamp = datetime.datetime.now(datetime.timezone.utc).strftime('%Y%m%dT%H%M%S.%fZ')
    suite = args.output.resolve() if args.output else ROOT/'outputs/benchmarks'/stamp
    commands = [dict(gpus=n, offload='off', directory=f'offload_off/g{n}', command=[sys.executable,
        '-m','torch.distributed.run','--standalone','--nnodes=1',f'--nproc_per_node={n}',
        str(suite/'scripts/worker.py'),'--plan',str(suite/'plan.json'),'--output',str(suite/f'offload_off/g{n}')])
        for n in args.gpus]
    if args.dry_run:
        for entry in commands:
            print(shlex.join(entry['command']))
        return
    from omegaconf import OmegaConf
    model = OmegaConf.to_container(OmegaConf.load(args.model_config), resolve=True)
    paths = {k: getattr(args,k+'_path').resolve() for k in ['dit','qwen','clip','vae']}
    for path in [paths['dit'], paths['qwen']/'config.json', paths['clip']/'config.json', paths['vae']/'vae/config.json']:
        if not path.is_file():
            parser.error(f'Missing model component: {path}')
    model['model']['checkpoint_path'] = str(paths['dit'])
    model['model']['vae']['checkpoint_path'] = str(paths['vae'])
    for k in ['qwen','clip']:
        model['model']['text_embedder'][k]['checkpoint_path'] = str(paths[k])
    params = model['model']['dit_params']
    heads = params['model_dim'] // sum(params['axes_dims'])
    if any(heads % n for n in args.gpus):
        parser.error(f'GPU counts must divide this model\'s {heads} attention heads')
    # Import official size validation without loading any weights or CUDA contexts.
    sys.path.insert(0,str(ROOT))
    from kandinsky.t2v_pipeline import Kandinsky5T2VPipeline
    if (config['height'],config['width']) not in Kandinsky5T2VPipeline.RESOLUTIONS[model['metrics']['resolution']]:
        parser.error('Width/height must match the official resolution bucket in the model YAML')
    if config['attention_engine'] not in ['auto','sdpa','flash_attention_2','flash_attention_3','sage']:
        parser.error('Unknown attention_engine')
    devices = select_devices(inventory(), os.environ.get('CUDA_VISIBLE_DEVICES'))
    if max(args.gpus) > len(devices):
        parser.error(f'Requested {max(args.gpus)} GPUs, only {len(devices)} selected')
    if args.baseline:
        read_json(args.baseline)  # Fail before detaching if an explicit reference is missing.
    if suite.exists():
        parser.error(f'Refusing to overwrite {suite}')
    if args.background:
        suite.parent.mkdir(parents=True,exist_ok=True)
        log_path = suite.parent/f'{suite.name}.launcher.log'
        forwarded = [x for x in sys.argv[1:] if x != '--background']
        if args.output is None:
            forwarded += ['--output',str(suite)]
        with log_path.open('x') as log:
            child = subprocess.Popen([sys.executable,str(HERE/'run.py'),*forwarded],stdout=log,
                stderr=subprocess.STDOUT,stdin=subprocess.DEVNULL,start_new_session=True)
        write_json(suite.parent/f'{suite.name}.job.json',dict(pid=child.pid,suite=str(suite),log=str(log_path)))
        print(f'PID: {child.pid}\nSuite: {suite}\nLog: {log_path}')
        return

    def interrupted(signum, frame):
        raise KeyboardInterrupt(f"Received signal {signum}")

    signal.signal(signal.SIGTERM, interrupted)
    suite.mkdir(parents=True)
    scripts = suite/'scripts'; scripts.mkdir()
    for path in HERE.iterdir():
        if path.is_file() and path.suffix in {'.py','.json','.sh','.lock','.md'}:
            shutil.copy2(path,scripts/path.name)
    try:
        commit = subprocess.check_output(['git','-C',str(ROOT),'rev-parse','HEAD'],text=True).strip()
        diff = subprocess.check_output(['git','-C',str(ROOT),'diff','HEAD','--','kandinsky'],text=True)
    except (FileNotFoundError,subprocess.CalledProcessError):
        commit,diff = 'unknown','git provenance unavailable'
    (suite/'inference_source.diff').write_text(diff)
    OmegaConf.save(OmegaConf.create(model),suite/'model.yaml')
    write_json(suite/'model_manifest.json',model_manifest(paths))
    baseline = suite/'historical_baseline.json' if args.baseline else None
    if baseline is not None:
        shutil.copy2(args.baseline,baseline)
    plan = dict(timing_schema_version=SCHEMA,created_utc=stamp,source_commit=commit,
        inference_source_modified=bool(diff),generation=config,model_config=portable_model_config(model),
        model_paths={k:str(v) for k,v in paths.items()},gpu_counts=args.gpus,selected_gpu_uuids=devices,
        offload=['off'],modes=args.modes,warmup=args.warmup,repeats=args.repeats,
        validate_video=args.validate_video,wait_for=str(args.wait_for.resolve()) if args.wait_for else None,
        pid=os.getpid())
    plan['sparse_attention_timing'] = os.environ.get('VIDEO_BENCH_SPARSE_ATTENTION') == '1'
    from sparse_attention import SCOPE, SCOPE_VERSION
    plan['sparse_attention_scope_version'] = SCOPE_VERSION if plan['sparse_attention_timing'] else None
    plan['sparse_attention_range'] = SCOPE if plan['sparse_attention_timing'] else None
    write_json(suite/'plan.json',plan)
    status = dict(status='waiting',pid=os.getpid(),total=len(commands),completed=[],current=None)
    from report import generate_report

    def waiting(reasons):
        status.update(status='waiting',waiting_reason=reasons)
        write_status(suite, status)
        print('Waiting: '+str(reasons),flush=True)

    write_status(suite, status)
    try:
        generate_report(suite,baseline=baseline)
        wait_for_suite(args.wait_for, waiting)
        for entry in commands:
            directory = suite/entry['directory']; directory.mkdir(parents=True)
            status['current'] = {k:entry[k] for k in ['gpus','offload','directory']}
            selected = devices[:entry['gpus']]
            with ExitStack() as locks:
                for handle in wait_for_idle(selected,waiting):
                    locks.enter_context(handle)
                status.update(status='running',waiting_reason=[])
                write_status(suite, status)
                env = {**os.environ,'CUDA_VISIBLE_DEVICES':','.join(selected),'KANDINSKY_SOURCE_ROOT':str(ROOT),
                    'OMP_NUM_THREADS':'8','HF_HUB_OFFLINE':'1','TRANSFORMERS_OFFLINE':'1',
                    'TOKENIZERS_PARALLELISM':'false','PYTORCH_CUDA_ALLOC_CONF':'expandable_segments:True'}
                (directory/'command.sh').write_text(shlex.join(entry['command'])+'\n')
                with (directory/'inference.log').open('w') as log:
                    child = subprocess.Popen(entry['command'],cwd=ROOT,env=env,stdout=log,stderr=subprocess.STDOUT,start_new_session=True)
                    try:
                        code = child.wait()
                    finally:
                        stop_worker(child)
            result = dict(gpus=entry['gpus'],offload='off',directory=entry['directory'],exit_code=code,
                          status='complete' if code==0 else 'failed')
            if code==0 and not measurements_complete(read_json(directory/'metrics.json'),plan):
                result['status']='failed'
            if code!=0:
                errors = [read_json(p) for p in directory.glob('error_rank*.json')]
                if errors and all(e['status']=='memory_insufficient' and e['phase'] in {'load','warmup'} for e in errors):
                    result['status']='memory_insufficient'
            status['completed'].append(result)
            write_status(suite, status)
            generate_report(suite,baseline=baseline)
            if result['status']=='failed':
                raise RuntimeError(f'Group failed; inspect {directory}/inference.log')
        status.update(status='complete' if all(r['status']=='complete' for r in status['completed']) else 'complete_with_failures',current=None)
    except BaseException as error:
        status['status']='interrupted_or_failed'
        write_status(suite, status, error=error)
        raise
    finally:
        write_status(suite, status)
        generate_report(suite,baseline=baseline)
    print(f'Report: {suite / "report.md"}',flush=True)
    if status['status']!='complete':
        raise SystemExit(1)


if __name__=='__main__':
    main()
