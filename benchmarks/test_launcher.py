"""Exercise the real coordinator lifecycle without starting GPU workers."""
from contextlib import ExitStack, redirect_stdout
import io
import signal
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import run as launcher
from common import HERE, ROOT, read_json, review_measurements, schedule, summarize, write_json


class LauncherTests(unittest.TestCase):
    def exercise(self, first_error=None):
        with tempfile.TemporaryDirectory() as folder, ExitStack() as patches:
            folder=Path(folder); suite=folder/'suite'; model=folder/'model'; model.mkdir()
            for name in ['vae','scheduler','text_encoder','vision_encoder','tokenizer','dit','lora','qwen','clip']:
                (model/name).mkdir()
            (model/'dit.safetensors').touch()
            (model/'lora/refinement_lora.safetensors').touch()
            for name in ['vae','qwen','clip']:
                write_json(model/name/'config.json',dict(scale_factor_spatial=8))
            trans=model/'transformer/720p_i2v_distilled_sparse';trans.mkdir(parents=True)
            write_json(trans/'config.json',{})
            image=folder/'image.png';image.write_bytes(b'test')
            frames=folder/'frames.npy';frames.write_bytes(b'test')
            args=['run.py','--gpus','2','1','--output',str(suite)]
            if ROOT.name=='HunyuanVideo-1.5':
                args+=['--model-path',str(model),'--image-path',str(image)]
            elif ROOT.name=='LongCat-Video':
                args+=['--model-path',str(model),'--input-frames',str(frames)]
                patches.enter_context(patch.object(launcher,'load_frames',return_value=SimpleNamespace(shape=[93,640,640,3])))
                patches.enter_context(patch.object(launcher,'input_sha256',return_value='input-hash'))
            else:
                # Load YAML dependencies before patch.dict restores sys.modules;
                # removing lazy imports leaves cached YAML classes inconsistent.
                from omegaconf import OmegaConf
                OmegaConf.load(ROOT/'configs/k5_pro_t2v_10s_sft_sd.yaml')
                for name,path in [('dit',model/'dit.safetensors'),('qwen',model/'qwen'),('clip',model/'clip'),('vae',model)]:
                    args += [f'--{name}-path',str(path)]
            modules={
                'hyvideo.commons':SimpleNamespace(PIPELINE_CONFIGS={'720p_i2v_distilled_sparse':{}}),
                'longcat_video.context_parallel.context_parallel_util':SimpleNamespace(get_optimal_split=lambda n:[1,n]),
                'kandinsky.t2v_pipeline':SimpleNamespace(Kandinsky5T2VPipeline=SimpleNamespace(RESOLUTIONS={512:[(512,768)]})),
            }
            patches.enter_context(patch.dict(sys.modules,modules))
            patches.enter_context(patch.object(sys,'argv',args))
            patches.enter_context(patch.object(launcher,'inventory',return_value=[dict(index=str(i),uuid=f'GPU-{i}') for i in range(2)]))
            patches.enter_context(patch.dict(launcher.os.environ,{'CUDA_VISIBLE_DEVICES':'1,0'}))
            patches.enter_context(patch.object(launcher,'model_manifest',return_value=dict(configs={})))
            patches.enter_context(patch.object(launcher.subprocess,'check_output',side_effect=['commit\n','']))
            patches.enter_context(patch.object(launcher,'wait_for_suite'))
            signals = patches.enter_context(patch.object(launcher.signal, 'signal'))
            handles=[]; launches=[]
            def idle(selected,notify):
                notify([dict(reason='test busy GPU')])
                state=read_json(suite/'status.json')
                self.assertEqual(state['status'],'waiting')
                self.assertTrue(state['waiting_reason'])
                handle=(folder/f'lock{len(handles)}').open('w')
                handles.append(handle)
                return [handle]
            patches.enter_context(patch.object(launcher,'wait_for_idle',side_effect=idle))
            def popen(command,**kwargs):
                self.assertFalse(handles[-1].closed)
                n=2 if not launches else 1
                launches.append(n)
                self.assertEqual(kwargs['env']['CUDA_VISIBLE_DEVICES'],','.join(['GPU-1','GPU-0'][:n]))
                self.assertEqual(read_json(suite/'status.json')['status'],'running')
                output=Path(command[command.index('--output')+1])
                if first_error == 'interrupt' and len(launches) == 1:
                    def interrupted_wait(**kwargs):
                        signals.call_args.args[1](signal.SIGTERM, None)
                    return SimpleNamespace(pid=123456789, wait=interrupted_wait)
                if first_error and len(launches)==1:
                    write_json(output/'error_rank0.json',dict(status=first_error,phase='load',traceback='RuntimeError: test failure'))
                    return SimpleNamespace(pid=123456789, wait=lambda **kwargs:1)
                records=[]
                for mode,warmup,repeat in schedule(['dit'],1,1):
                    records.append(dict(mode=mode,warmup=warmup,repeat=repeat,output_shape=[1,3,1,8,8],
                        response_e2e_s=10,distributed_e2e_s=11,dit_forward_sum_max_rank_s=5,dit_forward_step_max_sum_s=5))
                write_json(output/'metrics.json',dict(status='complete',records=records))
                return SimpleNamespace(pid=123456789, wait=lambda **kwargs:0)
            patches.enter_context(patch.object(launcher.subprocess,'Popen',side_effect=popen))
            with redirect_stdout(io.StringIO()):
                if first_error == 'interrupt':
                    with self.assertRaises(KeyboardInterrupt):
                        launcher.main()
                elif first_error=='failed':
                    with self.assertRaisesRegex(RuntimeError,'Group failed'):
                        launcher.main()
                elif first_error:
                    with self.assertRaises(SystemExit) as error:
                        launcher.main()
                    self.assertEqual(error.exception.code,1)
                else:
                    launcher.main()
            self.assertTrue(all(handle.closed for handle in handles))
            self.assertTrue((suite/'scripts/gpu_wait.py').is_file())
            self.assertTrue((suite/'scripts/state.py').is_file())
            self.assertFalse((suite/'historical_baseline.json').exists())
            state=read_json(suite/'status.json')
            self.assertEqual(state['status'],{'interrupt':'interrupted_or_failed','failed':'interrupted_or_failed','memory_insufficient':'complete_with_failures'}.get(first_error,'complete'))
            self.assertEqual(state['has_failures'],bool(first_error))
            self.assertEqual(launches,[2] if first_error in {'failed', 'interrupt'} else [2,1])
            self.assertEqual(read_json(suite/'plan.json')['selected_gpu_uuids'],['GPU-1','GPU-0'])

    def test_sigterm_updates_status_and_releases_locks(self):
        self.exercise('interrupt')

    def test_success(self):
        self.exercise()

    def test_worker_failure_stops_matrix(self):
        self.exercise('failed')

    def test_loading_oom_is_recorded_and_next_group_runs(self):
        self.exercise('memory_insufficient')

    def test_review_does_not_invent_single_sample_variation(self):
        row=dict(gpus=1,mode='dit',**summarize([dict(response_e2e_s=10)]))
        self.assertEqual(review_measurements([row]),[])
        row.update(summarize([dict(response_e2e_s=x) for x in [12,11,10]]))
        findings=review_measurements([row])
        self.assertEqual(len(findings),1)
        self.assertIn('波动较大',findings[0])
        self.assertIn('建议增加预热',findings[0])


if __name__=='__main__':
    unittest.main()
