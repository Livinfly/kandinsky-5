"""CPU checks for timing scope, scheduling, reports, portability and GPU waits."""
from contextlib import nullcontext, redirect_stdout
import copy
import io
import os
from pathlib import Path
import shlex
import shutil
import subprocess
import sys
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, patch

from common import HERE, ROOT, aggregate, measurements_complete, read_json, schedule, summarize, write_json
from gpu_wait import busy_reasons, select_devices, wait_for_idle
from report import collect, generate_report
from runtime import forward_layout
from timing import TimedDiT
import worker


def record(mode='dit', warmup=False, repeat=1):
    ranks = [dict(rank=i,response_e2e_s=10 if i==0 else None,distributed_e2e_s=11+i,
                  dit_step_ms=steps if mode=='dit' else None,peak_allocated_gib=1,peak_reserved_gib=2)
             for i,steps in enumerate([[1000,3000],[2000,1000]])]
    return aggregate(ranks,mode=mode,tag='test',warmup=warmup,repeat=repeat)


class BenchmarkTests(unittest.TestCase):
    def test_cfg_count_and_labels(self):
        config = read_json(HERE/'baseline.json')
        layout = forward_layout(config)
        self.assertEqual(len(layout),100)
        self.assertEqual(layout[1],dict(forward_index=1,scheduler_step_index=0,branch='unconditional'))
        self.assertEqual(layout[-1]['scheduler_step_index'],49)
        for cfg,count in [(1.0,50),(1.0000001,50),(1.01,100)]:
            config['guidance_weight']=cfg
            self.assertEqual(len(forward_layout(config)),count)

    def test_events_include_inner_pre_post_hooks(self):
        import torch
        trace=[]
        class Event:
            def __init__(self,**kwargs): pass
            def record(self): trace.append('event')
            def elapsed_time(self,other): return 10.
        fake = SimpleNamespace(cuda=SimpleNamespace(Event=Event,synchronize=lambda:trace.append('sync')))
        class Model(torch.nn.Module):
            def forward(self,x): trace.append('forward'); return x
        inner=Model()
        inner.register_forward_pre_hook(lambda *a:trace.append('fsdp_pre'))
        inner.register_forward_hook(lambda *a:trace.append('fsdp_post'))
        timer=TimedDiT(inner,fake,1)
        self.assertEqual(trace.count('sync'),1)
        trace.clear(); timer(torch.zeros(1))
        self.assertEqual(trace,['event','fsdp_pre','forward','fsdp_post','event'])
        self.assertEqual(timer.elapsed_ms(),[10.])
        trace.clear(); inner(torch.zeros(1))
        self.assertEqual(trace,['fsdp_pre','forward','fsdp_post'])

    def test_rank_aggregation_and_completion(self):
        r=record()
        self.assertEqual(r['distributed_e2e_s'],12)
        self.assertEqual(r['dit_forward_sum_max_rank_s'],4)
        self.assertEqual(r['dit_forward_step_max_sum_s'],5)
        self.assertEqual(r['per_rank'][1]['dit_step_ms'],[2000,1000])
        self.assertIsNone(summarize([r])['response_e2e_s']['stdev'])
        self.assertNotIn('dit_forward_sum_max_rank_s',record('e2e'))
        plan=dict(modes=['dit','e2e'],warmup=1,repeats=1)
        records=[record(mode,warmup,repeat) for mode,warmup,repeat in schedule(plan['modes'],1,1)]
        self.assertTrue(measurements_complete(dict(status='complete',records=records),plan))
        for bad in [records[:-1],records+[records[0]]]:
            self.assertFalse(measurements_complete(dict(status='complete',records=bad),plan))

    def test_gpu_selection_and_occupied_zero_utilization(self):
        devices=[dict(index=str(i),uuid=f'GPU-{i}',memory_mib=0,utilization=0) for i in range(3)]
        self.assertEqual(select_devices(devices,'2,GPU-0'),['GPU-2','GPU-0'])
        for mask in ['', '0,0','GPU-', '5']:
            with self.assertRaises(ValueError): select_devices(devices,mask)
        self.assertEqual(busy_reasons(devices,[],['GPU-0']),[])
        self.assertTrue(busy_reasons(devices,[['GPU-0','123']],['GPU-0']))
        devices[0]['memory_mib']=257
        self.assertTrue(busy_reasons(devices,[],['GPU-0']))
        devices[0].update(memory_mib=0,utilization=1)
        self.assertTrue(busy_reasons(devices,[],['GPU-0']))

    def test_gpu_wait_and_lock_exclusion(self):
        idle=[dict(index='0',uuid='GPU-test',memory_mib=0,utilization=0)]
        busy=copy.deepcopy(idle);busy[0]['utilization']=100
        with tempfile.TemporaryDirectory() as folder, patch('gpu_wait.LOCK_ROOT',Path(folder)), \
             patch('gpu_wait.inventory',side_effect=[busy,idle,idle]), patch('gpu_wait.query',return_value=[]), \
             patch('gpu_wait.time.sleep'):
            notices=[]
            handles=wait_for_idle(['GPU-test'],notices.append,interval=1)
            self.assertEqual(len(notices),2)
            def abort(reasons):
                self.assertEqual(reasons[0]['reason'],'another benchmark holds its lock')
                raise RuntimeError('stop waiting')
            try:
                with self.assertRaisesRegex(RuntimeError,'stop waiting'):
                    wait_for_idle(['GPU-test'],abort,interval=1)
            finally:
                for h in handles:h.close()

    def test_report_complete_speedups(self):
        with tempfile.TemporaryDirectory() as folder:
            suite=Path(folder)
            plan=dict(gpu_counts=[1,2],modes=['dit'],warmup=1,repeats=1,offload=['off'],
                source_commit='unknown',inference_source_modified=False,
                generation=read_json(HERE/'baseline.json'),model_config=dict(model=dict(dit_params={},attention={}),metrics={}),
                model_paths={},validate_video=False)
            write_json(suite/'plan.json',plan)
            write_json(suite/'status.json',dict(status='running',completed=[],total=2))
            for n in [1,2]:
                group=suite/f'offload_off/g{n}';group.mkdir(parents=True)
                write_json(group/'metrics.json',dict(status='running' if n==1 else 'complete',records=[record(warmup=True),record()]))
            self.assertNotIn('e2e_speedup',collect(suite,plan)[1])
            group=suite/'offload_off/g1'
            write_json(group/'metrics.json',dict(status='complete',records=[record(warmup=True),record()]))
            write_json(suite/'status.json', dict(status='running', total=2, completed=[
                dict(directory=f'offload_off/g{n}', status='complete') for n in [1, 2]]))
            self.assertEqual(collect(suite,plan)[1]['e2e_speedup'],1)
            generate_report(suite)
            self.assertIn('单次',(suite/'report.md').read_text())
            self.assertNotIn('## 历史对比',(suite/'report.md').read_text())
            self.assertTrue(all('historical_mismatches' not in r for r in read_json(suite/'summary.json')['results']))

    def test_relocated_dry_run(self):
        with tempfile.TemporaryDirectory(prefix='k5 space ') as folder:
            checkout=Path(folder)/'clone'
            shutil.copytree(HERE,checkout/'benchmarks',ignore=shutil.ignore_patterns('__pycache__'))
            command=[sys.executable,str(checkout/'benchmarks/run.py'),'--gpus','8','1','--dry-run']
            for k in ['dit','qwen','clip','vae']:command += [f'--{k}-path','relative path']
            result=subprocess.run(command,cwd=folder,text=True,capture_output=True)
            self.assertEqual(result.returncode,0,result.stderr)
            self.assertIn(str(checkout),result.stdout)
            # The caller's interpreter can live in the original checkout; all data/script
            # paths must resolve under the relocated clone, independently of that venv.
            for line in result.stdout.splitlines():
                self.assertNotIn(str(ROOT),' '.join(shlex.split(line)[1:]))
            self.assertFalse((checkout/'outputs').exists())


class WorkerTests(unittest.TestCase):
    def test_native_save_and_no_intermediate_sync_in_both_modes(self):
        import torch as real_torch
        for modes in [['e2e'],['dit','e2e']]:
            with self.subTest(modes=modes), tempfile.TemporaryDirectory() as folder:
                output=Path(folder);config=read_json(HERE/'baseline.json')
                config.update(time_length=1,height=8,width=8,num_steps=1)
                plan=dict(generation=config,modes=modes,warmup=1,repeats=1,validate_video=False)
                write_json(output/'plan.json',plan)
                trace=[]
                torch,dist,pipe,observer,timer=[MagicMock() for _ in range(5)]
                torch.distributed=dist
                torch.cuda.get_device_properties.return_value=SimpleNamespace(name='test',major=9,minor=0,total_memory=2**30)
                torch.cuda.max_memory_allocated.return_value=1
                torch.cuda.max_memory_reserved.return_value=2
                torch.cuda.OutOfMemoryError=MemoryError
                torch.cuda.synchronize.side_effect=lambda:trace.append('sync')
                torch.no_grad.side_effect=nullcontext
                torch.version.cuda='test';torch.backends.cudnn.version.return_value=1
                torch.get_num_threads.return_value=8
                torch.uint8=real_torch.uint8
                dist.get_world_size.return_value=1
                dist.barrier.side_effect=lambda:trace.append('barrier')
                dist.all_gather_object.side_effect=lambda dest,value:dest.__setitem__(0,value)
                observer.tiles=[]
                observer.validation=[real_torch.zeros(1),real_torch.zeros(1,3,25,8,8)]
                def prepare(warmup):observer.validation=[real_torch.zeros(1),real_torch.zeros(1,3,25,8,8)]
                observer.prepare.side_effect=prepare
                def infer(**kwargs):
                    self.assertTrue(kwargs['save_path'].endswith('.mp4'))
                    trace.append('pipeline_with_save')
                    return real_torch.zeros(1,3,25,8,8,dtype=real_torch.uint8)
                pipe.side_effect=infer
                timer.elapsed_ms.return_value=[100.,200.]
                argv=['worker.py','--plan',str(output/'plan.json'),'--output',str(output)]
                with patch.dict(sys.modules,{'torch':torch,'torch.distributed':dist}), patch.dict(os.environ,{'RANK':'0'}), \
                     patch.object(sys,'argv',argv), patch.object(worker,'load_pipeline',return_value=(pipe,{})), \
                     patch.object(worker,'VAEObservation',return_value=observer), \
                     patch.object(worker,'TimedDiT',return_value=timer) as events, \
                     patch.object(worker,'reset_random') as reset, \
                     patch.object(worker.time,'perf_counter',side_effect=lambda:trace.append('clock') or float(len(trace))), \
                     redirect_stdout(io.StringIO()):
                    worker.main()
                self.assertEqual(events.call_count,int('dit' in modes))
                self.assertEqual(reset.call_count,2*len(modes))
                for i,value in enumerate(trace):
                    if value=='pipeline_with_save':
                        self.assertEqual(trace[i-1:i+4],['clock','pipeline_with_save','clock','barrier','clock'])
                metrics=read_json(output/'metrics.json')
                self.assertEqual(metrics['status'],'complete')
                for r in metrics['records']:
                    self.assertEqual(r['per_rank'][0]['dit_step_ms'],[100.,200.] if r['mode']=='dit' else None)


if __name__=='__main__':
    unittest.main()
