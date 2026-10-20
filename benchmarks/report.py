"""CPU-only reports; historical comparison is explicitly opt-in."""
import argparse
import json
from pathlib import Path

from lifecycle import completed_groups
from common import review_measurements, FIELDS, measurements_complete, read_json, summarize, write_json


def collect(suite, plan):
    completed = completed_groups(suite)
    rows = []
    for gpus in plan['gpu_counts']:
        directory = suite / f'offload_off/g{gpus}'
        def optional(name):
            path = directory / name
            return read_json(path) if path.exists() else {}
        metrics, effective = optional('metrics.json'), optional('effective_config.json')
        for mode in plan['modes']:
            formal = [r for r in metrics.get('records', []) if r['mode'] == mode and not r['warmup']]
            warmup = [r for r in metrics.get('records', []) if r['mode'] == mode and r['warmup']]
            rows.append(dict(gpus=gpus, mode=mode, count=len(formal), effective=effective,
                             complete=str(directory.relative_to(suite)) in completed and measurements_complete(metrics, plan, mode),
                             warmup=summarize(warmup), **summarize(formal)))
    for row in rows:
        base = next((r for r in rows if r['gpus'] == 1 and r['mode'] == row['mode'] and r['complete']), None)
        for field, key in [(FIELDS[0], 'e2e_speedup'), (FIELDS[2], 'dit_speedup')]:
            if base and row['complete'] and field in row and field in base:
                row[key] = base[field]['mean'] / row[field]['mean']
    return rows


def fmt(row, field):
    if field not in row:
        return '—'
    s = row[field]
    return f"{s['mean']:.3f}（单次）" if s['stdev'] is None else f"{s['mean']:.3f} ± {s['stdev']:.3f}"


def mismatches(plan, row, historical):
    reasons = [k for k in ['source_commit','generation','model_config'] if plan[k] != historical[k]]
    if plan['inference_source_modified']:
        reasons.append('inference source modified')
    if not plan.get('timing_schema_version') or plan.get('timing_schema_version') != historical.get('timing_schema_version'):
        reasons.append('timing_schema_version')
    effective = row['effective']
    if not effective:
        return reasons + ['effective config pending']
    if any(d['name'] != historical['gpu_name'] for d in effective['devices']):
        reasons.append('GPU model')
    if effective['resolved_attention'] != historical['resolved_attention']:
        reasons.append('resolved attention')
    for key, value in historical['packages'].items():
        if effective['packages'].get(key) != value:
            reasons.append('package.'+key)
    return reasons


def generate_report(suite, baseline=None):
    suite = Path(suite)
    plan, status = [read_json(suite/name) for name in ['plan.json','status.json']]
    historical = read_json(baseline) if baseline is not None else None
    rows = collect(suite,plan)
    model = plan['model_config']
    lines = ['# Kandinsky-5 T2V 性能报告','',
        f"状态：**{status['status']}**；完成组 {len(status['completed'])}/{status['total']}。",'',
        'E2E：rank 0 从调用原生 pipeline 前至调用返回（MP4 已保存）；包含初始噪声、Qwen/CLIP、DiT、CFG、scheduler、通信、VAE、量化、CPU 转换和视频编码写盘。加载、排队、完整 warmup、验证和指标聚合排除在正式时间外。','',
        'DiT：在完整 DiT __call__ 外记录预分配 CUDA events，包括 FSDP/SP 前后 hooks 与 NABLA；各 rank 先求和再取最大值 max(sum)。排除 CFG 合并、scheduler、初始 STA mask、最终 latent all-gather、VAE，不代表纯 kernel 时间。','',
        'dit = E2E + DiT；e2e 请求不包裹 DiT、不记录 DiT events。两者均不增加逐步同步或 pipeline 返回后的 CUDA 同步。distributed E2E 额外计到保存后 barrier 返回，保留每 rank 原始值并取最大值。','',
        '## 模型与推理配置','',
        f"- 源码 commit：`{plan['source_commit']}`；推理源码改动：{plan['inference_source_modified']}。",
        f"- 本次模型路径：`{plan['model_paths']}`。",
        f"- DiT 参数：`{model['model']['dit_params']}`。",
        f"- Attention / NABLA：`{model['model']['attention']}`；metrics / RoPE：`{model['metrics']}`。",
        '- offload、MagCache、Qwen NF4、text token padding 关闭；原生内置 compile 保留。',
        '- 单卡保留 checkpoint BF16/FP32 混合存储；多卡为官方 FSDP2 FP32 分片存储，BF16 forward/reduction + DTensor 序列/heads 重分布。Qwen BF16、CLIP FP32；VAE FP16 权重、BF16 decode autocast。',
        '- 每 rank 完整 VAE decode；自动选择的 tile/stride 原样记录在 per_rank，未人为固定；两模式共用相同轻量记录器，无中间计时。',
        '- MP4：24 FPS、CRF=5，原生 torchvision writer；每请求在计时前重置 Python/NumPy/torch/CUDA RNG，原生噪声生成仍在计时内。',
        f"- 每模式 warmup={plan['warmup']}、正式={plan['repeats']}；额外 MP4 解码/hash 验证：{plan['validate_video']}。",
        '- 浮点 latent/VAE 有限值在 warmup 后检查；正式请求保留原生 tensor 生命周期，只检查输出形状/uint8。',
        '- 完整有效 YAML 见 [model.yaml](model.yaml)，模型配置 hash 与权重文件大小见 [model_manifest.json](model_manifest.json)，未逐字节核验权重。','',
        '```json',json.dumps(plan['generation'],ensure_ascii=False,indent=2),'```','']
    for row in rows:
        env = row['effective']
        if not env or row['mode'] != plan['modes'][0]:
            continue
        devices = ', '.join(sorted({f"{d['name']} / {d['total_memory_gib']:.2f} GiB" for d in env['devices']}))
        lines += [f"g{row['gpus']}：{devices}；实际 DiT forwards={env['actual_dit_forwards']}；非 NABLA attention=`{env['resolved_attention']}`。",
                  f"Python {env['python'].split()[0]}，CUDA {env['cuda']}，OMP threads={env['cpu_threads']}，包 `{env['packages']}`。",'']
    lines += ['## 耗时与多卡加速比','',
        '单位秒，多次为均值 ± 样本标准差；一次标记“单次”，JSON stdev=null。加速比使用同模式、已完成的单卡结果；单卡未完成时留空。','',
        '| GPU | 模式 | 次数 / 状态 | E2E | distributed E2E | DiT max(sum) | E2E 加速比 | DiT 加速比 |',
        '|---:|---|---|---:|---:|---:|---:|---:|']
    for row in rows:
        ratios = [f"{row[k]:.3f}×" if k in row else '—' for k in ['e2e_speedup','dit_speedup']]
        lines.append(f"| {row['gpus']} | {row['mode']} | {row['count']} / {'完成' if row['complete'] else '待完成'} | "+
                     ' | '.join([fmt(row,f) for f in FIELDS[:3]]+ratios)+' |')
    lines += ['','## Warmup 与模式对照','',
        '| GPU | 模式 | warmup E2E / DiT | 正式 E2E / DiT |','|---:|---|---|---|']
    comparisons = []
    for row in rows:
        lines.append(f"| {row['gpus']} | {row['mode']} | {fmt(row['warmup'],FIELDS[0])} / {fmt(row['warmup'],FIELDS[2])} | {fmt(row,FIELDS[0])} / {fmt(row,FIELDS[2])} |")
        other = next((r for r in rows if r['gpus']==row['gpus'] and r['mode']=='e2e' and r['complete']),None)
        if row['mode']=='dit' and row['complete'] and other:
            comparisons.append(dict(gpus=row['gpus'],mean_difference_pct=100*(row[FIELDS[0]]['mean']/other[FIELDS[0]]['mean']-1),
                paired_difference_pct=[100*(a/b-1) for a,b in zip(row[FIELDS[0]]['values'],other[FIELDS[0]]['values'])]))
    if comparisons:
        lines += ['',f'双模式 E2E 差异 (dit/e2e − 1)：`{comparisons}`；仅描述本次观测，不宣称统计显著性。']
    if historical is not None:
        lines += ['','## 历史对比','',
            '2026-10-13 H20 原始 baseline，每组 3 次正式测量。旧版有输出转换后的显式 CUDA 同步和额外时钟，本版直接让原生 pipeline 保存；差异不能直接归因于优化。原 auto 后端为 SDPA，本版默认显式 sdpa。不同配置/源码/硬件/核心包不计算百分比；模型权重内容未完整核验。','',
            '| GPU | 模式 | 旧 E2E / DiT | E2E 差异 | DiT 差异 | 可比性 |','|---:|---|---:|---:|---:|---|']
        for row in rows:
            old = next((r for r in historical['results'] if r['gpus']==row['gpus']),None)
            if old is None:
                continue
            reasons = mismatches(plan,row,historical)
            row['historical_mismatches']=reasons
            delta = [f"{100*(row[f]['mean']/old[f]['mean']-1):+.3f}%" if not reasons and row['complete'] and f in row else '—' for f in [FIELDS[0],FIELDS[2]]]
            lines.append(f"| {row['gpus']} | {row['mode']} | {old[FIELDS[0]]['mean']:.3f} / {old[FIELDS[2]]['mean']:.3f} | {' | '.join(delta)} | {', '.join(reasons) or '配置匹配，边界差异见上文'} |")
    lines += ['',f"组状态：`{status['completed']}`。",'',
        '原始数据：`offload_off/gN/metrics.json`；每 rank 的 forward 顺序标签见 `effective_config.json/forward_layout`，CFG 双分支分别计一次完整 forward；模型有效配置、日志与视频保存在同组目录。','']
    findings = review_measurements(rows)
    lines += ['', '复查提示：CV > 2% 或连续三次变快且首尾下降 > 2%；单次无法评估波动，不自动增加请求次数。', '']
    lines += [f'- {finding}' for finding in findings] or ['- 暂无。']
    from sparse_attention import report_lines
    lines += report_lines(rows)
    (suite/'report.md').write_text('\n'.join(lines))
    write_json(suite/'summary.json',dict(status=status['status'],plan=plan,
        results=[{k:v for k,v in r.items() if k!='effective'} for r in rows],timing_comparisons=comparisons, review=findings))


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('suite',type=Path)
    parser.add_argument('--baseline',type=Path,help='Opt in to historical comparison')
    args=parser.parse_args()
    generate_report(args.suite,baseline=args.baseline)

