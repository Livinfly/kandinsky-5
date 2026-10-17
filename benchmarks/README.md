# Kandinsky-5 T2V benchmark

独立于 `exp/` 和其他框架，可在本仓库 clone 后运行。默认沿用 Pro SFT 10s / SD / NABLA：
768×512、241 帧、24 FPS、50 个采样 step、CFG=5，共 **100 次完整 DiT forward**。
T2V 仅使用 [`baseline.json`](baseline.json) 中的文字 prompt，无参考图。

## 环境与启动

在本仓库根目录执行。先安装 [uv](https://docs.astral.sh/uv/getting-started/installation/)，
按官方 README 下载 Pro 10s DiT、Qwen2.5-VL-7B-Instruct、CLIP 和 Hunyuan VAE。
模型可以放在任意位置，下面路径均为示例；`--vae-path` 指含 `vae/` 子目录的根目录。

```bash
bash benchmarks/setup.sh
.venv/bin/python benchmarks/run.py \
  --dit-path /models/Kandinsky-5.0-T2V-Pro-sft-10s/model/kandinsky5pro_t2v_sft_10s.safetensors \
  --qwen-path /models/Qwen2.5-VL-7B-Instruct \
  --clip-path /models/clip-vit-large-patch14 \
  --vae-path /models/Kandinsky-5.0-aux \
  --gpus 8 4 2 1 --output outputs/benchmarks/pro_sd --background
python3 benchmarks/monitor.py outputs/benchmarks/pro_sd --watch
python3 benchmarks/report.py outputs/benchmarks/pro_sd
```

默认 **dit 模式（E2E + DiT）、1 次 warmup、1 次正式测试、offload 关闭**。
每组只加载一次模型，按指定卡数顺序串行执行。`--dry-run` 只打印命令；已有输出目录拒绝覆盖。
`--modes dit e2e --repeats 3` 可做双模式对照，每模式独立 warmup，正式请求按轮交替模式顺序。
`--validate-video` 在计时外完整解码检查 MP4 的帧数/尺寸/FPS，并记录文件 hash。

`setup.sh` 使用 uv、Python 3.12 和固定依赖 `requirements.lock`（torch 2.8 / CUDA 12.8）。
`UV`、`BENCH_VENV`、`PYTHON_VERSION` 可覆盖默认值，`--check` 仅检查现有环境。
默认非 NABLA attention 显式使用 SDPA，与旧环境 auto 实际选择一致；NABLA 仍使用官方 FlexAttention。
不需要额外安装 FlashAttention；切换其他后端需自行安装相应依赖。
脚本不按 GPU 名称改变计算，硬件仍须满足所装 PyTorch、Triton、BF16 和官方 kernel 的支持条件。

## 配置与 GPU 等待

复制 `baseline.json` 后通过 `--config my_config.json` 指定，直接调整 prompt、negative_caption、
seed、宽高、时长、采样步数、CFG、scheduler_scale、扩写开关和非 NABLA attention 后端。
也可用 `--prompt` 或 `--prompt-file` 临时覆盖 prompt。
宽高必须属于模型 YAML 的官方 resolution bucket；默认关闭扩写。
`--model-config` 可指定其他官方 T2V YAML 或自己的副本（例如 NABLA P / 窗口、RoPE），
默认 `configs/k5_pro_t2v_10s_sft_sd.yaml`。四个 checkpoint 路径统一由 CLI 覆盖，原 YAML 不被修改。
更换模型档位时须同步选择对应 checkpoint 和采样参数。卡数必须整除模型 attention heads；Pro 支持 1/2/4/8。
固定关闭 MagCache、Qwen NF4 和 text token padding；保留官方内部 compile。

GPU 不空闲时 **默认等待，不直接抢占或自动降配**：

- `CUDA_VISIBLE_DEVICES` 指定 nvidia-smi 索引或 GPU UUID（支持唯一前缀）；每组取其中前 N 张，
  子进程统一使用完整 UUID，避免重编号歧义。未设置时使用 nvidia-smi 列出的顺序。
- 目标 GPU 上存在计算进程、已用显存超过 256 MiB 或 utilization 非零，均保持等待；
  每 30 秒检查一次，连续两次空闲才启动。监控显示等待原因。查询失败直接报错，不当作空闲。
- 同机同用户的 Hunyuan / LongCat / Kandinsky 三个框架共用 `/tmp/video-benchmark-gpus-UID/` 的 GPU 文件锁；锁随进程退出释放。
  这不是系统调度器，无法阻止其他不遵守锁的程序在检查后启动。此实现面向整卡，不支持 MIG 分区选择。
- `--wait-for /another/suite/status.json` 可额外串行等待一个已有实验，避免它切换卡数组时的短暂无负载。
  前置实验异常结束或进程消失时报错。无前置依赖时不需要此参数。
- 进入推理后不再轮询 GPU；加载/预热阶段明确的 OOM 记录后继续下一卡数组；正式测量阶段 OOM 和其他 worker 错误停止矩阵。

## 计时与报告

[`worker.py`](worker.py) 明确标注 E2E START/END；[`timing.py`](timing.py) 在完整 DiT `__call__`
外记录 CUDA events，覆盖内部 FSDP/SP pre/post hooks。事件预分配、预初始化，无逐步同步。
`e2e` 请求直接使用原 DiT，不包裹、不记录事件。两模式都直接让官方 pipeline 保存 MP4，
不加中间 CUDA 同步；`distributed_e2e_s` 计至保存后的 barrier 返回。

DiT 主指标为各 rank 所有 forward 时间之和的最大值 `max(sum)`，另存 `sum(max)`。
`per_rank` 保留 E2E/barrier 时间、全部 forward events、实际 VAE tile/stride 和峰值显存。
只有 rank 0 保存视频，其他 rank 的 `response_e2e_s` 为 null。
`effective_config.json/forward_layout` 标明 scheduler step 与 conditional/unconditional 分支；
一次 forward 覆盖整个模型，不是单层 attention。

VAE 自适应切片选择后附带一个轻量 Python 记录，两模式相同，保持原 tile/stride。
只在排除的 warmup 保留浮点 latent/VAE 输出供计时后有限值验证；正式请求不延长这些浮点 tensor 的生命周期。
输入噪声、TE、CFG 合并、scheduler、最终 latent all-gather、VAE、转换与 MP4 写盘属于 E2E；
模型加载、等待、准备、预热、验证、事件查询与统计写盘排除在正式时间外。

`report.md` / `summary.json` 每组完成后更新，包含模型/推理配置、环境、E2E、DiT、
同模式单卡加速比、warmup 对照、可选双模式差异。单次标为“单次”，标准差为 null。
原始逐 rank 数据、日志和 MP4 在 `offload_off/gN/`；每 suite 保存计时代码快照，各组使用同一份代码。
编译缓存按 suite / group / rank 隔离；实验运行时不要修改框架推理源码、模型或输入配置快照。

历史比较默认关闭。可在启动时加 `--baseline benchmarks/reference/h20_20261013.json`，
或对已有结果按需生成，无需重新推理：

```bash
python3 benchmarks/report.py outputs/benchmarks/pro_sd --baseline benchmarks/reference/h20_20261013.json
```

不带 `--baseline` 重新生成会去掉历史章节和字段。旧版输出转换后有显式同步，报告会注明边界差异；
配置/源码/硬件/核心包不一致时不计算历史百分比。参考数值仅本地保存、不随脚本提交；使用 `--baseline` 时需提供本地参考 JSON。

CPU 检查：`.venv/bin/python -m unittest discover -s benchmarks -p 'test_*.py' -v`。


## 运行状态与失败提示

运行器自动更新实验目录的 `status.json`（原子替换文件），监控命令会直接显示错误摘要和日志位置。

- `status`：`waiting` / `running` / `complete` / `complete_with_failures` / `interrupted_or_failed`。
- `has_failures`：是否已有失败；失败后即使继续其他组也保留为 true。
- `failure_count`：已知失败组数；运行器自身错误由 `launcher_error` 单独记录。
- `last_error`：最近错误的 `summary`、`phase`、`directory`（组错误）、`log_file` 与 `error_files`。
- `completed`：每组结果、退出码和失败详情；`current` 为当前组，`updated_at_utc` 为状态更新时间。

`log_file` / `error_files` 相对于实验目录。后台启动日志位于实验目录旁的 `<实验名>.launcher.log`；
前台运行器自身异常也会写状态，完整堆栈显示在终端。各 rank 原始错误保存在组目录 `error_rankN.json`。
状态文件用于查看，不是控制入口：手动修改不会触发重试、跳过或恢复，并可能被下一次更新覆盖。
启动前参数校验失败时尚未创建实验目录；运行器被强制杀死也无法自行更新文件，监控会提示进程已退出。
已经启动的旧运行器保持原快照；新版监控可从其旧状态和 rank 错误文件展示失败信息，不改写其状态文件。

多次正式测量 CV > 2% 或至少三次连续变快且首尾下降 > 2% 时，报告提示复查；不自动增加请求次数。双模式对照保留均值和每轮配对差异。
