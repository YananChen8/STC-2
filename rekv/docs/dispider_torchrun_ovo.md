# 用 torchrun 多卡并行复现 Dispider 在 OVOBench 的推理

你当前目录里提到：
- `dispider.py`、`run_ovo.sh` 来自 ovo_ben（用于在 OVOBench 推理 Dispider）；
- 其余代码来自原始 Dispider。

下面给出**最小改造方案**：只改 `dispider.py` 的数据切分与结果聚合，再用 `torchrun` 启动。

---

## 1) 在 `dispider.py` 增加分布式初始化

核心要点：
- 从环境变量读取 `RANK`、`WORLD_SIZE`、`LOCAL_RANK`（torchrun 自动注入）；
- 每个进程绑定一张卡：`torch.cuda.set_device(local_rank)`；
- 初始化进程组；
- 推理后只在 rank0 汇总并落盘。

参考实现（可直接嵌入到你的 `dispider.py`）：

```python
import os
import json
import torch
import torch.distributed as dist


def setup_distributed(args):
    if not args.distributed:
        return 0, 1, 0

    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    local_rank = int(os.environ["LOCAL_RANK"])

    torch.cuda.set_device(local_rank)
    dist.init_process_group(backend="nccl", init_method="env://")
    return rank, world_size, local_rank


def cleanup_distributed(args):
    if args.distributed and dist.is_initialized():
        dist.barrier()
        dist.destroy_process_group()
```

---

## 2) 按 rank 切分 OVOBench 样本

不要让所有卡重复跑全量数据。推荐按索引切分：

```python
def shard_data(data, rank, world_size):
    # 与 DistributedSampler 对齐：rank 取自己那一份
    return data[rank::world_size]
```

主流程中：

```python
annotations = json.load(open(args.anno_path, "r"))
my_annotations = shard_data(annotations, rank, world_size)

local_results = []
for sample in my_annotations:
    # 调你现有的单样本推理逻辑
    pred = infer_one_sample(sample, model, processor, args)
    local_results.append(pred)
```

---

## 3) 多卡结果聚合（rank0 合并）

```python
def gather_results(local_results, rank, world_size, output_json):
    if world_size == 1:
        with open(output_json, "w") as f:
            json.dump(local_results, f, ensure_ascii=False, indent=2)
        return

    gathered = [None for _ in range(world_size)] if rank == 0 else None
    dist.gather_object(local_results, gathered, dst=0)

    if rank == 0:
        merged = []
        for part in gathered:
            merged.extend(part)
        with open(output_json, "w") as f:
            json.dump(merged, f, ensure_ascii=False, indent=2)
```

---

## 4) 参数建议

给 `dispider.py` 增加：
- `--distributed`（bool）
- `--merge_on_rank0`（bool，可选）
- `--output_dir`
- 你已有的 `--model_path --anno_path --video_dir ...`

并在主函数中串起来：

```python
rank, world_size, local_rank = setup_distributed(args)
...
my_annotations = shard_data(annotations, rank, world_size)
...
if args.merge_on_rank0:
    gather_results(local_results, rank, world_size, output_json)
...
cleanup_distributed(args)
```

---

## 5) 启动命令

已提供脚本：`run_ovo_torchrun.sh`。

也可手动执行：

```bash
torchrun --standalone --nnodes=1 --nproc_per_node=4 dispider.py \
  --model_path /path/to/ckpt \
  --anno_path /path/to/ovo_bench_new.json \
  --video_dir /path/to/chunked_videos \
  --output_dir ./outputs/ovobench \
  --distributed \
  --merge_on_rank0
```

---

## 6) 常见坑位

1. **重复推理**：没做 `data[rank::world_size]` 切分。
2. **卡号错位**：没 `torch.cuda.set_device(local_rank)`。
3. **死锁**：某 rank 异常退出，其他 rank 卡在 `barrier/gather`。
4. **结果覆盖**：每卡写同一个文件。建议先 gather 到 rank0 后统一写。
5. **样本顺序**：多卡 merge 后顺序可能变化，评测按 `id` 对齐即可。

---

## 7) 最小验证步骤

1. 先单卡跑 10 条样本（确认功能正确）。
2. 再 2 卡跑同 10 条（检查无重复、无缺失）。
3. 再全量多卡跑并评测。

如果你愿意，我可以下一步按你的 `dispider.py` 实际代码结构，给你一版**可直接粘贴的精确 patch**（包括函数名和插入位置）。
