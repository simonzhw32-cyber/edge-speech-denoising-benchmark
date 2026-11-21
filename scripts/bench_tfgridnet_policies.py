"""Measure TF-GridNet memory/time policies without touching training runs.

Each policy runs in its own process so a CUDA OOM cannot poison the next one.
This is a measurement harness, not a training entry point.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

POLICIES = [
    # stride, policy, checkpoint_blocks, amp
    (128, "fp32_none", 0, "none"),
    (128, "fp32_ckpt4", 4, "none"),
    (128, "fp32_ckpt6", 6, "none"),
    (128, "fp16_none", 0, "fp16"),
    (128, "fp16_ckpt4", 4, "fp16"),
    (128, "fp16_ckpt6", 6, "fp16"),
    (128, "fp16_ckpt_b0", "0", "fp16"),
    (128, "fp16_ckpt_b1", "1", "fp16"),
    (128, "fp16_ckpt_b2", "2", "fp16"),
    (128, "fp16_ckpt_b3", "3", "fp16"),
    (128, "fp16_ckpt_b4", "4", "fp16"),
    (128, "fp16_ckpt_b5", "5", "fp16"),
    (256, "fp32_none", 0, "none"),
    (256, "fp16_none", 0, "fp16"),
]


def _install_checkpointing(blocks, indices):
    import torch
    import torch.utils.checkpoint as cp
    from collections import OrderedDict

    selected = set(indices)

    def forward(self, input, ilens, additional=None):
        from speech_denoising.models.tfgridnet import network as netmod

        n_samples = input.shape[1]
        if self.n_imics == 1:
            input = input[..., None]
        mix_std = torch.std(input, dim=(1, 2), keepdim=True).clamp_min(1e-8)
        input = input / mix_std
        batch = self.enc(input, ilens)[0]
        batch0 = batch.transpose(1, 2)
        batch = torch.cat((batch0.real, batch0.imag), dim=1)
        n_batch, _, n_frames, n_freqs = batch.shape
        batch = self.conv(batch)
        for index in range(self.n_layers):
            if index in selected:
                batch = cp.checkpoint(blocks[index], batch, use_reentrant=False)
            else:
                batch = blocks[index](batch)
        batch = self.deconv(batch)
        batch = batch.view(n_batch, self.n_srcs, 2, n_frames, n_freqs)
        batch = netmod.new_complex_like(batch0, (batch[:, :, 0], batch[:, :, 1]))
        batch = self.dec(batch.view(-1, n_frames, n_freqs), ilens)[0]
        batch = self.pad2(batch.view(n_batch, self.n_srcs, -1), n_samples)
        batch = batch * mix_std
        return [batch[:, src] for src in range(self.n_srcs)], ilens, OrderedDict()

    return forward


def _run_child(args):
    import time

    import torch
    import yaml

    from speech_denoising.models.tfgridnet import network as netmod
    from speech_denoising.models.tfgridnet.model import TFGridNetModel

    config = yaml.safe_load(Path(args.config).read_text(encoding="utf-8"))
    params = dict(config["model_params"])
    params["stride"] = args.stride
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    result = {
        "stride": args.stride,
        "hop_ms": args.stride / 16.0,
        "config": args.config,
        "policy": args.policy,
        "checkpoint_blocks": args.checkpoint_blocks,
        "amp": args.amp,
        "status": "ok",
    }
    try:
        model = TFGridNetModel(**params).cuda().float()
        result["parameters"] = sum(p.numel() for p in model.parameters())
        if args.checkpoint_blocks:
            selected = (list(range(args.checkpoint_blocks))
                        if isinstance(args.checkpoint_blocks, int)
                        else args.checkpoint_blocks)
            netmod.TFGridNetNetwork.forward = _install_checkpointing(
                model.network.blocks, selected
            )
        if args.subblock_intra or args.subblock_inter or args.subblock_attention:
            from speech_denoising.training.tfgridnet_policy import install_subblock_checkpointing
            install_subblock_checkpointing(
                model.network,
                intra=args.subblock_intra,
                inter=args.subblock_inter,
                attention=args.subblock_attention,
            )
            result["subblock"] = {
                "intra": args.subblock_intra,
                "inter": args.subblock_inter,
                "attention": args.subblock_attention,
            }
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
        noisy = torch.randn(1, 64000, device="cuda")
        clean = torch.randn(1, 64000, device="cuda")

        def step():
            optimizer.zero_grad(set_to_none=True)
            if args.amp == "fp16":
                with torch.autocast("cuda", dtype=torch.float16):
                    estimate = model(noisy)
                    loss = (estimate.float() - clean).square().mean()
            else:
                estimate = model(noisy)
                loss = (estimate - clean).square().mean()
            loss.backward()
            optimizer.step()
            return float(loss.detach())

        # One warm-up step allocates cuDNN workspaces and JIT kernels.
        step()
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        times = []
        losses = []
        for _ in range(args.steps):
            torch.cuda.synchronize()
            started = time.perf_counter()
            losses.append(step())
            torch.cuda.synchronize()
            times.append(time.perf_counter() - started)
        result.update(
            peak_alloc_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
            step_seconds=sum(times) / len(times),
            step_seconds_min=min(times),
            step_seconds_max=max(times),
            loss_first=losses[0],
            loss_last=losses[-1],
            steps=args.steps,
        )
    except torch.OutOfMemoryError as error:
        result.update(
            status="oom",
            error=str(error).splitlines()[0],
            peak_alloc_gib=torch.cuda.max_memory_allocated() / 2**30,
            peak_reserved_gib=torch.cuda.max_memory_reserved() / 2**30,
        )
    print(json.dumps(result, sort_keys=True))


def _run_parent(args):
    rows = []
    for stride, policy, checkpoint_blocks, amp in POLICIES:
        command = [
            sys.executable,
            str(Path(__file__).resolve()),
            "--child",
            "--stride", str(stride),
            "--policy", policy,
            "--checkpoint-blocks", str(checkpoint_blocks),
            "--amp", amp,
            "--config", args.config,
            "--steps", str(args.steps),
            "--seed", str(args.seed),
        ]
        environment = dict(os.environ)
        environment["PYTHONPATH"] = str(Path(__file__).resolve().parents[1])
        completed = subprocess.run(
            command, cwd=Path(__file__).resolve().parents[1],
            capture_output=True, text=True, env=environment, timeout=args.timeout,
        )
        payload = None
        for line in reversed(completed.stdout.splitlines()):
            try:
                payload = json.loads(line)
                break
            except json.JSONDecodeError:
                continue
        if payload is None:
            payload = {
                "stride": stride,
                "policy": policy,
                "status": "crashed",
                "returncode": completed.returncode,
                "stderr": completed.stderr[-800:],
            }
        rows.append(payload)
        print(json.dumps(payload, sort_keys=True), flush=True)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(rows, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"Wrote {output}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--child", action="store_true")
    parser.add_argument("--stride", type=int, default=128)
    parser.add_argument("--policy", default="fp32_none")
    parser.add_argument("--checkpoint-blocks", default="0",
                        help="integer count of first blocks, or comma-separated indices")
    parser.add_argument("--amp", choices=("none", "fp16"), default="none")
    parser.add_argument("--subblock-intra", action="store_true")
    parser.add_argument("--subblock-inter", action="store_true")
    parser.add_argument("--subblock-attention", action="store_true")
    parser.add_argument("--config", default="configs/tfgridnet.yaml",
                        help="model config; use configs/tfgridnet_dns.yaml for 4-layer")
    parser.add_argument("--steps", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--timeout", type=int, default=900)
    parser.add_argument("--output", type=Path,
                        default=Path("results/paper_runs/tfgridnet_policy_bench.json"))
    args = parser.parse_args()
    raw = str(args.checkpoint_blocks).strip()
    args.checkpoint_blocks = (
        [int(part) for part in raw.split(",") if part.strip()]
        if "," in raw else int(raw)
    )
    if args.child:
        _run_child(args)
    else:
        _run_parent(args)


if __name__ == "__main__":
    main()
