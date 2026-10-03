r"""Launch the existing Phase 3.6 TF-GridNet probe with resource limits.

No source changes, assets, optimizer, checkpoint, commit or push.
Six-layer random model; unchanged batch=4, segment=64000, seed=42, epoch=1.
CUDA caching allocator: 50% of visible device memory; CPU threads=4,
interop threads=1; child wall-time limit=180 seconds. The allocation limit
is not a system-wide GPU/host-RAM limit or a guarantee of desktop responsiveness.

Download to Downloads. From the repo root:
  venv\gpu\Scripts\python.exe "%USERPROFILE%\Downloads\probe_tfgridnet_limited.py" --repo "%CD%" --check
  venv\gpu\Scripts\python.exe "%USERPROFILE%\Downloads\probe_tfgridnet_limited.py" --repo "%CD%"

The first command reads source/metadata only. The second launches exactly one
fresh child Python process. OOM under this cap is not proof that the unchanged
batch cannot fit with a different memory allowance. Timings are not benchmark RTF.
Normal completion records limits and launcher SHA-256 inside the new probe JSON.
Timeout/Ctrl+C stops only the child; no success report is invented. A killed
child may leave no JSON; an already published complete report is never replaced.
Retain this launcher with the report when archiving the machine's evidence.
"""

import argparse
import hashlib
import os
from pathlib import Path
import subprocess
import sys

FRACTION = 0.5
CPU_THREADS = 4
INTEROP_THREADS = 1
TIMEOUT_SECONDS = 180
EXPECTED = {'.gitignore': 'cdbae0e7ea14b194601c1ccbacc99ced50ca9895dc4a44d5d2190db4404420cc', 'README.md': '04d43e1d7387b70c730c31caed58de2fe7161167a4a1fe1bb5f85a60fad45495', 'benchmark_reports/gtcrn_pretrained_voicebank_test824_windows.json': '38a0444bd0aadb3f54a2dead4f9ebeb090f1fa8c4f9ba61c88ff5eda845f6304', 'benchmark_reports/gtcrn_windows_environment.txt': '5e14503995e77d894f0903a94815b7eae77ca6dd0298eb7b4289a281ba466c5b', 'benchmark_reports/tfgridnet_dns_all_metrics_windows_environment.txt': '0ffda45bf9d3498c74192705d99116e13bc609e675283ecefc653c076717e63e', 'benchmark_reports/tfgridnet_dns_evaluation_windows_environment.txt': 'cac1d33d7b539973c128a0278b5e213ac985334aa5243823b9c0e51a747be20e', 'benchmark_reports/tfgridnet_dns_pretrained_check_windows.json': 'ecfb318acf93fd70b9ab7048d980fa1a66487bc21a18d355ad6984d68a38ebc7', 'benchmark_reports/tfgridnet_dns_pretrained_voicebank_test824_windows.json': 'a853351ccce0a52692610ef4f97727766322f90fe0c3d94f9f85bbe3e82c09aa', 'benchmark_reports/tfgridnet_dns_pretrained_voicebank_test824_windows_without_pesq.json': '70b026a0c5655dc881d12cdf40be6378060352199f03a1d23f63dfac5305969e', 'benchmark_reports/tfgridnet_dns_windows_environment.txt': '1d2f6518e570e03c1e83de70a2848a9965f9d5fff5951d73c2a279e206632cdd', 'checkpoints/.gitkeep': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'configs/default.yaml': 'fc07fff636e2ef9e9efc9923c89675b16cc346428d14f7fd5ce253f33b7bcb62', 'configs/gtcrn.yaml': 'bfc9b1384aba8fca8d765a972861c7c23bb777da5cc5e4a7da6732807ac02d7e', 'configs/lisennet.yaml': '702fa7d50f0a078a1efd9b1593030859eaf26e4ff1458b458b0db009084ba556', 'configs/tfgridnet.yaml': '4332b61571dbdd0260a75d205bc5e8593020cb17b6d1d371bd9a325dac001a93', 'configs/tfgridnet_dns.yaml': 'e4c032f95cd5382a358aff4a903aea0cdd2dc97a68675715955fef124e820c69', 'configs/training_protocol.yaml': '7222e3b94fbdff817a617096a803016fecce200ccdba49f42dc5181f82ca279d', 'configs/voicebank.json': '84e498e77a6c038b335482bf3272af98118ce82812e6b8e4875e4c99462e01a6', 'docs/benchmark_protocol.md': '88efccb6d76e25148eb538326664d4cfb0aa6c9d2b1f5069af545da15e834d6c', 'docs/history.md': '1aeeaebbe3c2edca1f83082c44371c799bc47578ce7ade00bf9525bbf7803c59', 'docs/lisennet_checkpoint_audit.json': '7a7c87bed6dd3c5ed838ec60a085b7859181b835272c1fb46ff7775b8623018a', 'docs/lisennet_checkpoint_audit.md': '743e8d7c26e28d6f877ca317fe8c7fb96d12e7fb24e506dbd65412cdf706b26d', 'docs/results.md': '531d95ce7eb18a3d8ace9bdc7bb186c4c0973050b08997b5da29c2825f66db31', 'docs/tfgridnet_checkpoint_audit.json': 'bb1699b6349c646a29abe9b50dd5efbab5fb2f08fd9aaa40e07b6ec76e29d61c', 'docs/tfgridnet_checkpoint_audit.md': '2e87a73d3cf3d93bebca25d18c4653d773c8845f680f7d5fbcca0eb6aee18e60', 'docs/tfgridnet_dns_evaluation.md': 'b5383a663c55e15f0f5ad8c78a884856a3f25aee8237920827a51f86d87a3431', 'docs/tfgridnet_dns_evaluation_integration.json': '07e8326dc215c52c0545733f6e7bfe75c661381c990f03798523f1f37b985472', 'docs/tfgridnet_dns_pretrained_integration.json': '076e62cb07f540304f684e125fbc3a2d05573730645a87e2ffbe9bd6b87f28de', 'docs/training_components.md': '2571928ddc1798c2e95f25b722011ed856c33370812f019e58e68280d0d941e8', 'docs/training_preflight.md': 'c7651d77b24dc721141a6288a577b0182dd61fcdd74def6fb4f461027435b51d', 'docs/training_protocol.md': '35da25bb803abebb5a8c280962f8ee68c96e0be76318fb6cfba9e3b4390126aa', 'docs/training_runner.md': '127a1fa7351fab1a2f1b4719e84cf5551c6f1bc2ae7b01622a676099a931a85f', 'docs/training_state.md': 'b02834f92581f7ca5b2626355d30a67c616eaaf4b8716447bca5c19046bbef25', 'docs/windows_restore.md': '4d70a880cf97d10c1d79a063546d1c0c5ce6733ccc966f397bb62c01ff029d4f', 'pyproject.toml': 'fd6c04e144e9cafe40aa04132e03ea6acf5dc5d65aa6ee54792ede3ad642ce99', 'requirements-benchmark.txt': 'c4e36bf431677db068b962ff8a6bdde15e32aac9e0dcb1e075ab690a99a6ccf1', 'requirements-tfgridnet.txt': 'c74241b13b2775c7e2f8694728bf02abdcea58564cd2c68212795c2135ab9af2', 'requirements.txt': '2af95a4463562ceb359623481879bd934a81ff2aea92e48d84becc2278f54099', 'results/.gitkeep': 'e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855', 'scripts/_tfgridnet_dns_reference.py': 'f3405e4f2ab9eda995caee71f085cec9b1cf061c3ff71b5e077855f90b572b84', 'scripts/evaluate.py': '48bb2a1368ecfc31a2d30263922fdb9007f1a1e5c368d622351028b2a48136f6', 'scripts/fetch_gtcrn_checkpoint.py': 'e00e29bb35cd37e921d335f98cb1d4a8123edc1b4c114792ce4b0cc4f4921f82', 'scripts/fetch_tfgridnet_checkpoint.py': '131a4ceff007b367094b2e353a82c8ab79b8bf66631b48364ecb50d134014c2a', 'scripts/inference.py': '99468f57c30d3de65af95185cfe426bd7724b83fb0a69f95be817f78b3a57a58', 'scripts/inference_tfgridnet.py': 'e696170983f2ca794b9e7c9f955f80cec4184189f41bb8a827b385be009cba3e', 'scripts/plan_training.py': '489991278688c9b8fd271604ca7de9251761f28259acc8e2738a64d5b73b9959', 'scripts/preflight_training.py': '29df0a6f5ee8368710c812b3a67272543bb4cda41a45c5797c6c066100cedf78', 'scripts/prepare_data.py': 'd1da28abbd28496f6cc414befcaad517341a8657c6708a495268923f92909858', 'scripts/smoke_benchmark.py': 'b05684237b49fc624e030d891c2d8cea1affc48ac489360135d9d4d7dd7377e4', 'scripts/smoke_evaluation_cli.py': '881f233d8bb7924e85c67cea6e6be9b63848028e003173333fe654c55dff570d', 'scripts/smoke_gtcrn.py': '9b2152539e3062a35487603ea87e6f8753b4d4d2b8ee576d3847f2d0dcbb60fc', 'scripts/smoke_gtcrn_pretrained.py': '0559d18b44e1b6b6e38c3d6ec8cb167b8096f4d7bb70666df82b0b8b98e240c9', 'scripts/smoke_lisennet.py': '699c6ec021fc6c719dcd46dd3ccbeccaa641ca1b672fbdd895dbdce7a100569e', 'scripts/smoke_tfgridnet.py': '292b11a40a229f527aae16a97476bcbc4b47e02f9a5c6f19082a64930129046f', 'scripts/smoke_tfgridnet_pretrained.py': 'd597bc8788bd623639fa83567539b8685a54919b3b0d5c423aaaa4c754e61020', 'scripts/smoke_training_components.py': '454fac65ff9ae3fe98362c13ccc422ed4b85516cc4e41a92f5d5d6bd1361d22d', 'scripts/smoke_training_plan.py': 'c47d3cc7ef8efa507f20163ebedaaec9f6352170f5a0e25f2fabd0e7a3c2c4a5', 'scripts/smoke_training_preflight.py': '38a98aff394be14a46b2099f0e9d9770dd5abb8100b05a58c8bcef1c6de27733', 'scripts/smoke_training_runner.py': '4eb8321c2ca773c36480123504bb96289457bae79801b7f76c21a8d0366c1dd9', 'scripts/smoke_training_state.py': 'e29d60e43b2e6ef8cc99d3e1b076f556acbf2090072a580fc9d0007ef2c1c966', 'scripts/train.py': '1d2ae8d0eb8904166216563343838c2cb1e6f694d6782c5be2085bbcfa42b190', 'scripts/validate.py': 'f69e9304d0a817d9a8afc898ba0ec644bcbde1500e1fae15222380594545865a', 'speech_denoising/__init__.py': '10da722ac6c0a2c3cfd5dc80e2bcae4227833d312510736b0ac892c71409df53', 'speech_denoising/datasets/__init__.py': '6b51668c0ba6c88e54d33f9029951cf51bf27fc38864cebc0bde751829688f94', 'speech_denoising/datasets/voicebank.py': '15e387cd9a9cc2723773b3f48ae3caa5773270a443f9fbe64d8cd04407405993', 'speech_denoising/losses/loss.py': '591f022c00593c34b353c90129efaf84ab427b30f0e8ec7a7714e18172798609', 'speech_denoising/metrics/__init__.py': 'f9f605f0744e007851322684d0b737d8c1a581b2ff4a001ddfc594e702c2623a', 'speech_denoising/metrics/audio_metrics.py': '11fb7a68a79abb64620d5e8c816f777cef7b367595d1bc3335839faf81da6cd2', 'speech_denoising/models/__init__.py': 'd6b811ca0a6cdd7502b043a7d923a15dbd48d561115041c235e279d4d01d49a8', 'speech_denoising/models/base.py': 'dd8a34b2cfcf5895d2f76b7f9fb16edffaa97accf8c15d2c5d5b3bddd09d80b5', 'speech_denoising/models/gtcrn/LICENSE.upstream': 'bb7b4687ee3021f513381ab8b547435702f3ed851709b05675294a3935c733be', 'speech_denoising/models/gtcrn/PRETRAINED.md': '1903e2a3425bc55d6db20dbc36532704bf47bcd5bf1c1b1526d88f22974fde8a', 'speech_denoising/models/gtcrn/SOURCE.md': '572d889fbe5a59b1d50741d2e6623bf57936e9a961e18dbec8f1bd6f92dff10b', 'speech_denoising/models/gtcrn/checkpoint.py': '440375b8aefef1ac3c929ddc8195107b13f66dd64e92351160562d49d2668489', 'speech_denoising/models/gtcrn/model.py': 'e5b29cd019daca68b459e2e20e8b9ca571f9d931c95b6c7168acf9cf2b21e082', 'speech_denoising/models/gtcrn/network.py': 'a9a2a8268c682318b58bedc2c9627764fb9397e95adc62c491e885657e0bd9c7', 'speech_denoising/models/lisennet/LICENSE.upstream': '60f81c335a77fe6023bb4b2b9c03e1e0b224a58a138c5a3c99ef1003c66eef92', 'speech_denoising/models/lisennet/SOURCE.md': '1335dd5596f51b5b2978015cfdfaa874d37af6e252820df1778ec6ea6c49c709', 'speech_denoising/models/lisennet/dpr_layer.py': '2108c05517d0f82763ef374b9858a477b11a35a56b295a0492f4bf64ce10c5a3', 'speech_denoising/models/lisennet/model.py': '3b8c1be5d2963e9221fac55736c80bd5bd685050c5885bdda41de07b951e780a', 'speech_denoising/models/lisennet/network.py': 'bb45dffa1cc5e110f7549cc98c5519e0eb61437fdecaa40846e2a38347529dfc', 'speech_denoising/models/tfgridnet/CHECKPOINT_CARD.upstream.md': '541171c2c99a4f4931ebd7fbb382ad6063ed7cf83d8e157a107b3b1404b29458', 'speech_denoising/models/tfgridnet/LICENSE.upstream': '4696c3c9551da6fef1368be1e4ed2c80cf13e55448c6dcf2aba9462f5ff29ef5', 'speech_denoising/models/tfgridnet/PRETRAINED.md': '8f2518a25434c893035995c36d112e412c30a537681298783680e8d139253158', 'speech_denoising/models/tfgridnet/SOURCE.md': '13528ff6446653badbf38e7047f5d30958cc1ee3080dff745bf7328b925fef9f', 'speech_denoising/models/tfgridnet/assets.py': '0d37607df576a1874ecbff22d96b0b0cf2bb04f7269d6069e8b07ba6460682c8', 'speech_denoising/models/tfgridnet/checkpoint.py': 'f58120d19dee811ecfde763a5a16927f2badddddc4a199320e2dc3bfbeec5d01', 'speech_denoising/models/tfgridnet/model.py': '2357b33a7948d7ac791250574b52fe057ead9723f8c538f94ac8858cf69c77a8', 'speech_denoising/models/tfgridnet/network.py': '1e2eea9e2caa42c5a54ac671f042940293cd496b392fd8f153642db364b816da', 'speech_denoising/models/tfgridnet/profiles.py': '2ec2bda08e1f3d850f7dd144f9e09b13552a8c0e152262ec2d7a5ab97faf35c4', 'speech_denoising/models/tfgridnet/spectral.py': 'a79935743938ccba2fbf7757a373d7b746dbaeaa84cdef4149814aadaf2285ad', 'speech_denoising/training/__init__.py': 'ed673b8581b465994a3c8b345b2b70471f524799899feefdad438e279b05dfa5', 'speech_denoising/training/batching.py': '5af6b9dbae1e607c2a841dbf68525a259139ce11c10b6713aaaf972d99f5c4da', 'speech_denoising/training/data.py': 'f99b585ad6578036e1aa55c225dbf1a387b1ae31b347efb734a218357228bba4', 'speech_denoising/training/fixture.py': '80d5abff7379b980575109f4381f8d415bb4f152bf3259fab5eee67bf4ebf203', 'speech_denoising/training/preflight.py': '3730728f365f2208494a9025f767418c55b87ca0414d21cda04d4da721fccbc2', 'speech_denoising/training/protocol.py': '3147b22ad6993baf09147b80b6d25030e43ae220fce873333b15261f7a28d0a8', 'speech_denoising/training/run_store.py': '58025d6f73db8c2a6a7d5c3037bb287c52ad349aca557c09119fd97e06494c24', 'speech_denoising/training/runner.py': '2e4a330613ab6b4edd11b182f7d7a3318d55119a17913fcd0fc7b538b0101286', 'speech_denoising/training/selection.py': '76137caa5e860f937af7f8c68a1dfb258e81eecc5794078fbed517b5dce5dd42', 'speech_denoising/training/state.py': '549e212e57769f245f637945a5100bfffaeb0639304fd97e7dc92ba07b50fca0', 'speech_denoising/training/validation.py': '78a92fc45f39eb1ddd81bb24b03c0f69cd775f50519fe7eb3f702cda3de5733f', 'speech_denoising/utils/utils.py': 'c8e34a5d2195564ae2bf863544133662580e717f3fe9f873e1917ff387d466d9', 'docs/training_probe.md': '310d12edfb51d8403ae66ed09cf270a186bd8b1af1579543e95988a0be381f99', 'scripts/probe_training.py': '121bef274d0f53f60dad20b15ac8fc462a554e793223ed750d2fc328d1d471ab', 'scripts/smoke_training_probe.py': '0594928db00e95111abe21b4d0f78cc7c2e56ce9548da8d290061854109eceb7', 'speech_denoising/training/probe.py': '08fd1bda174cc47fb8033d5cc48b123adca12a68e48264eead119b09912d5609'}


def file_hash(path):
    return hashlib.sha256(path.read_text(encoding="utf-8").encode()).hexdigest()


def verify_sources(repo):
    for name, wanted in EXPECTED.items():
        path = repo / name
        if path.is_symlink() or any(p.is_symlink() for p in path.parents):
            raise ValueError(f"Symlink source path: {name}")
        if not path.is_file() or file_hash(path) != wanted:
            raise ValueError(f"Expected verified Phase 3.6 source: {name}")


def prepare(repo, output):
    verify_sources(repo)
    os.chdir(repo)
    sys.path.insert(0, str(repo))
    from speech_denoising.training.probe import bind_preflight, output_path
    manifest = repo / "data/voicebank-demand-16k/train_manifest.json"
    preflight = repo / "results/training_preflight/gtcrn_audio.json"
    output = output_path(output, (manifest, preflight))
    _, _, _, plan, _ = bind_preflight(manifest, preflight, "tfgridnet")
    if (plan["model_settings"]["profile"] != "local_6layer"
            or plan["model_settings"]["constructor_kwargs"]["n_layers"] != 6
            or plan["protocol"]["training"]["batch_size"] != 4
            or plan["protocol"]["training"]["segment_samples"] != 64000):
        raise ValueError("The six-layer draft batch/profile differs from this launcher")
    return manifest, preflight, output


def configure(torch):
    # Must happen before model imports, eager execution or allocation.
    torch.set_num_threads(CPU_THREADS)
    torch.set_num_interop_threads(INTEROP_THREADS)
    if not torch.cuda.is_available():
        raise ValueError("CUDA unavailable; no CPU fallback")
    if torch.cuda.memory.get_allocator_backend() != "native":
        raise ValueError("This bounded check requires the native CUDA caching allocator")
    torch.cuda.memory.set_per_process_memory_fraction(FRACTION, 0)
    total = torch.cuda.get_device_properties(0).total_memory
    return {"cuda_allocator": "native", "cuda_allocator_fraction": FRACTION,
            "cuda_allocator_limit_bytes": int(total * FRACTION),
            "cpu_threads": CPU_THREADS, "interop_threads": INTEROP_THREADS,
            "child_wall_time_limit_seconds": TIMEOUT_SECONDS,
            "coverage": "PyTorch caching allocator; not all driver GPU or host RAM allocations"}


def worker(repo, manifest, preflight, output, launcher):
    import torch
    limits = configure(torch)
    from scripts import probe_training
    original = probe_training.publish_report
    launcher_hash = file_hash(launcher)
    def publish(path, report, protected=()):
        verify_sources(repo)
        if file_hash(launcher) != launcher_hash:
            raise ValueError("Launcher changed during the probe")
        report["runtime"]["resource_limits"] = limits
        report["launcher"] = {"file_name": launcher.name, "source_sha256": launcher_hash}
        return original(path, report, protected)
    old_argv = sys.argv
    probe_training.publish_report = publish
    sys.argv = ["probe_training", "--model", "tfgridnet", "--device", "cuda",
                "--manifest", str(manifest), "--preflight", str(preflight), "--output", str(output)]
    print(f"Limits: CUDA allocator {limits['cuda_allocator_limit_bytes'] / 2**30:.2f} GiB; "
          f"CPU threads {CPU_THREADS}; child time {TIMEOUT_SECONDS}s", flush=True)
    try:
        probe_training.main()
    finally:
        sys.argv = old_argv
        probe_training.publish_report = original


def child_environment():
    environment = os.environ.copy()
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
        environment[name] = str(CPU_THREADS)
    # Isolate allocator selection to the child and record it with the result.
    environment["PYTORCH_ALLOC_CONF"] = "backend:native"
    environment["PYTORCH_CUDA_ALLOC_CONF"] = "backend:native"
    return environment


def stop_child(child):
    if child.poll() is None:
        child.terminate()
        try:
            child.wait(timeout=10)
        except subprocess.TimeoutExpired:
            child.kill()
            child.wait(timeout=10)


def supervise(command, repo):
    flags = subprocess.CREATE_NEW_PROCESS_GROUP if os.name == "nt" else 0
    child = subprocess.Popen(command, cwd=repo, env=child_environment(), creationflags=flags)
    try:
        return child.wait(timeout=TIMEOUT_SECONDS)
    except subprocess.TimeoutExpired:
        stop_child(child)
        print("Stopped: 180-second child time limit. No success result is inferred; "
              "inspect the console and any already published report.", file=sys.stderr)
        return 124
    except KeyboardInterrupt:
        stop_child(child)
        print("Stopped: user interruption; existing reports preserved.", file=sys.stderr)
        return 130
    finally:
        stop_child(child)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path,
                        default=Path("results/training_probe/tfgridnet_cuda_limited.json"))
    parser.add_argument("--check", action="store_true", help="Read source and preflight only; no Torch/model")
    parser.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    args = parser.parse_args()
    try:
        repo = args.repo.resolve(strict=True)
        output = args.output if args.output.is_absolute() else repo / args.output
        manifest, preflight, output = prepare(repo, output)
        launcher = Path(__file__).resolve(strict=True)
        if args.check:
            print("PASS: preserved Phase 3.6 sources, current preflight and new output path")
            print("Six-layer TF-GridNet; batch 4 x 64000; no model execution")
            return 0
        if args.worker:
            worker(repo, manifest, preflight, output, launcher)
            return 0
        command = [sys.executable, str(launcher), "--repo", str(repo), "--output", str(output), "--worker"]
        return supervise(command, repo)
    except (OSError, ValueError, RuntimeError, KeyError, TypeError, ImportError) as error:
        print(f"ERROR: {error}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
