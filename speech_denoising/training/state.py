"""Local epoch-boundary state, separate from upstream pretrained formats."""

import copy
import hashlib
import io
import os
from pathlib import Path
import random
import platform
import tempfile

import numpy as np
import torch

from .selection import validate_identity


def capture_rng():
    numpy_state = np.random.get_state()
    return {
        "python": random.getstate(),
        "numpy": [numpy_state[0], numpy_state[1].tolist(), int(numpy_state[2]),
                  int(numpy_state[3]), float(numpy_state[4])],
        "torch_cpu": torch.get_rng_state().clone(),
        "torch_cuda": [state.clone() for state in torch.cuda.get_rng_state_all()]
                      if torch.cuda.is_available() else [],
    }


def restore_rng(state):
    random.setstate(state["python"])
    numpy_state = state["numpy"]
    np.random.set_state((numpy_state[0], np.asarray(numpy_state[1], dtype=np.uint32),
                         numpy_state[2], numpy_state[3], numpy_state[4]))
    torch.set_rng_state(state["torch_cpu"])
    cuda_states = state["torch_cuda"]
    if len(cuda_states) != (torch.cuda.device_count() if torch.cuda.is_available() else 0):
        raise ValueError("CUDA topology changed; exact RNG restoration is unavailable")
    if cuda_states:
        torch.cuda.set_rng_state_all(cuda_states)


def file_sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _runtime(model):
    return {"python": platform.python_version(), "torch": str(torch.__version__),
            "numpy": np.__version__, "cuda": torch.version.cuda,
            "devices": sorted({str(value.device) for value in
                               list(model.parameters()) + list(model.buffers())})}


def _signature(model):
    return {name: {"shape": list(value.shape), "dtype": str(value.dtype)}
            for name, value in model.state_dict().items()}


def _finite(value):
    if isinstance(value, torch.Tensor):
        if value.is_floating_point() or value.is_complex():
            if not torch.isfinite(value).all():
                raise ValueError("Non-finite checkpoint tensor")
    elif isinstance(value, dict):
        for child in value.values():
            _finite(child)
    elif isinstance(value, (tuple, list)):
        for child in value:
            _finite(child)
    elif isinstance(value, float) and not np.isfinite(value):
        raise ValueError("Non-finite checkpoint scalar")


def _optimizer_signature(model, optimizer):
    names = {id(parameter): name for name, parameter in model.named_parameters()}
    groups = []
    for group in optimizer.param_groups:
        if any(id(parameter) not in names for parameter in group["params"]):
            raise ValueError("Optimizer contains parameters outside the model")
        groups.append([names[id(parameter)] for parameter in group["params"]])
    qualified = type(optimizer).__module__ + "." + type(optimizer).__qualname__
    return {"class": qualified, "groups": groups}



def _check_optimizer_state(optimizer, state):
    if not isinstance(optimizer, torch.optim.AdamW):
        raise ValueError("This draft supports AdamW state only")
    groups = state["param_groups"]
    if len(groups) != len(optimizer.param_groups):
        raise ValueError("Optimizer group count differs")
    parameters = {}
    for saved, current in zip(groups, optimizer.param_groups):
        if len(saved["params"]) != len(current["params"]):
            raise ValueError("Optimizer parameter count differs")
        for number, parameter in zip(saved["params"], current["params"]):
            if type(number) is not int or number in parameters:
                raise ValueError("Optimizer parameter IDs must be unique integers")
            parameters[number] = (parameter, saved.get("amsgrad", False))
    if not set(state["state"]).issubset(parameters):
        raise ValueError("Unexpected optimizer state IDs")
    for number, moments in state["state"].items():
        parameter, amsgrad = parameters[number]
        expected = {"step", "exp_avg", "exp_avg_sq"}
        if amsgrad:
            expected.add("max_exp_avg_sq")
        if set(moments) != expected:
            raise ValueError("Incomplete AdamW moment state")
        step = moments["step"]
        if not isinstance(step, torch.Tensor) or step.numel() != 1 or step.item() < 0:
            raise ValueError("Invalid AdamW step")
        for name in expected - {"step"}:
            value = moments[name]
            if not isinstance(value, torch.Tensor) or value.shape != parameter.shape or value.dtype != parameter.dtype:
                raise ValueError("AdamW moment shape or dtype differs")
    _finite(state)


def save_training_state(path, model, optimizer, identity, *, completed_epoch, global_step):
    """Save after an epoch, before validation; never replace an existing file.

    Gradients and mid-epoch loader cursors are deliberately not resumable. The
    future loop must recreate deterministic crops/shuffling for the next epoch.
    """
    validate_identity(identity)
    if type(completed_epoch) is not int or completed_epoch < 1:
        raise ValueError("Save only at a completed epoch boundary")
    if type(global_step) is not int or global_step < 0:
        raise ValueError("global_step must be a nonnegative integer")
    payload = {
        "schema_version": 1, "kind": "local_epoch_training_state",
        "identity": copy.deepcopy(identity), "completed_epoch": completed_epoch,
        "global_step": global_step, "model_signature": _signature(model),
        "runtime": _runtime(model),
        "optimizer_signature": _optimizer_signature(model, optimizer),
        "model": copy.deepcopy(model.state_dict()),
        "optimizer": copy.deepcopy(optimizer.state_dict()), "rng": capture_rng(),
        "module_modes": {name: module.training for name, module in model.named_modules()},
    }
    _finite(payload)
    _check_optimizer_state(optimizer, payload["optimizer"])
    path = Path(path)
    if path.is_symlink() or path.exists():
        raise FileExistsError(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=".training-state-", dir=path.parent)
    try:
        with os.fdopen(descriptor, "wb") as handle:
            torch.save(payload, handle)
            handle.flush()
            os.fsync(handle.fileno())
        # A same-directory hard link publishes complete bytes without replacing
        # an existing checkpoint, including one created by another process.
        os.link(temporary, path)
    finally:
        Path(temporary).unlink(missing_ok=True)
    return file_sha256(path)


def load_training_state(path, model, optimizer, identity):
    """Strictly restore local state with rollback if a load operation fails."""
    validate_identity(identity)
    data = Path(path).read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    payload = torch.load(io.BytesIO(data), map_location="cpu", weights_only=True)
    if payload.get("schema_version") != 1 or payload.get("kind") != "local_epoch_training_state":
        raise ValueError("Not a local epoch-boundary training checkpoint")
    if payload["identity"] != identity or payload["model_signature"] != _signature(model):
        raise ValueError("Checkpoint experiment or model signature differs")
    if payload["runtime"] != _runtime(model):
        raise ValueError("Runtime or device changed; this stage requires the same environment")
    if payload["optimizer_signature"] != _optimizer_signature(model, optimizer):
        raise ValueError("Checkpoint optimizer type or parameter ordering differs")
    epoch, step = payload["completed_epoch"], payload["global_step"]
    if type(epoch) is not int or epoch < 1 or type(step) is not int or step < 0:
        raise ValueError("Invalid checkpoint progress")
    modules = dict(model.named_modules())
    if set(payload["module_modes"]) != set(modules) or any(
            type(flag) is not bool for flag in payload["module_modes"].values()):
        raise ValueError("Invalid module mode state")
    _finite(payload)
    _check_optimizer_state(optimizer, payload["optimizer"])
    before = (copy.deepcopy(model.state_dict()), copy.deepcopy(optimizer.state_dict()),
              capture_rng(), {name: module.training for name, module in modules.items()})
    try:
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        restore_rng(payload["rng"])
        for name, flag in payload["module_modes"].items():
            modules[name].training = flag
    except Exception:
        model.load_state_dict(before[0], strict=True)
        optimizer.load_state_dict(before[1])
        restore_rng(before[2])
        for name, flag in before[3].items():
            modules[name].training = flag
        raise
    # Gradients are not serialized and must not survive restoration.
    optimizer.zero_grad(set_to_none=True)
    return {"completed_epoch": epoch, "next_epoch": epoch + 1, "global_step": step,
            "checkpoint_sha256": digest}
