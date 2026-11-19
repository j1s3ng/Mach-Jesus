"""Stable tensor identity independent of a checkpoint serialization container."""
import hashlib
import json


def tensor_state_sha256(state):
    """Bind sorted tensor names, dtypes, shapes and exact uncompressed bytes."""
    import torch
    digest = hashlib.sha256()
    for name, tensor in sorted(state.items()):
        if not isinstance(tensor, torch.Tensor):
            raise ValueError("Model state must contain only tensors")
        value = tensor.detach().cpu().contiguous()
        header = json.dumps([name, str(value.dtype), list(value.shape)], separators=(",", ":")).encode()
        digest.update(len(header).to_bytes(8, "big"));digest.update(header)
        data = value.reshape(-1).view(torch.uint8).numpy().tobytes()
        digest.update(len(data).to_bytes(8, "big"));digest.update(data)
    return digest.hexdigest()
