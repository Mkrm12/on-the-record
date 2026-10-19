"""Pre-flight for the GPU environment. Run from gpu/."""

import argparse
import sys
import time

EXPECTED_GPU = "3070"
LOW_VRAM_MIB = 2048  # the model needs well under 1 GB; below this something else is hogging the card


def assess(info: dict, strict: bool = False) -> tuple[list[str], list[str]]:
    """(errors, warnings) for a description of the machine. Pure, so the decisions can be tested."""
    if info["cuda_version"] is None:
        return [
            f"torch {info['torch_version']} is a CPU-only build. Install the CUDA build: in gpu/ run "
            "`uv sync --reinstall-package torch` (pyproject.toml points torch at the cu128 index)."
        ], []
    if not info["cuda_available"]:
        return [
            f"torch was built for CUDA {info['cuda_version']} but cannot use a GPU. Check `nvidia-smi` and the NVIDIA "
            "driver (591.86 is expected), and that no other process has taken the card."
        ], []
    errors, warnings = [], []
    if EXPECTED_GPU not in (info["device_name"] or ""):
        message = f"expected an RTX {EXPECTED_GPU}, found {info['device_name']!r}"
        (errors if strict else warnings).append(message)
    if info["free_mib"] < LOW_VRAM_MIB:
        warnings.append(f"only {info['free_mib']} MiB of VRAM is free (of {info['total_mib']} MiB): close other GPU programs")
    if info.get("on_battery"):
        warnings.append(
            "running on battery: a laptop GPU is power-capped there (measured about 4x slower for the reranker), so timings "
            "mislead and a bulk run takes minutes instead of seconds. Plug in for real work."
        )
    return errors, warnings


def battery_from_ac_status(ac_line_status: int) -> bool | None:
    """True on battery, False on mains, None when Windows does not know (255) or says something unexpected."""
    return {0: True, 1: False}.get(ac_line_status)


def on_battery() -> bool | None:
    """Whether this Windows laptop is running on battery. None on other systems or if it cannot tell."""
    if sys.platform != "win32":
        return None
    import ctypes

    class PowerStatus(ctypes.Structure):  # SYSTEM_POWER_STATUS
        _fields_ = [
            ("ACLineStatus", ctypes.c_ubyte),
            ("BatteryFlag", ctypes.c_ubyte),
            ("BatteryLifePercent", ctypes.c_ubyte),
            ("SystemStatusFlag", ctypes.c_ubyte),
            ("BatteryLifeTime", ctypes.c_ulong),
            ("BatteryFullLifeTime", ctypes.c_ulong),
        ]

    status = PowerStatus()
    if not ctypes.windll.kernel32.GetSystemPowerStatus(ctypes.byref(status)):
        return None
    return battery_from_ac_status(status.ACLineStatus)


def inspect_environment() -> dict:
    import torch

    info = {
        "torch_version": torch.__version__,
        "cuda_version": torch.version.cuda,
        "cuda_available": torch.cuda.is_available(),
        "device_name": None,
        "compute_capability": None,
        "total_mib": 0,
        "free_mib": 0,
        "on_battery": on_battery(),
    }
    if info["cuda_available"]:
        free, total = torch.cuda.mem_get_info(0)
        info.update(
            device_name=torch.cuda.get_device_name(0),
            compute_capability=torch.cuda.get_device_capability(0),
            total_mib=total // 2**20,
            free_mib=free // 2**20,
        )
    return info


def half_precision_selftest() -> tuple[bool, float]:
    """A small fp16 matrix multiply on the GPU: proves kernels launch and results are finite."""
    import torch

    a = torch.randn(1024, 384, device="cuda:0", dtype=torch.float16)
    b = torch.randn(384, 1024, device="cuda:0", dtype=torch.float16)
    torch.cuda.synchronize()
    start = time.perf_counter()
    result = a @ b
    torch.cuda.synchronize()
    return bool(torch.isfinite(result).all()), (time.perf_counter() - start) * 1000


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--strict", action="store_true", help="treat a GPU other than an RTX 3070 as an error")
    args = parser.parse_args(argv)

    info = inspect_environment()
    print(f"torch          {info['torch_version']} (CUDA build {info['cuda_version']})")
    print(f"cuda available {info['cuda_available']}")
    if info["cuda_available"]:
        print(f"device         {info['device_name']} (compute capability {info['compute_capability'][0]}.{info['compute_capability'][1]})")
        print(f"vram           {info['free_mib']:,} MiB free of {info['total_mib']:,} MiB")
    errors, warnings = assess(info, strict=args.strict)
    if info["cuda_available"] and not errors:
        finite, ms = half_precision_selftest()
        print(f"fp16 matmul    {'ok' if finite else 'NON-FINITE RESULT'} ({ms:.1f} ms)")
        if not finite:
            errors.append("an fp16 matrix multiply produced non-finite values")
    for w in warnings:
        print(f"WARNING: {w}")
    for e in errors:
        print(f"ERROR: {e}")
    print("OK" if not errors else "FAILED")
    return 1 if errors else 0


if __name__ == "__main__":
    sys.exit(main())
