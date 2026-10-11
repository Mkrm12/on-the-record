"""check_cuda.py: the decision logic, separate from probing the machine."""

import pytest

from check_cuda import LOW_VRAM_MIB, assess, battery_from_ac_status, inspect_environment, on_battery

GOOD = {
    "torch_version": "2.11.0+cu128",
    "cuda_version": "12.8",
    "cuda_available": True,
    "device_name": "NVIDIA GeForce RTX 3070 Laptop GPU",
    "compute_capability": (8, 6),
    "total_mib": 8192,
    "free_mib": 7400,
}


def info(**changes):
    return {**GOOD, **changes}


def test_the_expected_machine_has_no_problems():
    assert assess(GOOD) == ([], [])


def test_a_cpu_only_torch_build_is_an_error_with_the_fix():
    errors, _ = assess(info(torch_version="2.11.0+cpu", cuda_version=None, cuda_available=False, device_name=None))
    assert len(errors) == 1 and "CPU-only" in errors[0] and "cu128" in errors[0]


def test_cuda_build_but_no_usable_device_points_at_the_driver():
    errors, _ = assess(info(cuda_available=False, device_name=None))
    assert len(errors) == 1 and "driver" in errors[0].lower()


def test_a_different_gpu_is_a_warning_not_an_error_by_default():
    errors, warnings = assess(info(device_name="NVIDIA GeForce RTX 4090"))
    assert errors == [] and any("3070" in w for w in warnings)


def test_strict_mode_turns_a_different_gpu_into_an_error():
    errors, warnings = assess(info(device_name="NVIDIA GeForce RTX 4090"), strict=True)
    assert any("3070" in e for e in errors)


@pytest.mark.parametrize("name", ["NVIDIA GeForce RTX 3070", "NVIDIA GeForce RTX 3070 Laptop GPU", "NVIDIA GeForce RTX 3070 Ti"])
def test_any_3070_variant_is_accepted(name):
    assert assess(info(device_name=name), strict=True)[0] == []


def test_low_free_vram_is_a_warning():
    errors, warnings = assess(info(free_mib=LOW_VRAM_MIB - 1))
    assert errors == [] and any("free" in w.lower() for w in warnings)


def test_enough_free_vram_is_fine_at_the_boundary():
    assert assess(info(free_mib=LOW_VRAM_MIB)) == ([], [])


# On battery this laptop's GPU is power-capped (measured: 630 MHz at a flat 30 W against a 2,100 MHz maximum), which
# made the reranker about 4x slower than on mains.


def test_running_on_battery_is_a_warning_that_says_why_it_matters():
    errors, warnings = assess(info(on_battery=True))
    assert errors == [] and len(warnings) == 1
    assert "battery" in warnings[0] and "slower" in warnings[0] and "plug in" in warnings[0].lower()


@pytest.mark.parametrize("on_battery", [False, None])
def test_mains_power_or_an_unknown_power_source_is_silent(on_battery):
    assert assess(info(on_battery=on_battery)) == ([], [])


def test_a_machine_description_without_power_information_is_still_accepted():
    assert "on_battery" not in GOOD and assess(GOOD) == ([], [])


@pytest.mark.parametrize("code,expected", [(0, True), (1, False), (255, None), (2, None)])
def test_windows_ac_line_status_codes(code, expected):
    assert battery_from_ac_status(code) is expected


def test_the_live_power_probe_answers_true_false_or_none_and_never_raises():
    assert on_battery() in (True, False, None)


def test_inspect_environment_includes_the_power_source():
    pytest.importorskip("torch")
    assert "on_battery" in inspect_environment()
