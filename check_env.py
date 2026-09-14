# Copyright (c) 2026 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Check the local PaddlePaddle / GPU environment used by the AP tests.

Usage:
    python check_env.py
"""

import subprocess
import sys
from datetime import datetime, timezone

import paddle


def _print_title(title):
    print()
    print(title)


def _print_kv(key, value):
    print(f"    {key:<28}: {value}")


def _safe(fn, default="N/A"):
    """Call fn() and swallow errors so one missing API cannot abort the check."""
    try:
        value = fn()
    except Exception as e:  # noqa: BLE001
        return f"{default} ({type(e).__name__}: {e})"
    return default if value is None or value == "" else value


def check_paddle_version():
    _print_title("[1] PaddlePaddle Version")
    version = paddle.version
    _print_kv("full_version", _safe(lambda: version.full_version))
    _print_kv("commit", _safe(lambda: version.commit))
    _print_kv("python", sys.version.split()[0])
    _print_kv("cuda version (build)", _safe(version.cuda))
    _print_kv("cudnn version (build)", _safe(version.cudnn))


def check_compile_env():
    _print_title("[2] Compile Environment")
    with_cinn = _safe(paddle.is_compiled_with_cinn)
    with_cuda = _safe(paddle.is_compiled_with_cuda)
    _print_kv("is_compiled_with_cinn", with_cinn)
    _print_kv("is_compiled_with_cuda", with_cuda)
    return with_cinn is True and with_cuda is True


def _nvidia_driver_version():
    out = subprocess.run(
        ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"],
        capture_output=True,
        text=True,
        timeout=20,
        check=False,
    )
    return out.stdout.strip().splitlines()[0]


def check_gpu_env():
    _print_title("[3] GPU Runtime Environment")
    if not paddle.is_compiled_with_cuda():
        print("    Paddle is NOT compiled with CUDA, skip GPU checks.")
        return False

    device_count = _safe(paddle.device.cuda.device_count, default=0)
    _print_kv("device_count", device_count)
    _print_kv("driver version", _safe(_nvidia_driver_version))

    if not isinstance(device_count, int) or device_count == 0:
        print("    No visible CUDA device.")
        return False

    # All devices are assumed homogeneous, so summarize device 0 only.
    _print_kv("device name", _safe(lambda: paddle.device.cuda.get_device_name(0)))
    _print_kv(
        "device capability",
        _safe(lambda: "%d.%d" % paddle.device.cuda.get_device_capability(0)),
    )
    props = _safe(lambda: paddle.device.cuda.get_device_properties(0), default=None)
    if props is not None and not isinstance(props, str):
        _print_kv("device total_memory", f"{props.total_memory / 1024 ** 3:.2f} GiB")
    return True


_HEADER_WIDTH = 75


def _print_header():
    title = "AP Full Test Environment Self-Check"
    local_now = datetime.now().astimezone()
    utc_now = local_now.astimezone(timezone.utc)
    print("=" * _HEADER_WIDTH)
    print(title.center(_HEADER_WIDTH))
    print(
        (
            f"Checked at: {local_now.strftime('%Y-%m-%d %H:%M:%S %Z%z')}"
            f" (UTC {utc_now.strftime('%H:%M:%S')})"
        ).center(_HEADER_WIDTH)
    )
    print("=" * _HEADER_WIDTH)


def _print_conclusion(compile_ok, gpu_ok):
    _print_title("[Conclusion]")
    if compile_ok and gpu_ok:
        print("    OK: framework and GPU environment are ready for AP tests.")
        return
    if not compile_ok:
        print("    FAILED: paddle is not compiled with both CINN and CUDA.")
    if not gpu_ok:
        print("    FAILED: no usable GPU device.")


def main():
    _print_header()
    check_paddle_version()
    compile_ok = check_compile_env()
    gpu_ok = check_gpu_env()
    _print_conclusion(compile_ok, gpu_ok)
    print()


if __name__ == "__main__":
    main()
