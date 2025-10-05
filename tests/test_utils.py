import os
import sys
from pathlib import Path

import numpy as np
import pytest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from threadx.utils.xp import (
    ascupy,
    asnumpy,
    benchmark_operation,
    configure_backend,
    get_backend_name,
    get_xp,
    gpu_available,
    is_gpu_backend,
    memory_pool_info,
    refresh_cupy_cache,
    to_device,
    to_host,
)


class FakeCuPy:
    class ndarray(np.ndarray):
        pass

    def __init__(self):
        class Runtime:
            @staticmethod
            def getDeviceCount():
                return 1

        class Stream:
            @staticmethod
            def synchronize():
                pass

        class Event:
            def __init__(self):
                self._recorded = False

            def record(self, stream=None):
                self._recorded = True

            def synchronize(self):
                pass

        class CUDA:
            runtime = Runtime()

            @staticmethod
            def Event():
                return Event()

            @staticmethod
            def get_current_stream():
                return Stream()

            @staticmethod
            def get_elapsed_time(start, end):
                return 1.0

            @staticmethod
            def get_current_device():
                class Device:
                    id = 0

                return Device()

            @staticmethod
            def get_default_memory_pool():
                class Pool:
                    def used_bytes(self):
                        return 0

                    def total_bytes(self):
                        return 0

                    def free_all_blocks(self):
                        pass

                return Pool()

            @staticmethod
            def get_default_pinned_memory_pool():
                class Pool:
                    def used_bytes(self):
                        return 0

                    def total_bytes(self):
                        return 0

                    def free_all_blocks(self):
                        pass

                return Pool()

        self.cuda = CUDA()

    @staticmethod
    def asarray(obj, dtype=None):
        return np.asarray(obj, dtype=dtype)

    @staticmethod
    def asnumpy(obj):
        return np.asarray(obj)


@pytest.fixture(autouse=True)
def reset_backend():
    refresh_cupy_cache()
    configure_backend("auto")
    yield
    refresh_cupy_cache()
    configure_backend("auto")


def test_default_backend_numpy():
    backend = get_xp(prefer_gpu=False)
    assert backend is np
    assert get_backend_name() == "numpy"
    assert not is_gpu_backend()


def test_gpu_available_without_cupy():
    assert gpu_available() is False


def test_to_device_and_to_host_roundtrip():
    arr = np.arange(5, dtype=np.float32)
    device = to_device(arr)
    assert isinstance(device, np.ndarray)
    host = to_host(device)
    np.testing.assert_array_equal(host, arr)


def test_asnumpy_alias():
    arr = np.array([1, 2, 3])
    assert isinstance(asnumpy(arr), np.ndarray)


def test_ascupy_requires_cupy():
    arr = np.array([1, 2, 3])
    with pytest.raises(RuntimeError):
        ascupy(arr)


def test_gpu_path_with_fake_cupy():
    fake = FakeCuPy()
    import threadx.utils.xp as xp_module

    with patch.object(xp_module, "_get_device_manager", return_value=type("_DM", (), {"is_available": lambda self: True})()):
        with patch.object(xp_module.importlib, "import_module", return_value=fake):
            configure_backend("cupy")
            backend = get_xp(prefer_gpu=True)
            assert backend in {fake, np}
            info = memory_pool_info()
            assert "backend" in info
            result, elapsed = benchmark_operation(lambda: fake.asarray([1, 2, 3]))
            assert isinstance(result, np.ndarray)
            assert elapsed >= 0


def test_benchmark_operation_cpu():
    result, elapsed = benchmark_operation(lambda: np.arange(10))
    assert isinstance(result, np.ndarray)
    assert elapsed >= 0
