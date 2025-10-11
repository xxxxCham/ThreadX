"""
ThreadX Environment Check Tool - Phase 10
========================================

Diagnostic complet de l'environnement ThreadX avec micro-benchmarks
et génération de rapport JSON + recommandations.

Features:
- Détection venv, versions Python et packages
- Probe CPU (cores/logicals), RAM, disques
- GPU detection (RTX 5090/2060) et NCCL status
- Micro-benchmarks: NumPy, Pandas, Parquet, CuPy (si dispo)
- Rapport console formaté + JSON structuré
- Mode --strict avec exit codes appropriés

Usage:
    python tools/check_env.py
    python tools/check_env.py --json env_report.json
    python tools/check_env.py --strict --json report.json
"""

import argparse
import json
import logging
import os
import platform
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import psutil

# ThreadX imports with fallbacks
try:
    from src.threadx.config.settings import load_settings
    from src.threadx.utils.log import get_logger
    THREADX_AVAILABLE = True
except ImportError:
    def get_logger(name: str) -> logging.Logger:
        return logging.getLogger(name)

    def load_settings():
        return None

    THREADX_AVAILABLE = False

logger = get_logger(__name__)

@dataclass
class SystemSpecs:
    """System specifications."""
    python_version: str
    python_executable: str
    venv_active: bool
    venv_path: Optional[str]
    platform: str
    architecture: str
    cpu_count: int
    cpu_count_logical: int
    memory_total_gb: float
    memory_available_gb: float
    disk_free_gb: Dict[str, float]

@dataclass
class PackageInfo:
    """Package version information."""
    name: str
    version: Optional[str]
    installed: bool
    required_version: Optional[str] = None
    status: str = "ok"  # ok, missing, outdated

@dataclass
class GPUInfo:
    """GPU information."""
    detected: bool
    device_count: int
    devices: List[Dict[str, Any]]
    nccl_available: bool
    cuda_version: Optional[str]
    driver_version: Optional[str]

@dataclass
class BenchmarkResult:
    """Benchmark result."""
    name: str
    duration_sec: float
    operations_per_sec: float
    throughput_mb_per_sec: Optional[float] = None
    details: Optional[Dict[str, Any]] = None

@dataclass
class EnvironmentReport:
    """Complete environment report."""
    timestamp: str
    system: SystemSpecs
    packages: List[PackageInfo]
    gpu: GPUInfo
    benchmarks: List[BenchmarkResult]
    recommendations: List[str]
    warnings: List[str]
    errors: List[str]

def get_python_info() -> Tuple[str, str, bool, Optional[str]]:
    """Get Python version and virtual environment info."""
    version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    executable = sys.executable

    # Check if in virtual environment
    venv_active = hasattr(sys, 'real_prefix') or (
        hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix
    )

    venv_path = None
    if venv_active:
        venv_path = sys.prefix

    return version, executable, venv_active, venv_path

def get_system_specs() -> SystemSpecs:
    """Gather system specifications."""
    logger.debug("Gathering system specifications")

    py_version, py_executable, venv_active, venv_path = get_python_info()

    # Memory info
    memory = psutil.virtual_memory()
    memory_total_gb = memory.total / (1024**3)
    memory_available_gb = memory.available / (1024**3)

    # Disk space for relevant paths
    disk_free = {}
    paths_to_check = [".", "./data", "./logs", "./cache"]

    for path_str in paths_to_check:
        try:
            path = Path(path_str).resolve()
            if path.exists():
                usage = shutil.disk_usage(path)
                disk_free[path_str] = usage.free / (1024**3)
        except Exception as e:
            logger.debug(f"Could not check disk space for {path_str}: {e}")
            disk_free[path_str] = 0.0

    return SystemSpecs(
        python_version=py_version,
        python_executable=py_executable,
        venv_active=venv_active,
        venv_path=venv_path,
        platform=platform.platform(),
        architecture=platform.architecture()[0],
        cpu_count=psutil.cpu_count(logical=False) or 1,
        cpu_count_logical=psutil.cpu_count(logical=True) or 1,
        memory_total_gb=memory_total_gb,
        memory_available_gb=memory_available_gb,
        disk_free_gb=disk_free
    )

def check_package_versions() -> List[PackageInfo]:
    """Check versions of required packages."""
    logger.debug("Checking package versions")

    # Critical packages for ThreadX
    required_packages = {
        'numpy': '>=1.21.0',
        'pandas': '>=1.5.0',
        'pyarrow': '>=8.0.0',
        'psutil': '>=5.8.0',
        'toml': '>=0.10.0',
        'plotly': '>=5.0.0',
        'streamlit': '>=1.20.0',
        'tkinter': None,  # Built-in, check differently
    }

    # Optional packages
    optional_packages = {
        'cupy': None,
        'numba': '>=0.56.0',
        'joblib': '>=1.1.0',
    }

    packages = []

    # Check required packages
    for pkg_name, min_version in required_packages.items():
        if pkg_name == 'tkinter':
            # Special case for tkinter (built-in)
            try:
                import tkinter
                packages.append(PackageInfo(
                    name=pkg_name,
                    version="built-in",
                    installed=True,
                    required_version=None,
                    status="ok"
                ))
            except ImportError:
                packages.append(PackageInfo(
                    name=pkg_name,
                    version=None,
                    installed=False,
                    required_version=None,
                    status="missing"
                ))
        else:
            try:
                import importlib
                module = importlib.import_module(pkg_name)
                version = getattr(module, '__version__', 'unknown')

                # Check version requirement
                status = "ok"
                if min_version and version != 'unknown':
                    # Simple version comparison (could be more sophisticated)
                    if version < min_version.replace('>=', ''):
                        status = "outdated"

                packages.append(PackageInfo(
                    name=pkg_name,
                    version=version,
                    installed=True,
                    required_version=min_version,
                    status=status
                ))
            except ImportError:
                packages.append(PackageInfo(
                    name=pkg_name,
                    version=None,
                    installed=False,
                    required_version=min_version,
                    status="missing"
                ))

    # Check optional packages
    for pkg_name, min_version in optional_packages.items():
        try:
            import importlib
            module = importlib.import_module(pkg_name)
            version = getattr(module, '__version__', 'unknown')
            packages.append(PackageInfo(
                name=pkg_name,
                version=version,
                installed=True,
                required_version=min_version,
                status="ok"
            ))
        except ImportError:
            packages.append(PackageInfo(
                name=pkg_name,
                version=None,
                installed=False,
                required_version=min_version,
                status="optional"
            ))

    return packages

def detect_gpu_info() -> GPUInfo:
    """Detect GPU information and NCCL availability."""
    logger.debug("Detecting GPU configuration")

    gpu_info = GPUInfo(
        detected=False,
        device_count=0,
        devices=[],
        nccl_available=False,
        cuda_version=None,
        driver_version=None
    )

    # Try CuPy for GPU detection
    try:
        import cupy as cp

        gpu_info.detected = True
        gpu_info.device_count = cp.cuda.runtime.getDeviceCount()

        # Get device details
        for i in range(gpu_info.device_count):
            with cp.cuda.Device(i):
                device_props = cp.cuda.runtime.getDeviceProperties(i)

                # Try to identify RTX 5090/2060 from name
                device_name = device_props['name'].decode('utf-8')
                is_target_gpu = any(model in device_name for model in ['5090', '2060', 'RTX'])

                gpu_info.devices.append({
                    'id': i,
                    'name': device_name,
                    'memory_gb': device_props['totalGlobalMem'] / (1024**3),
                    'compute_capability': f"{device_props['major']}.{device_props['minor']}",
                    'is_target_gpu': is_target_gpu
                })

        # Check CUDA version
        gpu_info.cuda_version = '.'.join(map(str, cp.cuda.runtime.runtimeGetVersion()))

        # Check NCCL (always mark as available in CuPy context, actual usage is no-op)
        try:
            # NCCL is available through CuPy but we use no-op for ThreadX
            gpu_info.nccl_available = True
        except Exception:
            gpu_info.nccl_available = False

    except ImportError:
        logger.debug("CuPy not available - no GPU acceleration")
    except Exception as e:
        logger.debug(f"GPU detection failed: {e}")

    return gpu_info

def run_numpy_benchmark() -> BenchmarkResult:
    """Run NumPy vector operations benchmark."""
    logger.debug("Running NumPy benchmark")

    # Vector operations
    size = 1000000
    iterations = 10

    start_time = time.time()

    for _ in range(iterations):
        a = np.random.random(size)
        b = np.random.random(size)

        # Various operations
        c = a + b
        d = np.sin(a)
        e = np.dot(a, b)
        f = np.fft.fft(a[:1000])  # Smaller size for FFT

    duration = time.time() - start_time
    ops_per_sec = (iterations * 4) / duration  # 4 operations per iteration

    return BenchmarkResult(
        name="NumPy Vector Operations",
        duration_sec=duration,
        operations_per_sec=ops_per_sec,
        details={
            'vector_size': size,
            'iterations': iterations,
            'operations': ['add', 'sin', 'dot', 'fft']
        }
    )

def run_pandas_benchmark() -> BenchmarkResult:
    """Run Pandas groupby/resample benchmark."""
    logger.debug("Running Pandas benchmark")

    # Create sample timeseries data
    dates = pd.date_range('2024-01-01', periods=100000, freq='1min')
    df = pd.DataFrame({
        'timestamp': dates,
        'value': np.random.random(len(dates)),
        'category': np.random.choice(['A', 'B', 'C'], len(dates))
    }).set_index('timestamp')

    iterations = 5
    start_time = time.time()

    for _ in range(iterations):
        # Groupby operations
        grouped = df.groupby('category').agg({
            'value': ['mean', 'sum', 'std']
        })

        # Resampling
        resampled = df.resample('1h').agg({
            'value': 'mean'
        })

    duration = time.time() - start_time
    ops_per_sec = (iterations * 2) / duration  # 2 operations per iteration

    return BenchmarkResult(
        name="Pandas GroupBy/Resample",
        duration_sec=duration,
        operations_per_sec=ops_per_sec,
        details={
            'rows': len(df),
            'iterations': iterations,
            'operations': ['groupby', 'resample']
        }
    )

def run_parquet_benchmark() -> BenchmarkResult:
    """Run Parquet read/write benchmark."""
    logger.debug("Running Parquet I/O benchmark")

    # Create sample data
    size = 50000
    df = pd.DataFrame({
        'timestamp': pd.date_range('2024-01-01', periods=size, freq='1min'),
        'open': np.random.random(size) * 100,
        'high': np.random.random(size) * 100 + 100,
        'low': np.random.random(size) * 100,
        'close': np.random.random(size) * 100 + 50,
        'volume': np.random.random(size) * 1000000
    }).set_index('timestamp')

    # Temporary file
    temp_file = Path("temp_benchmark.parquet")

    try:
        start_time = time.time()

        # Write
        df.to_parquet(temp_file, compression='snappy')

        # Read
        df_read = pd.read_parquet(temp_file)

        duration = time.time() - start_time

        # Calculate throughput
        file_size_mb = temp_file.stat().st_size / (1024**2)
        throughput_mb_per_sec = (file_size_mb * 2) / duration  # Read + Write

        return BenchmarkResult(
            name="Parquet I/O",
            duration_sec=duration,
            operations_per_sec=2 / duration,  # read + write
            throughput_mb_per_sec=throughput_mb_per_sec,
            details={
                'rows': size,
                'file_size_mb': file_size_mb,
                'compression': 'snappy'
            }
        )

    finally:
        # Cleanup
        if temp_file.exists():
            temp_file.unlink()

def run_cupy_benchmark() -> Optional[BenchmarkResult]:
    """Run CuPy benchmark if available."""
    try:
        import cupy as cp
        logger.debug("Running CuPy benchmark")

        size = 1000000
        iterations = 5

        start_time = time.time()

        for _ in range(iterations):
            # GPU operations
            a_gpu = cp.random.random(size).astype(cp.float32)
            b_gpu = cp.random.random(size).astype(cp.float32)

            # Various operations
            c_gpu = a_gpu + b_gpu
            d_gpu = cp.sin(a_gpu)
            e_gpu = cp.sum(a_gpu)

            # Synchronize to ensure completion
            cp.cuda.Stream.null.synchronize()

        duration = time.time() - start_time
        ops_per_sec = (iterations * 3) / duration

        return BenchmarkResult(
            name="CuPy GPU Operations",
            duration_sec=duration,
            operations_per_sec=ops_per_sec,
            details={
                'vector_size': size,
                'iterations': iterations,
                'operations': ['add', 'sin', 'sum'],
                'device': cp.cuda.Device().id
            }
        )

    except ImportError:
        logger.debug("CuPy not available for benchmark")
        return None
    except Exception as e:
        logger.debug(f"CuPy benchmark failed: {e}")
        return None

def run_micro_benchmarks(*, with_gpu: Optional[bool] = None) -> List[BenchmarkResult]:
    """Run all micro-benchmarks."""
    logger.info("Running micro-benchmarks")

    benchmarks = []

    # Always run CPU benchmarks
    try:
        benchmarks.append(run_numpy_benchmark())
    except Exception as e:
        logger.warning(f"NumPy benchmark failed: {e}")

    try:
        benchmarks.append(run_pandas_benchmark())
    except Exception as e:
        logger.warning(f"Pandas benchmark failed: {e}")

    try:
        benchmarks.append(run_parquet_benchmark())
    except Exception as e:
        logger.warning(f"Parquet benchmark failed: {e}")

    # GPU benchmark if available and requested
    if with_gpu is not False:
        cupy_result = run_cupy_benchmark()
        if cupy_result:
            benchmarks.append(cupy_result)

    logger.info(f"Completed {len(benchmarks)} benchmarks")
    return benchmarks

def generate_recommendations(
    system: SystemSpecs,
    packages: List[PackageInfo],
    gpu: GPUInfo,
    benchmarks: List[BenchmarkResult]
) -> List[str]:
    """Generate actionable recommendations."""
    recommendations = []

    # Python version check
    if system.python_version < "3.10":
        recommendations.append(
            f"Upgrade Python to 3.10+ (current: {system.python_version})"
        )

    # Virtual environment check
    if not system.venv_active:
        recommendations.append(
            "Activate virtual environment for better dependency isolation"
        )

    # Memory recommendations
    if system.memory_available_gb < 4.0:
        recommendations.append(
            f"Low available memory ({system.memory_available_gb:.1f}GB). "
            "Close other applications or add more RAM"
        )

    # Disk space warnings
    for path, free_gb in system.disk_free_gb.items():
        if free_gb < 1.0:
            recommendations.append(
                f"Low disk space for {path}: {free_gb:.1f}GB free"
            )

    # Package recommendations
    missing_critical = [p for p in packages if not p.installed and p.status == "missing"]
    if missing_critical:
        pkg_names = [p.name for p in missing_critical]
        recommendations.append(
            f"Install missing critical packages: {', '.join(pkg_names)}"
        )

    outdated = [p for p in packages if p.status == "outdated"]
    if outdated:
        pkg_names = [f"{p.name} (current: {p.version})" for p in outdated]
        recommendations.append(
            f"Update outdated packages: {', '.join(pkg_names)}"
        )

    # GPU recommendations
    if gpu.detected:
        target_gpus = [d for d in gpu.devices if d['is_target_gpu']]
        if target_gpus:
            recommendations.append(
                f"GPU acceleration available: {len(target_gpus)} compatible device(s) detected"
            )
        else:
            recommendations.append(
                "GPU detected but may not be optimal for ThreadX (target: RTX 5090/2060)"
            )
    else:
        recommendations.append(
            "No GPU detected. Install CuPy for GPU acceleration if compatible hardware available"
        )

    # Performance recommendations based on benchmarks
    numpy_bench = next((b for b in benchmarks if "NumPy" in b.name), None)
    if numpy_bench and numpy_bench.operations_per_sec < 10:
        recommendations.append(
            f"NumPy performance low ({numpy_bench.operations_per_sec:.1f} ops/sec). "
            "Check BLAS configuration or CPU thermal throttling"
        )

    # Threading recommendations
    if system.cpu_count_logical > system.cpu_count:
        recommendations.append(
            f"Hyperthreading detected ({system.cpu_count_logical} logical cores). "
            "Consider setting OMP_NUM_THREADS=1 for NumPy operations to avoid oversubscription"
        )

    # Batch size recommendations
    if system.memory_total_gb >= 16:
        recommendations.append(
            "High memory system: use larger batch sizes for better performance"
        )
    elif system.memory_total_gb < 8:
        recommendations.append(
            "Limited memory: use smaller batch sizes to avoid OOM errors"
        )

    return recommendations

def print_report(system: SystemSpecs, packages: List[PackageInfo], gpu: GPUInfo, benchmarks: List[BenchmarkResult], recommendations: List[str]) -> None:
    """Print formatted report to console."""

    print("\n" + "=" * 80)
    print("ThreadX Environment Report")
    print("=" * 80)

    # System Info
    print(f"\n🖥️  SYSTEM SPECIFICATIONS")
    print(f"   Python:      {system.python_version} ({'virtual env' if system.venv_active else 'system'})")
    print(f"   Platform:    {system.platform}")
    print(f"   CPU Cores:   {system.cpu_count} physical, {system.cpu_count_logical} logical")
    print(f"   Memory:      {system.memory_available_gb:.1f}GB available / {system.memory_total_gb:.1f}GB total")

    # Disk Space
    print(f"   Disk Space:")
    for path, free_gb in system.disk_free_gb.items():
        status = "⚠️" if free_gb < 1.0 else "✅"
        print(f"     {path:<12} {free_gb:>8.1f}GB {status}")

    # Package Status
    print(f"\n📦 PACKAGE STATUS")
    for pkg in sorted(packages, key=lambda x: (x.status != "ok", x.name)):
        if pkg.installed:
            status_icon = "✅" if pkg.status == "ok" else "⚠️" if pkg.status == "outdated" else "ℹ️"
            version_info = f"v{pkg.version}"
        else:
            status_icon = "❌" if pkg.status == "missing" else "⚪"
            version_info = "not installed"

        print(f"   {pkg.name:<12} {version_info:<15} {status_icon}")

    # GPU Info
    print(f"\n🎮 GPU CONFIGURATION")
    if gpu.detected:
        print(f"   Detected:    {gpu.device_count} device(s)")
        if gpu.cuda_version:
            print(f"   CUDA:        {gpu.cuda_version}")
        print(f"   NCCL:        {'Available (no-op)' if gpu.nccl_available else 'Not available'}")

        for device in gpu.devices:
            target_marker = "🎯" if device['is_target_gpu'] else "  "
            print(f"   {target_marker} Device {device['id']}: {device['name']} ({device['memory_gb']:.1f}GB)")
    else:
        print(f"   Status:      No GPU detected")
        print(f"   Note:        Install CuPy for GPU acceleration")

    # Benchmarks
    print(f"\n⚡ PERFORMANCE BENCHMARKS")
    for bench in benchmarks:
        print(f"   {bench.name:<25} {bench.operations_per_sec:>8.1f} ops/sec ({bench.duration_sec:.3f}s)")
        if bench.throughput_mb_per_sec:
            print(f"   {'':>25} {bench.throughput_mb_per_sec:>8.1f} MB/sec")

    # Recommendations
    if recommendations:
        print(f"\n💡 RECOMMENDATIONS")
        for i, rec in enumerate(recommendations, 1):
            print(f"   {i}. {rec}")

    print("\n" + "=" * 80)

def write_json_report(path: Path, report: EnvironmentReport) -> Path:
    """Write JSON report to file."""
    logger.info(f"Writing JSON report to {path}")

    with open(path, 'w') as f:
        json.dump(asdict(report), f, indent=2, default=str)

    return path

def gather_specs() -> SystemSpecs:
    """Gather system specifications."""
    return get_system_specs()

def main(argv: Optional[List[str]] = None) -> int:
    """
    Main check_env entry point.

    Args:
        argv: Command line arguments

    Returns:
        Exit code (0 for success, non-zero for critical issues in --strict mode)
    """
    parser = argparse.ArgumentParser(
        description="Check ThreadX environment and run benchmarks",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                                    # Basic environment check
  %(prog)s --json env_report.json            # Save JSON report
  %(prog)s --strict                          # Exit non-zero if critical issues
  %(prog)s --no-gpu --json report.json       # Skip GPU benchmarks
        """
    )

    parser.add_argument(
        '--json',
        type=Path,
        help='Write JSON report to file'
    )
    parser.add_argument(
        '--strict',
        action='store_true',
        help='Exit with non-zero code if critical requirements missing'
    )
    parser.add_argument(
        '--no-gpu',
        action='store_true',
        help='Skip GPU detection and benchmarks'
    )
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Enable verbose logging'
    )

    args = parser.parse_args(argv)

    # Configure logging
    log_level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    logger.info("Starting ThreadX environment check")

    # Gather system info
    system_specs = get_system_specs()
    packages = check_package_versions()
    gpu_info = detect_gpu_info() if not args.no_gpu else GPUInfo(
        detected=False, device_count=0, devices=[], nccl_available=False,
        cuda_version=None, driver_version=None
    )

    # Run benchmarks
    benchmarks = run_micro_benchmarks(with_gpu=not args.no_gpu)

    # Generate recommendations and check for critical issues
    recommendations = generate_recommendations(system_specs, packages, gpu_info, benchmarks)

    warnings = []
    errors = []

    # Check for critical issues (for --strict mode)
    critical_missing = [p for p in packages if not p.installed and p.status == "missing" and p.name in ['numpy', 'pandas', 'pyarrow']]
    if critical_missing:
        error_msg = f"Critical packages missing: {[p.name for p in critical_missing]}"
        errors.append(error_msg)
        logger.error(error_msg)

    if system_specs.python_version < "3.10":
        error_msg = f"Python version too old: {system_specs.python_version} (requires 3.10+)"
        errors.append(error_msg)
        logger.error(error_msg)

    if system_specs.memory_available_gb < 2.0:
        error_msg = f"Insufficient memory: {system_specs.memory_available_gb:.1f}GB available (requires 2GB+)"
        errors.append(error_msg)
        logger.error(error_msg)

    # Warnings for non-critical issues
    if not system_specs.venv_active:
        warnings.append("Virtual environment not active")

    outdated_packages = [p for p in packages if p.status == "outdated"]
    if outdated_packages:
        warnings.append(f"Outdated packages: {[p.name for p in outdated_packages]}")

    # Create report
    report = EnvironmentReport(
        timestamp=datetime.now([System.DateTime]::Utc).isoformat(),
        system=system_specs,
        packages=packages,
        gpu=gpu_info,
        benchmarks=benchmarks,
        recommendations=recommendations,
        warnings=warnings,
        errors=errors
    )

    # Print console report
    print_report(system_specs, packages, gpu_info, benchmarks, recommendations)

    # Write JSON report if requested
    if args.json:
        try:
            write_json_report(args.json, report)
        except Exception as e:
            logger.error(f"Failed to write JSON report: {e}")
            return 1

    # Exit with appropriate code
    if args.strict and errors:
        logger.error(f"Critical issues found in --strict mode: {len(errors)} errors")
        return 1

    if errors:
        logger.warning(f"Found {len(errors)} errors (use --strict for non-zero exit)")

    logger.info("Environment check completed successfully")
    return 0

if __name__ == "__main__":
    sys.exit(main())

