"""
ThreadX Utils Tests - Phase 9
Comprehensive unit tests for utils module functionality.

Tests timing, caching, device-agnostic computing, and integration features.
All tests are deterministic (seed=42), headless, and require no network/env vars.
"""

import os
import sys
import time
import threading
import tempfile
import unittest
from unittest.mock import Mock, patch, MagicMock
import numpy as np

# Add src to path for imports
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

# Import ThreadX utils modules
from threadx.utils.timing import Timer, measure_throughput, track_memory, performance_context
from threadx.utils.cache import LRUCache, TTLCache, cached, generate_stable_key
from threadx.utils.xp import xp, gpu_available, to_device, to_host


class TestTiming(unittest.TestCase):
    """Test timing and performance measurement utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
    
    def test_timer_context_manager(self):
        """Test Timer as context manager."""
        with Timer() as timer:
            time.sleep(0.01)  # 10ms
            
        elapsed = timer.elapsed_sec
        self.assertGreater(elapsed, 0.005)  # At least 5ms
        self.assertLess(elapsed, 0.05)      # Less than 50ms (generous for CI)
    
    def test_timer_manual_control(self):
        """Test Timer with manual start/stop."""
        timer = Timer()
        
        # Initially no time elapsed
        self.assertEqual(timer.elapsed_sec, 0.0)
        
        timer.start()
        time.sleep(0.01)
        timer.stop()
        
        elapsed = timer.elapsed_sec
        self.assertGreater(elapsed, 0.005)
        
        # Time should remain stable after stop
        time.sleep(0.001)
        self.assertEqual(timer.elapsed_sec, elapsed)
    
    def test_timer_running_elapsed(self):
        """Test elapsed time while timer is running."""
        timer = Timer()
        timer.start()
        
        time.sleep(0.01)
        elapsed1 = timer.elapsed_sec
        
        time.sleep(0.01) 
        elapsed2 = timer.elapsed_sec
        
        # Should increase while running
        self.assertGreater(elapsed2, elapsed1)
        
        timer.stop()
    
    @patch('threadx.utils.timing.logger')
    def test_measure_throughput_decorator(self, mock_logger):
        """Test throughput measurement decorator."""
        
        @measure_throughput("test_function", unit_of_work="item")
        def process_items(items):
            time.sleep(0.01)  # Simulate work
            return [x * 2 for x in items]
        
        # Test with 100 items
        items = list(range(100))
        result = process_items(items)
        
        # Check result correctness
        self.assertEqual(len(result), 100)
        self.assertEqual(result[0], 0)
        self.assertEqual(result[50], 100)
        
        # Check logging was called
        self.assertTrue(mock_logger.info.called)
        
        # Check log message contains throughput info
        log_calls = mock_logger.info.call_args_list
        log_message = str(log_calls[-1])
        self.assertIn("items/min", log_message)
        self.assertIn("test_function", log_message)
    
    @patch('threadx.utils.timing.logger')
    @patch('threadx.utils.timing.load_settings')
    def test_throughput_warning_threshold(self, mock_settings, mock_logger):
        """Test WARNING when throughput below threshold."""
        
        # Mock settings to return low threshold
        mock_settings_obj = Mock()
        mock_settings_obj.MIN_TASKS_PER_MIN = 1000000  # Very high threshold
        mock_settings.return_value = mock_settings_obj
        
        @measure_throughput()
        def slow_function():
            time.sleep(0.1)  # Slow function
            return [1]  # Only 1 item
        
        result = slow_function()
        
        # Should have logged warning
        warning_calls = [call for call in mock_logger.warning.call_args_list 
                        if 'PERFORMANCE WARNING' in str(call)]
        self.assertTrue(len(warning_calls) > 0)
    
    @patch('threadx.utils.timing.psutil')
    @patch('threadx.utils.timing.logger')
    def test_track_memory_decorator(self, mock_logger, mock_psutil):
        """Test memory tracking decorator."""
        
        # Mock psutil
        mock_process = Mock()
        mock_memory_info = Mock()
        mock_memory_info.rss = 100 * 1024 * 1024  # 100MB
        mock_process.memory_info.return_value = mock_memory_info
        mock_psutil.Process.return_value = mock_process
        
        @track_memory("test_memory")
        def memory_function():
            return "result"
        
        result = memory_function()
        
        # Check result
        self.assertEqual(result, "result")
        
        # Check logging
        self.assertTrue(mock_logger.info.called)
        log_message = str(mock_logger.info.call_args_list[-1])
        self.assertIn("memory usage", log_message)
        self.assertIn("test_memory", log_message)
    
    @patch('threadx.utils.timing.PSUTIL_AVAILABLE', False)
    @patch('threadx.utils.timing.logger')
    def test_track_memory_fallback(self, mock_logger):
        """Test memory tracking fallback when psutil unavailable."""
        
        @track_memory()
        def test_function():
            return "result"
        
        result = test_function()
        
        # Should still work
        self.assertEqual(result, "result")
        
        # Should log unavailable message
        info_calls = mock_logger.info.call_args_list
        fallback_logged = any("psutil unavailable" in str(call) for call in info_calls)
        self.assertTrue(fallback_logged)
    
    def test_performance_context(self):
        """Test performance context manager."""
        
        with performance_context("test_context", task_count=50, unit_of_work="test") as perf:
            time.sleep(0.01)
            
        # Check metrics were populated
        self.assertGreater(perf.elapsed_sec, 0.005)
        self.assertEqual(perf.tasks_completed, 50)
        self.assertGreater(perf.tasks_per_min, 0)
        self.assertEqual(perf.function_name, "test_context")
        self.assertEqual(perf.unit_of_work, "test")


class TestCache(unittest.TestCase):
    """Test caching infrastructure."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
    
    def test_lru_cache_basic_operations(self):
        """Test basic LRU cache operations."""
        cache = LRUCache[str, int](capacity=3)
        
        # Test empty cache
        self.assertEqual(cache.size, 0)
        self.assertIsNone(cache.get("missing"))
        self.assertFalse(cache.contains("missing"))
        
        # Test set/get
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        
        self.assertEqual(cache.size, 3)
        self.assertEqual(cache.get("a"), 1)
        self.assertEqual(cache.get("b"), 2)
        self.assertEqual(cache.get("c"), 3)
        self.assertTrue(cache.contains("a"))
        
        # Test LRU eviction
        cache.set("d", 4)  # Should evict "a" (least recently used)
        
        self.assertEqual(cache.size, 3)
        self.assertIsNone(cache.get("a"))  # Evicted
        self.assertEqual(cache.get("d"), 4)  # New item
        
        # Test LRU update on access
        cache.get("b")  # Access "b" to make it most recent
        cache.set("e", 5)  # Should evict "c" (now least recent)
        
        self.assertIsNone(cache.get("c"))  # Evicted
        self.assertEqual(cache.get("b"), 2)  # Still there
        self.assertEqual(cache.get("e"), 5)  # New item
    
    def test_lru_cache_stats(self):
        """Test LRU cache statistics."""
        cache = LRUCache[str, int](capacity=2)
        
        # Initial stats
        stats = cache.stats()
        self.assertEqual(stats.hits, 0)
        self.assertEqual(stats.misses, 0)
        self.assertEqual(stats.evictions, 0)
        self.assertEqual(stats.hit_rate, 0.0)
        
        # Generate some activity
        cache.set("a", 1)
        cache.set("b", 2)
        
        cache.get("a")  # Hit
        cache.get("missing")  # Miss
        
        cache.set("c", 3)  # Eviction
        
        stats = cache.stats()
        self.assertEqual(stats.hits, 1)
        self.assertEqual(stats.misses, 1)
        self.assertEqual(stats.evictions, 1)
        self.assertEqual(stats.hit_rate, 50.0)
        self.assertEqual(stats.current_size, 2)
        self.assertEqual(stats.capacity, 2)
    
    def test_ttl_cache_basic_operations(self):
        """Test basic TTL cache operations."""
        cache = TTLCache[str, int](ttl_seconds=0.1)  # 100ms TTL
        
        # Test set/get within TTL
        cache.set("a", 1)
        self.assertEqual(cache.get("a"), 1)
        self.assertTrue(cache.contains("a"))
        
        # Test immediate expiration check
        self.assertEqual(cache.size, 1)
        
        # Wait for expiration
        time.sleep(0.15)  # 150ms > 100ms TTL
        
        # Should be expired
        self.assertIsNone(cache.get("a"))
        self.assertFalse(cache.contains("a"))
    
    @patch('threadx.utils.cache.time.time')
    def test_ttl_cache_expiration_mocked(self, mock_time):
        """Test TTL cache expiration with mocked time."""
        
        # Mock time progression
        mock_time.side_effect = [0.0, 0.0, 0.0, 100.0, 100.0, 200.0, 200.0]
        
        cache = TTLCache[str, int](ttl_seconds=50.0)
        
        # Set at time 0
        cache.set("a", 1)
        
        # Get at time 0 (within TTL)
        self.assertEqual(cache.get("a"), 1)
        
        # Get at time 100 (still within TTL of 50s from time 0)
        self.assertEqual(cache.get("a"), 1)
        
        # Get at time 200 (expired)
        self.assertIsNone(cache.get("a"))
    
    def test_ttl_cache_purge_expired(self):
        """Test explicit purging of expired items."""
        cache = TTLCache[str, int](ttl_seconds=0.05)  # 50ms TTL
        
        # Add multiple items
        cache.set("a", 1)
        cache.set("b", 2)
        cache.set("c", 3)
        
        self.assertEqual(cache.size, 3)
        
        # Wait for some to expire
        time.sleep(0.03)  # 30ms
        cache.set("d", 4)  # Fresh item
        
        time.sleep(0.03)  # Total 60ms, first 3 should be expired
        
        # Explicit purge
        expired_count = cache.purge_expired()
        
        # Should have purged the first 3 expired items
        self.assertEqual(expired_count, 3)
        self.assertEqual(cache.size, 1)  # Only "d" remains
        self.assertEqual(cache.get("d"), 4)
    
    def test_stable_key_generation(self):
        """Test stable key generation for caching."""
        
        def test_func(x, y, z=10):
            return x + y + z
        
        # Same arguments should produce same key
        key1 = generate_stable_key(test_func, (1, 2), {'z': 10})
        key2 = generate_stable_key(test_func, (1, 2), {'z': 10})
        self.assertEqual(key1, key2)
        
        # Different arguments should produce different keys
        key3 = generate_stable_key(test_func, (1, 3), {'z': 10})
        self.assertNotEqual(key1, key3)
        
        # Different kwargs should produce different keys
        key4 = generate_stable_key(test_func, (1, 2), {'z': 20})
        self.assertNotEqual(key1, key4)
        
        # Keys should be strings and reasonably short
        self.assertIsInstance(key1, str)
        self.assertLess(len(key1), 100)
        
        # Should include function name
        self.assertIn("test_func", key1)
    
    def test_cached_decorator_ttl(self):
        """Test @cached decorator with TTL."""
        
        call_count = 0
        
        @cached(ttl=1)  # 1 second TTL
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # First call
        result1 = expensive_function(5)
        self.assertEqual(result1, 10)
        self.assertEqual(call_count, 1)  # Function called
        
        # Second call (should hit cache)
        result2 = expensive_function(5)
        self.assertEqual(result2, 10)
        self.assertEqual(call_count, 1)  # Function not called again
        
        # Different argument (cache miss)
        result3 = expensive_function(6)
        self.assertEqual(result3, 12)
        self.assertEqual(call_count, 2)  # Function called again
    
    def test_cached_decorator_lru(self):
        """Test @cached decorator with LRU."""
        
        call_count = 0
        
        @cached(lru=2)  # Capacity of 2
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # Fill cache
        expensive_function(1)  # Call 1
        expensive_function(2)  # Call 2
        self.assertEqual(call_count, 2)
        
        # Hit cache
        expensive_function(1)  # Cache hit
        self.assertEqual(call_count, 2)
        
        # Cause eviction
        expensive_function(3)  # Call 3, should evict 2
        self.assertEqual(call_count, 3)
        
        # Verify eviction
        expensive_function(2)  # Should be cache miss (evicted)
        self.assertEqual(call_count, 4)
    
    def test_cached_decorator_combined(self):
        """Test @cached decorator with both TTL and LRU."""
        
        call_count = 0
        
        @cached(ttl=10, lru=2)  # 10s TTL + LRU capacity 2
        def expensive_function(x):
            nonlocal call_count
            call_count += 1
            return x * 2
        
        # Basic functionality
        result1 = expensive_function(1)
        result2 = expensive_function(1)  # Cache hit
        
        self.assertEqual(result1, 2)
        self.assertEqual(result2, 2)
        self.assertEqual(call_count, 1)
        
        # Test cache management methods
        self.assertTrue(hasattr(expensive_function, 'cache_stats'))
        self.assertTrue(hasattr(expensive_function, 'cache_clear'))
        self.assertTrue(hasattr(expensive_function, 'cache_info'))
        
        # Test cache info
        info = expensive_function.cache_info()
        self.assertIn('type', info)
        self.assertEqual(info['type'], 'combined_lru_ttl')
    
    def test_thread_safety(self):
        """Test cache thread safety."""
        cache = LRUCache[int, int](capacity=100)
        results = []
        errors = []
        
        def worker(thread_id):
            try:
                for i in range(50):
                    key = thread_id * 100 + i
                    cache.set(key, key * 2)
                    
                for i in range(50):
                    key = thread_id * 100 + i
                    value = cache.get(key)
                    if value is not None:
                        results.append(value)
            except Exception as e:
                errors.append(e)
        
        # Start multiple threads
        threads = []
        for i in range(5):
            t = threading.Thread(target=worker, args=(i,))
            threads.append(t)
            t.start()
        
        # Wait for completion
        for t in threads:
            t.join()
        
        # Check no errors occurred
        self.assertEqual(len(errors), 0)
        
        # Check some results were recorded
        self.assertGreater(len(results), 0)


class TestXP(unittest.TestCase):
    """Test device-agnostic computing utilities."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
    
    def test_gpu_available_detection(self):
        """Test GPU availability detection."""
        # Should return boolean without error
        available = gpu_available()
        self.assertIsInstance(available, bool)
    
    def test_xp_returns_module(self):
        """Test xp() returns array module."""
        xp_module = xp()
        
        # Should be either numpy or cupy
        self.assertTrue(hasattr(xp_module, 'array'))
        self.assertTrue(hasattr(xp_module, 'mean'))
        self.assertTrue(hasattr(xp_module, 'sum'))
        
        # Should work for basic operations
        arr = xp_module.array([1, 2, 3, 4, 5])
        mean_val = xp_module.mean(arr)
        self.assertEqual(float(mean_val), 3.0)
    
    def test_to_device_to_host_roundtrip(self):
        """Test device/host transfers."""
        # Create test data
        cpu_data = np.array([1, 2, 3, 4, 5])
        
        # Move to device (might be no-op if no GPU)
        device_data = to_device(cpu_data)
        
        # Should still be array-like
        self.assertTrue(hasattr(device_data, '__len__'))
        self.assertEqual(len(device_data), 5)
        
        # Move back to host
        host_data = to_host(device_data)
        
        # Should be numpy array
        self.assertIsInstance(host_data, np.ndarray)
        np.testing.assert_array_equal(host_data, cpu_data)
    
    def test_to_device_to_host_with_lists(self):
        """Test device transfers with Python lists."""
        original_list = [1.0, 2.0, 3.0]
        
        device_data = to_device(original_list)
        host_data = to_host(device_data)
        
        # Should preserve values
        np.testing.assert_array_equal(host_data, np.array(original_list))
    
    @patch('threadx.utils.xp.gpu_available', return_value=False)
    def test_xp_cpu_fallback(self, mock_gpu_available):
        """Test CPU fallback when GPU unavailable."""
        xp_module = xp()
        
        # Should return numpy
        self.assertEqual(xp_module.__name__, 'numpy')
        
        # Device operations should be no-ops
        data = np.array([1, 2, 3])
        device_data = to_device(data)
        
        # Should be same object (no-op)
        self.assertIs(device_data, data)
    
    @patch('threadx.utils.xp.CUPY_AVAILABLE', True)
    @patch('threadx.utils.xp.cp')
    def test_xp_gpu_path(self, mock_cupy):
        """Test GPU path when CuPy available."""
        # Mock CuPy module
        mock_cupy.cuda.runtime.getDeviceCount.return_value = 1
        mock_cupy.cuda.runtime.memGetInfo.return_value = (1000000, 2000000)
        
        # Mock device context
        mock_device = Mock()
        mock_cupy.cuda.Device.return_value = mock_device
        
        # Reset GPU state to force re-initialization
        import threadx.utils.xp as xp_module
        xp_module._gpu_enabled = None
        xp_module._gpu_devices_available = None
        
        # Should detect GPU as available
        available = xp_module.gpu_available()
        self.assertTrue(available)
    
    def test_get_array_info(self):
        """Test array information gathering."""
        from threadx.utils.xp import get_array_info
        
        # Test with numpy array
        arr = np.random.randn(100, 50).astype(np.float32)
        info = get_array_info(arr)
        
        self.assertIn('device', info)
        self.assertIn('shape', info)
        self.assertIn('dtype', info)
        self.assertIn('memory_mb', info)
        
        self.assertEqual(info['shape'], (100, 50))
        self.assertEqual(info['dtype'], np.float32)
        self.assertGreater(info['memory_mb'], 0)
    
    def test_ensure_array_type(self):
        """Test array type conversion."""
        from threadx.utils.xp import ensure_array_type
        
        # Test with list
        data = [1, 2, 3, 4, 5]
        arr = ensure_array_type(data, dtype=np.float64)
        
        self.assertTrue(hasattr(arr, 'dtype'))
        self.assertEqual(arr.dtype, np.float64)
        np.testing.assert_array_equal(arr, [1.0, 2.0, 3.0, 4.0, 5.0])


class TestIntegration(unittest.TestCase):
    """Test integration between different utils modules."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
    
    def test_cached_indicator_mock(self):
        """Test caching integration with mock indicator function."""
        
        # Mock an indicator calculation
        call_count = 0
        
        @cached(ttl=60, lru=100, namespace="indicators")
        def compute_bollinger_bands(prices, period, std_dev):
            nonlocal call_count
            call_count += 1
            
            # Simulate expensive calculation
            time.sleep(0.001)  # 1ms
            
            xp_module = xp()
            prices_arr = xp_module.asarray(prices)
            
            # Simple moving average
            if len(prices_arr) < period:
                return None
                
            sma = xp_module.convolve(
                prices_arr, 
                xp_module.ones(period) / period, 
                mode='valid'
            )
            
            return to_host(sma)  # Ensure result on host
        
        # Test data
        prices = np.random.randn(1000) + 100  # Random walk around 100
        
        # First call
        result1 = compute_bollinger_bands(prices, 20, 2.0)
        self.assertEqual(call_count, 1)
        self.assertIsNotNone(result1)
        self.assertIsInstance(result1, np.ndarray)
        
        # Second call with same params (cache hit)
        result2 = compute_bollinger_bands(prices, 20, 2.0)
        self.assertEqual(call_count, 1)  # Not called again
        np.testing.assert_array_equal(result1, result2)
        
        # Different params (cache miss)
        result3 = compute_bollinger_bands(prices, 21, 2.0)
        self.assertEqual(call_count, 2)  # Called again
        self.assertIsNotNone(result3)
    
    @patch('threadx.utils.timing.logger')
    def test_performance_measurement_with_caching(self, mock_logger):
        """Test performance measurement combined with caching."""
        
        @measure_throughput("cached_computation", unit_of_work="calculation")
        @cached(ttl=30)
        def expensive_computation(data_size):
            # Simulate expensive work
            xp_module = xp()
            data = xp_module.random.randn(data_size)
            result = xp_module.mean(data)
            return to_host(result)
        
        # First call (cache miss + measurement)
        result1 = expensive_computation(1000)
        self.assertIsInstance(float(result1), float)
        
        # Second call (cache hit + measurement)
        result2 = expensive_computation(1000)
        self.assertEqual(result1, result2)
        
        # Check logging occurred
        self.assertTrue(mock_logger.info.called)
        
        # Should have logged about calculations/min
        log_messages = [str(call) for call in mock_logger.info.call_args_list]
        throughput_logged = any("calculations/min" in msg for msg in log_messages)
        self.assertTrue(throughput_logged)
    
    def test_vectorized_operations_performance(self):
        """Test vectorized operations meet performance requirements."""
        
        @measure_throughput(unit_of_work="element")
        def vectorized_calculation(data):
            xp_module = xp()
            arr = xp_module.asarray(data)
            
            # Vectorized operations
            result = xp_module.sqrt(xp_module.abs(arr))
            result = xp_module.mean(result)
            
            return to_host(result)
        
        # Large dataset to test performance
        large_data = np.random.randn(50000)  # 50k elements
        
        # Time the operation
        start_time = time.time()
        result = vectorized_calculation(large_data)
        elapsed = time.time() - start_time
        
        # Should be reasonably fast (< 100ms for 50k elements)
        self.assertLess(elapsed, 0.1)
        self.assertIsInstance(float(result), float)
        
        # Calculate elements per second
        elements_per_sec = len(large_data) / elapsed
        
        # Should process > 500k elements/sec (reasonable for modern hardware)
        self.assertGreater(elements_per_sec, 500000)


class TestBenchmarkSuite(unittest.TestCase):
    """Internal benchmark tests for barème de succès validation."""
    
    def setUp(self):
        """Set up test fixtures."""
        np.random.seed(42)
    
    def test_throughput_benchmark_cpu(self):
        """Benchmark CPU throughput to meet >1500 tasks/min requirement."""
        
        @measure_throughput("benchmark_cpu", unit_of_work="operation")
        def cpu_benchmark():
            # Simulate typical indicator calculations
            data = np.random.randn(1000)
            
            # Multiple vectorized operations
            sma = np.convolve(data, np.ones(20)/20, mode='valid')
            std = np.std(data)
            result = np.mean(sma) + std
            
            return result
        
        # Run multiple iterations to measure throughput
        start_time = time.time()
        results = []
        
        for _ in range(100):  # 100 operations
            result = cpu_benchmark()
            results.append(result)
            
        elapsed = time.time() - start_time
        operations_per_min = (100 * 60) / elapsed
        
        # Should achieve > 1500 operations/min
        self.assertGreater(operations_per_min, 1500)
        
        # Results should be consistent
        self.assertEqual(len(results), 100)
        self.assertTrue(all(isinstance(r, (int, float)) for r in results))
    
    def test_cache_performance_benchmark(self):
        """Benchmark cache performance for hit rate and access speed."""
        
        # Create cache with reasonable size
        cache = LRUCache[str, np.ndarray](capacity=1000)
        
        # Generate test data
        test_arrays = {}
        for i in range(100):
            key = f"array_{i}"
            value = np.random.randn(100)
            test_arrays[key] = value
            cache.set(key, value)
        
        # Benchmark cache access speed
        start_time = time.time()
        
        hits = 0
        for _ in range(10000):  # 10k accesses
            key = f"array_{np.random.randint(0, 100)}"
            result = cache.get(key)
            if result is not None:
                hits += 1
                
        elapsed = time.time() - start_time
        accesses_per_sec = 10000 / elapsed
        hit_rate = hits / 10000 * 100
        
        # Performance requirements
        self.assertGreater(accesses_per_sec, 100000)  # > 100k accesses/sec
        self.assertGreater(hit_rate, 90)  # > 90% hit rate
        
        # Verify cache stats
        stats = cache.stats()
        self.assertGreater(stats.hit_rate, 90)
        self.assertEqual(stats.current_size, 100)  # All items fit
    
    def test_memory_tracking_accuracy(self):
        """Test memory tracking accuracy and performance."""
        
        # Only run if psutil available
        try:
            import psutil
        except ImportError:
            self.skipTest("psutil not available")
        
        @track_memory("memory_test")
        def memory_intensive_function():
            # Allocate and use memory
            large_arrays = []
            for _ in range(10):
                arr = np.random.randn(10000)  # ~80KB each
                large_arrays.append(arr)
                
            # Do some computation
            total = sum(np.sum(arr) for arr in large_arrays)
            return total
        
        result = memory_intensive_function()
        
        # Should complete without error
        self.assertIsInstance(result, (int, float))


if __name__ == '__main__':
    # Set up test environment
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)  # No env vars
    
    # Run tests with verbose output
    unittest.main(verbosity=2, buffer=True)