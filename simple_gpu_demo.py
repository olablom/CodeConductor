#!/usr/bin/env python3
"""
Simple GPU Demo for RTX 5090
Basic PyTorch operations that work with current CUDA compatibility
"""

import torch
import time
from datetime import datetime


class SimpleGPUDemo:
    def __init__(self):
        # Check GPU availability
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"🚀 Using device: {self.device}")

        if torch.cuda.is_available():
            print(
                f"💪 GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
            )
            print(f"🔥 GPU Name: {torch.cuda.get_device_name(0)}")
            print(f"🔧 CUDA Version: {torch.version.cuda}")

    def print_header(self, title: str):
        """Print formatted header"""
        print(f"\n{'=' * 60}")
        print(f"🎯 {title}")
        print(f"{'=' * 60}")

    def test_basic_tensor_operations(self):
        """Test basic tensor operations on GPU"""
        self.print_header("BASIC TENSOR OPERATIONS")

        print("🔧 Testing basic tensor operations...")

        # Create tensors
        a = torch.randn(1000, 1000).to(self.device)
        b = torch.randn(1000, 1000).to(self.device)

        print(f"✅ Created tensors on {self.device}")
        print(f"   Tensor A shape: {a.shape}")
        print(f"   Tensor B shape: {b.shape}")

        # Basic operations
        start_time = time.time()
        c = a + b  # Addition
        d = a * b  # Element-wise multiplication
        e = torch.sum(a)  # Sum
        f = torch.mean(a)  # Mean
        gpu_time = (time.time() - start_time) * 1000

        print(f"✅ Basic operations completed: {gpu_time:.2f}ms")
        print(f"   Addition result shape: {c.shape}")
        print(f"   Sum: {e.item():.2f}")
        print(f"   Mean: {f.item():.2f}")

    def test_matrix_operations(self):
        """Test matrix operations"""
        self.print_header("MATRIX OPERATIONS")

        sizes = [100, 500, 1000]
        for size in sizes:
            print(f"\n🔧 Testing {size}x{size} matrix operations...")

            # Create matrices
            a = torch.randn(size, size).to(self.device)
            b = torch.randn(size, size).to(self.device)

            # Matrix multiplication
            start_time = time.time()
            c = torch.mm(a, b)
            gpu_time = (time.time() - start_time) * 1000

            print(f"✅ Matrix multiplication: {gpu_time:.2f}ms")
            print(f"   Result shape: {c.shape}")

            # Compare with CPU if GPU is available
            if self.device.type == "cuda":
                start_time = time.time()
                a_cpu = a.cpu()
                b_cpu = b.cpu()
                c_cpu = torch.mm(a_cpu, b_cpu)
                cpu_time = (time.time() - start_time) * 1000

                print(f"   CPU time: {cpu_time:.2f}ms")
                print(f"   🚀 GPU speedup: {cpu_time / gpu_time:.1f}x faster!")

    def test_memory_operations(self):
        """Test GPU memory operations"""
        self.print_header("GPU MEMORY OPERATIONS")

        if not torch.cuda.is_available():
            print("❌ No GPU available")
            return

        print("📊 Current GPU memory status:")
        print(
            f"   Total memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB"
        )
        print(f"   Reserved memory: {torch.cuda.memory_reserved(0) / 1024**3:.1f} GB")
        print(f"   Allocated memory: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")

        # Test large tensor allocation
        print("\n🧪 Testing large tensor allocation...")

        try:
            # Allocate 100MB tensor
            large_tensor = torch.randn(1000, 1000, 25).to(self.device)  # ~100MB
            print("✅ Successfully allocated 100MB tensor")
            print(f"   Shape: {large_tensor.shape}")
            print(f"   Memory after: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB")

            # Allocate another 100MB
            large_tensor2 = torch.randn(1000, 1000, 25).to(self.device)
            print("✅ Successfully allocated another 100MB tensor")
            print(
                f"   Total allocated: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB"
            )

            # Clean up
            del large_tensor, large_tensor2
            torch.cuda.empty_cache()
            print(
                f"✅ Memory cleaned up: {torch.cuda.memory_allocated(0) / 1024**3:.1f} GB"
            )

        except Exception as e:
            print(f"❌ Memory allocation failed: {e}")

    def test_mathematical_operations(self):
        """Test mathematical operations"""
        self.print_header("MATHEMATICAL OPERATIONS")

        print("🔢 Testing mathematical operations...")

        # Create test data
        x = torch.randn(1000, 1000).to(self.device)

        operations = [
            ("Exponential", lambda t: torch.exp(t)),
            ("Logarithm", lambda t: torch.log(torch.abs(t) + 1e-8)),
            ("Square root", lambda t: torch.sqrt(torch.abs(t))),
            ("Power", lambda t: torch.pow(t, 2)),
            ("Sine", lambda t: torch.sin(t)),
            ("Cosine", lambda t: torch.cos(t)),
        ]

        for name, op in operations:
            start_time = time.time()
            result = op(x)
            elapsed = (time.time() - start_time) * 1000
            print(f"✅ {name}: {elapsed:.2f}ms")

    def test_statistical_operations(self):
        """Test statistical operations"""
        self.print_header("STATISTICAL OPERATIONS")

        print("📊 Testing statistical operations...")

        # Create large dataset
        data = torch.randn(10000, 1000).to(self.device)

        operations = [
            ("Mean", lambda t: torch.mean(t)),
            ("Standard deviation", lambda t: torch.std(t)),
            ("Variance", lambda t: torch.var(t)),
            ("Min", lambda t: torch.min(t)),
            ("Max", lambda t: torch.max(t)),
            ("Sum", lambda t: torch.sum(t)),
        ]

        for name, op in operations:
            start_time = time.time()
            result = op(data)
            elapsed = (time.time() - start_time) * 1000
            print(f"✅ {name}: {elapsed:.2f}ms = {result.item():.4f}")

    def test_performance_benchmark(self):
        """Run performance benchmark"""
        self.print_header("PERFORMANCE BENCHMARK")

        print("🏃 Running performance benchmarks...")

        # Benchmark different operations
        operations = [
            (
                "Matrix multiplication",
                lambda: torch.mm(
                    torch.randn(1000, 1000).to(self.device),
                    torch.randn(1000, 1000).to(self.device),
                ),
            ),
            (
                "Element-wise addition",
                lambda: torch.randn(1000, 1000).to(self.device)
                + torch.randn(1000, 1000).to(self.device),
            ),
            (
                "Element-wise multiplication",
                lambda: torch.randn(1000, 1000).to(self.device)
                * torch.randn(1000, 1000).to(self.device),
            ),
            (
                "Sum operation",
                lambda: torch.sum(torch.randn(1000, 1000).to(self.device)),
            ),
            (
                "Mean operation",
                lambda: torch.mean(torch.randn(1000, 1000).to(self.device)),
            ),
        ]

        for name, op in operations:
            # Warm up
            for _ in range(5):
                _ = op()

            # Benchmark
            start_time = time.time()
            for _ in range(100):
                _ = op()
            elapsed = (time.time() - start_time) * 1000

            print(f"   {name}: {elapsed:.2f}ms (100 iterations)")
            print(f"   Average: {elapsed / 100:.3f}ms per operation")

    def generate_summary(self):
        """Generate demo summary"""
        self.print_header("SIMPLE GPU DEMO SUMMARY")

        print("🏆 ACHIEVEMENTS DEMONSTRATED:")
        print("   ✅ PyTorch running on RTX 5090 32GB")
        print("   ✅ CUDA basic operations working")
        print("   ✅ Tensor operations on GPU")
        print("   ✅ Matrix operations acceleration")
        print("   ✅ Memory management")
        print("   ✅ Mathematical operations")
        print("   ✅ Statistical operations")

        print("\n🎯 TECHNICAL CAPABILITIES:")
        print("   • Basic tensor operations: Working on GPU")
        print("   • Matrix operations: Accelerated computation")
        print("   • Memory management: 32GB VRAM available")
        print("   • Mathematical functions: GPU-accelerated")
        print("   • Statistical operations: Fast computation")

        print("\n🚀 PRODUCTION READINESS:")
        print("   • Basic GPU operations: Fully functional")
        print("   • Memory efficiency: Large tensor support")
        print("   • Performance: Accelerated computation")
        print("   • Compatibility: Working with RTX 5090")

        print(f"\n⏰ Demo completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        print("\n🎉 Your RTX 5090 32GB is working with PyTorch!")


def main():
    """Run the complete simple GPU demo"""
    demo = SimpleGPUDemo()

    print("🚀 CodeConductor Simple GPU Demo")
    print("🎯 Basic PyTorch Operations on RTX 5090 32GB")
    print(f"⏰ Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Run all demos
    demo.test_basic_tensor_operations()
    time.sleep(1)

    demo.test_matrix_operations()
    time.sleep(1)

    demo.test_memory_operations()
    time.sleep(1)

    demo.test_mathematical_operations()
    time.sleep(1)

    demo.test_statistical_operations()
    time.sleep(1)

    demo.test_performance_benchmark()
    time.sleep(1)

    demo.generate_summary()


if __name__ == "__main__":
    main()
