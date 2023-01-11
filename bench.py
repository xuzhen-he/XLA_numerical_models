import argparse
import math


class bench:

    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Benchmarking ops')
        self.parser.add_argument('--size',
                            type=int,
                            action='store',
                            # default=16 * 1024 * 1024,
                            # default=64 * 1024 * 1024,
                            default=512 * 512,
                            help='array size')
        self.parser.add_argument('--time',
                            type=float,
                            action='store',
                            default=5.0,
                            help='benchmark duration')

        args = self.parser.parse_args()

        self.benchtime = 5.0
        self.memory_transfer_per_loop = 0.0
        self.loops = 0.0
        self.duration = 0.0

        self.total_size = args.size
        self.elem_size = 0.0
        self.benchtime = args.time

    def assert_shape_and_dtype(self, x):
        assert x.shape == (self.total_size, )
        assert x.dtype.itemsize == self.elem_size

    def print_bench(self):
        print("\nBench info: 1d")
        print("  Total size: ", self.total_size)
        print("  Size of element ", self.elem_size, " Byte")
        print("  Benchmark duration ", self.benchtime, " s")

    def print_performance(self):
        bandwidth = self.memory_transfer_per_loop * self.loops / self.duration
        print("\nPerformance")
        print("  Memory transfer per loop ", self.memory_transfer_per_loop,
              " GB")
        print("  Run time: ", self.duration, " seconds")
        print("  Loops: ", self.loops)
        print("  bandwith: ", bandwidth, " GB/s")


class bench2d(bench):

    def __init__(self):
        super().__init__()
        self.side_size = int(math.sqrt(self.total_size))
        self.total_size = self.side_size * self.side_size

    def assert_shape_and_dtype(self, x):
        assert x.shape == (self.side_size, self.side_size)
        assert x.dtype.itemsize == self.elem_size

    def print_bench(self):
        print("\nBench info: 2d")
        print("  Total size: ", self.total_size)
        print("  Side size: ", self.side_size)
        print("  Size of element ", self.elem_size, " Byte")
        print("  Benchmark duration ", self.benchtime, " s")
