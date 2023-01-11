#include <iostream>
#include <cmath>

#ifndef BENCH_H
#define BENCH_H

template <typename value_t, typename index_t>
struct bench
{
    double memory_transfer_per_loop;
    double benchtime;
    int loops;
    double duration;

    index_t total_size;

    int block;

    bench(int narg, char **arg)
    {
        memory_transfer_per_loop = 0.0;
        benchtime = 5.0;

        total_size = 64 * 1024 * 1024;
        block = 512;

        int i = 1;
        size_t found = 0;
        while (i < narg)
        {
            found = std::string(arg[i]).find("-size");
            if (found != std::string::npos)
            {
                total_size = (index_t)atoi(std::string(arg[i]).erase(0, found + 6).c_str());
            }

            found = std::string(arg[i]).find("-time");
            if (found != std::string::npos)
            {
                benchtime = (double)atof(std::string(arg[i]).erase(0, found + 6).c_str());
            }

            found = std::string(arg[i]).find("-block");
            if (found != std::string::npos)
            {
                block = (int)atoi(std::string(arg[i]).erase(0, found + 8).c_str());
            }
            i++;
        }
    }

    void print_bench()
    {
        std::cout << "\nBench info:\n"
                  << "  Total size: " << total_size << '\n'
                  << "  Size of element " << sizeof(value_t) << " Byte\n"
                  << "  Benchmark duration " << benchtime << " s\n";
    }

    void print_performance()
    {
        double bandwidth = memory_transfer_per_loop * loops / duration;
        std::cout << "\nPerformance\n"
                  << "  Memory transfer per loop " << memory_transfer_per_loop << " GB\n"
                  << "  Run time: " << duration << " seconds\n"
                  << "  Loops: " << loops << '\n'
                  << "  bandwith: " << bandwidth << " GB/s\n\n";
    }
};

template <typename value_t, typename index_t>
struct bench2d : public bench<value_t, index_t>
{
    using bench<value_t, index_t>::total_size;
    using bench<value_t, index_t>::benchtime;
    
    int block0;
    int block1;
    index_t side_size;

    bench2d(int narg, char **arg) : bench<value_t, index_t>(narg, arg)
    {
        block0 = 1;
        block1 = 512;

        int i = 1;
        size_t found = 0;
        while (i < narg)
        {
            found = std::string(arg[i]).find("-block0");
            if (found != std::string::npos)
            {
                block0 = (int)atoi(std::string(arg[i]).erase(0, found + 8).c_str());
            }

            found = std::string(arg[i]).find("-block1");
            if (found != std::string::npos)
            {
                block1 = (int)atoi(std::string(arg[i]).erase(0, found + 8).c_str());
            }
            i++;
        }

        side_size = sqrt(total_size);
        total_size = side_size * side_size;
    }

    void print_bench()
    {
        std::cout << "\nBench info:\n"
                  << "  Side size: " << side_size << '\n'
                  << "  Total size: " << total_size << '\n'
                  << "  Size of element " << sizeof(value_t) << " Byte\n"
                  << "  Benchmark duration " << benchtime << " s\n";
    }
};

#endif
