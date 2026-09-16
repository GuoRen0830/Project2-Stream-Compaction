#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "efficient.h"

namespace StreamCompaction {
    namespace Efficient {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }

        __global__ void kernUpSweep(int n, int d, int* data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            
            size_t stride = 1 << (d + 1);
            size_t k = static_cast<size_t>(index) * stride;

            if (k < static_cast<size_t>(n)) {
                data[k + stride - 1] += data[k + (size_t(1) << d) - 1];
            }
        }

        __global__ void kernDownSweep(int n, int d, int* data) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;

            size_t stride = size_t(1) << (d + 1);
            size_t k = static_cast<size_t>(index) * stride;

            if (k < static_cast<size_t>(n)) {
                size_t left = k + (size_t(1) << d) - 1;
                size_t right = k + stride - 1;

                int temp = data[left];
                data[left] = data[right];
                data[right] += temp;
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int depth = ilog2ceil(n);
            int paddedSize = 1 << depth;

            int* dev_data;
            cudaMalloc((void**)&dev_data, paddedSize * sizeof(int));
            cudaMemset(dev_data, 0, paddedSize * sizeof(int));
            cudaMemcpy(dev_data, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            const int blockSize = 512;
            
            timer().startGpuTimer();

            // Up Sweep
            for (int d = 0; d < depth; ++d) {
                int activeThreads = paddedSize >> (d + 1);
                int blocks = (activeThreads + blockSize - 1) / blockSize;

                kernUpSweep << <blocks, blockSize >> > (paddedSize, d, dev_data);
            }

            // Set last element = 0
            cudaMemset(dev_data + paddedSize - 1, 0, sizeof(int));

            // Down Sweep
            for (int d = depth - 1; d >= 0; --d) {
                int activeThreads = paddedSize >> (d + 1);
                int blocks = (activeThreads + blockSize - 1) / blockSize;

                kernDownSweep << <blocks, blockSize >> > (paddedSize, d, dev_data);
            }
            
            timer().endGpuTimer();

            checkCUDAError("Efficient scan");

            cudaMemcpy(odata, dev_data, n * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_data);
        }

        /**
         * Performs stream compaction on idata, storing the result into odata.
         * All zeroes are discarded.
         *
         * @param n      The number of elements in idata.
         * @param odata  The array into which to store elements.
         * @param idata  The array of elements to compact.
         * @returns      The number of elements remaining after compaction.
         */
        int compact(int n, int *odata, const int *idata) {
            int depth = ilog2ceil(n);
            int paddedSize = 1 << depth;

            int* dev_idata;
            int* dev_odata;
            int* dev_bools;
            int* dev_indices;

            cudaMalloc((void**)&dev_idata, n * sizeof(int));
            cudaMalloc((void**)&dev_odata, n * sizeof(int));
            cudaMalloc((void**)&dev_bools, n * sizeof(int));
            cudaMalloc((void**)&dev_indices, paddedSize * sizeof(int));

            cudaMemset(dev_indices, 0, paddedSize * sizeof(int));

            cudaMemcpy(dev_idata, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            const int blockSize = 128;
            const int blocks = (n + blockSize - 1) / blockSize;
            const int scanBlocks = (paddedSize + blockSize - 1) / blockSize;

            timer().startGpuTimer();

            // Map
            Common::kernMapToBoolean << <blocks, blockSize >> > (n, dev_bools, dev_idata);

            // Up / Down Sweep
            cudaMemcpy(dev_indices, dev_bools, n * sizeof(int), cudaMemcpyDeviceToDevice);

            for (int d = 0; d < depth; ++d) {
                kernUpSweep << <scanBlocks, blockSize >> > (paddedSize, d, dev_indices);
            }

            cudaMemset(dev_indices + paddedSize - 1, 0, sizeof(int));

            for (int d = depth - 1; d >= 0; --d) {
                kernDownSweep << <scanBlocks, blockSize >> > (paddedSize, d, dev_indices);
            }

            // Scatter
            Common::kernScatter << <blocks, blockSize >> > (n, dev_odata, dev_idata, dev_bools, dev_indices);
            
            timer().endGpuTimer();

            int lastIndex;
            int lastBool;
            cudaMemcpy(&lastIndex, dev_indices + n - 1, sizeof(int), cudaMemcpyDeviceToHost);
            cudaMemcpy(&lastBool, dev_bools + n - 1, sizeof(int), cudaMemcpyDeviceToHost);

            int count = lastIndex + lastBool;

            cudaMemcpy(odata, dev_odata, count * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_idata);
            cudaFree(dev_odata);
            cudaFree(dev_bools);
            cudaFree(dev_indices);

            return count;
        }
    }
}
