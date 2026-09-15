#include <cuda.h>
#include <cuda_runtime.h>
#include "common.h"
#include "naive.h"

namespace StreamCompaction {
    namespace Naive {
        using StreamCompaction::Common::PerformanceTimer;
        PerformanceTimer& timer()
        {
            static PerformanceTimer timer;
            return timer;
        }
        
        __global__ void kernScan(int n, int offset, int* odata, const int* idata) {
            int index = blockIdx.x * blockDim.x + threadIdx.x;
            if (index >= n) {
                return;
            }

            if (index >= offset) {
                odata[index] = idata[index] + idata[index - offset];
            }
            else {
                odata[index] = idata[index];
            }
        }

        /**
         * Performs prefix-sum (aka scan) on idata, storing the result into odata.
         */
        void scan(int n, int *odata, const int *idata) {
            int* dev_a;
            int* dev_b;
            cudaMalloc((void**)&dev_a, n * sizeof(int));
            cudaMalloc((void**)&dev_b, n * sizeof(int));

            cudaMemcpy(dev_a, idata, n * sizeof(int), cudaMemcpyHostToDevice);

            const int blockSize = 128;
            const int blocks = (n + blockSize - 1) / blockSize;

            timer().startGpuTimer();

            for (int offset = 1; offset < n; offset *= 2) {
                kernScan << <blocks, blockSize >> > (n, offset, dev_b, dev_a);
                std::swap(dev_a, dev_b);
            }

            timer().endGpuTimer();

            odata[0] = 0;
            cudaMemcpy(odata + 1, dev_a, (n - 1) * sizeof(int), cudaMemcpyDeviceToHost);

            cudaFree(dev_a);
            cudaFree(dev_b);
        }
    }
}
