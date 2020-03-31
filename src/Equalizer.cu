#include <vector>

#include <Equalizer.hpp>
#include <Utils.hpp>


#define D_1_3f 0.3333333333333333f
#define D_2_3f 0.6666666666666666f

namespace equalizer {
    
    
    ////////////////////////////////////////////////////////////////////////////
    /////////////////////////////// HOST & DEVICE //////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    
    __host__ __device__ void toHsva(const float4 rgba, float4 *hsva) {
        float r = rgba.x;
        float g = rgba.y;
        float b = rgba.z;
        
        float max = fmaxf(r, fmaxf(g, b));
        float min = fminf(r, fminf(g, b));
        float d = max - min;
        float h = 0, s = 0, v = max;
        
        if (d != 0) {
            float dR = (((max - r) / 6) + (d / 2)) / d;
            float dG = (((max - g) / 6) + (d / 2)) / d;
            float dB = (((max - b) / 6) + (d / 2)) / d;
            
            if (r == max) {
                h = dB - dG;
            }
            else if (g == max) {
                h = (D_1_3f) + dR - dB;
            }
            else if (b == max) {
                h = (D_2_3f) + dG - dR;
            }
            
            s = d / v;
            
            if (h < 0) {
                h += 1.f;
            }
            else if (h > 1) {
                h -= 1.f;
            }
        }
        
        hsva->x = h;
        hsva->y = s;
        hsva->z = v;
        hsva->w = rgba.w;
    }
    
    
    __host__ __device__ void toRgba(const float4 hsva, float4 *rgba) {
        float r = NAN, g = NAN, b = NAN;
        float h = hsva.x;
        float s = hsva.y;
        float v = hsva.z;
        
        if (s == 0) {
            r = g = b = v;
        }
        
        else {
            h *= 6;
            
            float i = static_cast<uint8_t>(h);
            float c = v * (1 - s);
            float m = v * (1 - s * (h - i));
            float X = v * (1 - s * (1 - (h - i)));
            
            switch (static_cast<uint8_t>(i)) {
                case 0:
                    r = v;
                    g = X;
                    b = c;
                    break;
                case 1:
                    r = m;
                    g = v;
                    b = c;
                    break;
                case 2:
                    r = c;
                    g = v;
                    b = X;
                    break;
                case 3:
                    r = c;
                    g = m;
                    b = v;
                    break;
                case 4:
                    r = X;
                    g = c;
                    b = v;
                    break;
                case 5:
                    r = v;
                    g = c;
                    b = m;
                    break;
                default:
                    r = m;
                    g = m;
                    b = m;
                    break;
            }
        }
        
        rgba->x = r;
        rgba->y = g;
        rgba->z = b;
        rgba->w = hsva.w;
    }
    
    
    ////////////////////////////////////////////////////////////////////////////
    /////////////////////////////////// DEVICE /////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    
    __constant__ float DEV_SCALE;
    __constant__ uint32_t DEV_WIDTH;
    __constant__ uint32_t DEV_HEIGHT;
    
    
    __global__ void toHsvaArrayAndHistoGpu(const float4 *rgba, float4 *hsva, float *histo) {
        uint32_t gridBlockDimX = gridDim.x * blockDim.x;
        uint32_t gridBlockDimY = gridDim.y * blockDim.y;
        uint32_t index;
        
        for (uint32_t indexY = blockIdx.y * blockDim.y + threadIdx.y; indexY < DEV_HEIGHT; indexY += gridBlockDimY) {
            for (uint32_t indexX = blockIdx.x * blockDim.x + threadIdx.x; indexX < DEV_WIDTH; indexX += gridBlockDimX) {
                index = indexY * DEV_WIDTH + indexX;
                toHsva(rgba[index], &hsva[index]);
                atomicAdd(&histo[static_cast<uint8_t>(hsva[indexY * DEV_WIDTH + indexX].z)], 1);
            }
        }
    }
    
    
    __global__ void computeScaledCumulatedHistoGpu(const float *histo, float *scaled) {
        for (int i = 0; i <= threadIdx.x; i++) {
            scaled[threadIdx.x] += histo[i];
        }
        scaled[threadIdx.x] *= DEV_SCALE;
    }
    
    
    __global__ void equalizeAndToRgbaArrayGpu(const float4 *hsva, const float *scaled, float4 *rgba) {
        uint32_t gridBlockDimX = gridDim.x * blockDim.x;
        uint32_t gridBlockDimY = gridDim.y * blockDim.y;
        uint32_t index;
        float4 tmp;
        
        for (uint32_t indexY = blockIdx.y * blockDim.y + threadIdx.y; indexY < DEV_HEIGHT; indexY += gridBlockDimY) {
            for (uint32_t indexX = blockIdx.x * blockDim.x + threadIdx.x; indexX < DEV_WIDTH; indexX += gridBlockDimX) {
                index = indexY * DEV_WIDTH + indexX;
                tmp = hsva[index];
                tmp.z = scaled[static_cast<uint8_t>(tmp.z)];
                toRgba(tmp, &rgba[index]);
            }
        }
    }
    
    
    void equalizeGpu(const float4 *const input, float4 *output, uint32_t width, uint32_t height) {
        std::vector<float> histogram(256), cumulated(256), scaled(256);
        ChronoCPU total;
        
        const uint32_t imageSize = width * height * sizeof(float4);
        const uint32_t histogramSize = 256 * sizeof(float);
        const dim3 nbThreads = dim3(32, 32);
        const dim3 nbBlocks = dim3(
                (width + nbThreads.x - 1) / nbThreads.x,
                (height + nbThreads.y - 1) / nbThreads.y
        );
        const float scale = 255.f / (width * height);
        float4 *devRGBA = nullptr;
        float4 *devHSVA = nullptr;
        float *devHistogram = nullptr;
        float *devScaled = nullptr;
        if (VERBOSE) {
            std::cout << "==================== GPU ====================" << std::endl;
            total.start();
        }
        
        GPU_VERBOSE(
                HANDLE_CUDA_ERROR(cudaMalloc((void **) &devRGBA, imageSize));
                HANDLE_CUDA_ERROR(cudaMalloc((void **) &devHSVA, imageSize));
                HANDLE_CUDA_ERROR(cudaMalloc((void **) &devHistogram, histogramSize));
                HANDLE_CUDA_ERROR(cudaMalloc((void **) &devScaled, histogramSize));
                HANDLE_CUDA_ERROR(cudaMemset(devHistogram, 0, histogramSize));
                HANDLE_CUDA_ERROR(cudaMemset(devScaled, 0, histogramSize));
                cudaMemcpyToSymbol(DEV_WIDTH, &width, sizeof(float));
                cudaMemcpyToSymbol(DEV_HEIGHT, &height, sizeof(float));
                cudaMemcpyToSymbol(DEV_SCALE, &scale, sizeof(float)),
                "Allocating memory on device.."
        );
        
        GPU_VERBOSE(
                HANDLE_CUDA_ERROR(cudaMemcpy(devRGBA, input, imageSize, cudaMemcpyHostToDevice)),
                "Copying image to device..."
        );
        
        // We could do the complete equalization in one kernel with __syncthreads(), but it is not very convenient for
        // measuring the time taken by each step.
        GPU_VERBOSE(
                (toHsvaArrayAndHistoGpu<<<nbBlocks, nbThreads>>>(devRGBA, devHSVA, devHistogram)),
                "Converting RGBA to HSVA and computing histogram..."
        );
        GPU_VERBOSE(
                (computeScaledCumulatedHistoGpu<<<1, 256>>>(devHistogram, devScaled)),
                "Computing scaled cumulated histogram..."
        );
        GPU_VERBOSE(
                (equalizeAndToRgbaArrayGpu<<<nbBlocks, nbThreads>>>(devHSVA, devScaled, devRGBA)),
                "Equalizing histogram and converting HSVA to RGBA..."
        );
        
        GPU_VERBOSE(
                HANDLE_CUDA_ERROR(cudaMemcpy(output, devRGBA, imageSize, cudaMemcpyDeviceToHost)),
                "Copying equalized image to host..."
        );
        
        GPU_VERBOSE(
                HANDLE_CUDA_ERROR(cudaFree(devRGBA));
                HANDLE_CUDA_ERROR(cudaFree(devHSVA));
                HANDLE_CUDA_ERROR(cudaFree(devHistogram));
                HANDLE_CUDA_ERROR(cudaFree(devScaled)),
                "Freeing memory on device..."
        );
        
        if (VERBOSE) {
            total.stop();
            std::cout << "Total : " << total.elapsedTime() << " ms" << std::endl;
        }
    }
    
    
    ////////////////////////////////////////////////////////////////////////////
    //////////////////////////////////// HOST //////////////////////////////////
    ////////////////////////////////////////////////////////////////////////////
    
    float HOST_WIDTH;
    float HOST_HEIGHT;
    
    void toHsvaArrayAndHistoCpu(const float4 *rgba, float4 *hsva, float *histo) {
        uint32_t index;
        
        for (uint32_t indexY = 0; indexY < HOST_HEIGHT; indexY++) {
            for (uint32_t indexX = 0; indexX < HOST_WIDTH; indexX++) {
                index = indexY * HOST_WIDTH + indexX;
                toHsva(rgba[index], &hsva[index]);
                histo[static_cast<uint8_t>(hsva[index].z)]++;
            }
        }
    }
    
    void equalizeAndToRgbaArrayCpu(const float *scaled, const float4 *hsva, float4 *rgba) {
        uint32_t index;
        float4 tmp;
        
        for (uint32_t indexY = 0; indexY < HOST_HEIGHT; indexY++) {
            for (uint32_t indexX = 0; indexX < HOST_WIDTH; indexX++) {
                index = indexY * HOST_WIDTH + indexX;
                tmp = hsva[index];
                tmp.z = scaled[static_cast<uint8_t>(tmp.z)];
                toRgba(tmp, &rgba[index]);
            }
        }
    }
    
    
    void equalizeCpu(const float4 *const input, float4 *output, uint32_t width, uint32_t height) {
        std::vector<float> histogram(256), cumulated(256), scaled(256);
        std::vector<float4> hsva;
        ChronoCPU total;
        
        HOST_WIDTH = width;
        HOST_HEIGHT = height;
        
        if (VERBOSE) {
            std::cout << "==================== CPU ====================" << std::endl;
            total.start();
        }
        
        CPU_VERBOSE(
                hsva.resize(width * height);
                toHsvaArrayAndHistoCpu(input, hsva.data(), histogram.data()),
                "Converting RGBA to HSVA and computing histogram..."
        );
        
        CPU_VERBOSE(
                cumulated[0] = histogram[0];
                for (int i = 1; i < 256; i++) {
                    cumulated[i] = histogram[i] + cumulated[i - 1];
                },
                "Computing cumulated histogram..."
        );
        
        CPU_VERBOSE(
                float scale = 255.f / (width * height);
                for (uint32_t i = 0; i < 256; i++) {
                    scaled[i] = scale * cumulated[i];
                },
                "Scaling histogram..."
        );
        
        CPU_VERBOSE(
                equalizeAndToRgbaArrayCpu(scaled.data(), hsva.data(), output),
                "Equalizing histogram and converting HSVA to RGBA..."
        );
        
        if (VERBOSE) {
            total.stop();
            std::cout << "Total : " << total.elapsedTime() << " ms" << std::endl;
        }
    }
}
