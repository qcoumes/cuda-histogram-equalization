#ifndef GPGPU_EQUALIZER_HPP
#define GPGPU_EQUALIZER_HPP

#include <cstdint>
#include <string>

#include <cuda_runtime.h>
#include <vector_types.h>


namespace equalizer {
    
    void equalizeGpu(const float4 *const input, float4 *output, uint32_t width, uint32_t height);
    
    void equalizeCpu(const float4 *const input, float4 *output, uint32_t width, uint32_t height);
}

#endif //GPGPU_EQUALIZER_HPP
