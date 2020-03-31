#ifndef GPGPU_IMAGE_HPP
#define GPGPU_IMAGE_HPP

#include <vector>

#include <vector_types.h>


namespace equalizer {
    
    class Image {
        
        public:
            
            static void decodePNG(const std::string &filename, std::vector<float4> &out, uint32_t &w, uint32_t &h);
            
            static void encodePNG(const std::string &path, const std::vector<float4> &in, uint32_t w, uint32_t h);
    };
}

#endif // GPGPU_IMAGE_HPP
