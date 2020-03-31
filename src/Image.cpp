#include <lodepng/lodepng.hpp>

#include <Image.hpp>
#include <iostream>


namespace equalizer {
    
    void Image::decodePNG(const std::string &filename, std::vector<float4> &out, uint32_t &w, uint32_t &h) {
        std::vector<uint8_t> raw;
        
        uint32_t error = lodepng::decode(raw, w, h, filename);
        if (error) {
            std::cerr << "Could not load image '" << filename << "' : " << lodepng_error_text(error) << std::endl;
            exit(EXIT_FAILURE);
        }
        
        uint32_t size = w * h;
        out.resize(size);
        for (uint32_t i = 0; i < size; i++) {
            out[i] = {
                    static_cast<float>(raw[i * 4 + 0]),
                    static_cast<float>(raw[i * 4 + 1]),
                    static_cast<float>(raw[i * 4 + 2]),
                    static_cast<float>(raw[i * 4 + 3])
            };
        }
    }
    
    
    void Image::encodePNG(const std::string &path, const std::vector<float4> &in, uint32_t w, uint32_t h) {
        std::vector<uint8_t> raw;
        
        uint32_t size = w * h;
        raw.resize(size * 4);
        for (uint32_t i = 0; i < size; i++) {
            raw[i * 4 + 0] = in[i].x;
            raw[i * 4 + 1] = in[i].y;
            raw[i * 4 + 2] = in[i].z;
            raw[i * 4 + 3] = in[i].w;
        }
        
        uint32_t error = lodepng::encode(path, raw, w, h);
        if (error) {
            std::cerr << "Could not save image to '" << path << "' : " << lodepng_error_text(error) << std::endl;
            exit(EXIT_FAILURE);
        }
    }
}
