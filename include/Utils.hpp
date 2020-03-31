#ifndef GPGPU_UTILS_HPP
#define GPGPU_UTILS_HPP

#include <iostream>
#include <sstream>

#include <cuda_runtime.h>

#include <ChronoCPU.hpp>
#include <ChronoGPU.hpp>


namespace equalizer {
    
    class Utils {
        
        public :
            
            static void HandleError(cudaError_t err, const char *file, const int line) {
                if (err != cudaSuccess) {
                    std::stringstream ss;
                    ss << cudaGetErrorString(err) << " (file: " << std::string(file) << " at line: " << line << ")";
                    throw std::runtime_error(ss.str());
                }
            }
    };
    
    
    
    extern bool VERBOSE;
    
    #define HANDLE_CUDA_ERROR(err) (Utils::HandleError(err, __FILE__, __LINE__))
    
    /**
     * Time the execution of 'code' on the host, described by 'description' if VERBOSE is
     * set to true.
     *
     * @param code Executed code.
     * @param description Description of the executed code.
     */
    #define CPU_VERBOSE(code, description) ({\
         {\
             equalizer::ChronoCPU _verbose_chrono;\
             if (equalizer::VERBOSE) {\
                std::cout << description << std::endl;\
                 _verbose_chrono.start();\
             }\
             code;\
             if (equalizer::VERBOSE) {\
                _verbose_chrono.stop();\
                std::cout << "\t-> done in : " << _verbose_chrono.elapsedTime() << " ms (host)" << std::endl;\
             }\
         }\
    })
    
    /**
     * Time the execution of 'code' on the device, described by 'description' if VERBOSE is
     * set to true.
     *
     * If kernel launch are present in 'code', they must be around parenthesis.
     *
     * @param code Executed code.
     * @param description Description of the executed code.
     */
    #define GPU_VERBOSE(code, description) ({\
         {\
             equalizer::ChronoGPU _verbose_chrono;\
             if (equalizer::VERBOSE) {\
                std::cout << description << std::endl;\
                 _verbose_chrono.start();\
             }\
             code;\
             if (equalizer::VERBOSE) {\
                _verbose_chrono.stop();\
                std::cout << "\t-> done in : " << _verbose_chrono.elapsedTime() << " ms (device)" << std::endl;\
             }\
         }\
    })
}

#endif //GPGPU_UTILS_HPP
