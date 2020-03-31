/*
* File: chronoGPU.hpp
* Author: Maxime MARIA
*/

#ifndef CHRONO_GPU_HPP
#define CHRONO_GPU_HPP

#include <driver_types.h>


namespace equalizer {
    
    class ChronoGPU {
        private:
            cudaEvent_t m_start;
            cudaEvent_t m_end;
            bool m_started;
        
        public:
            
            ChronoGPU();
            
            ~ChronoGPU();
            
            void start();
            
            void stop();
            
            float elapsedTime();
    };
}

#endif // CHRONO_GPU_HPP

