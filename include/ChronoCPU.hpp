/*
* File: chronoCPU.hpp
* Author: Maxime MARIA
*/

#ifndef __CHRONO_CPU_HPP
#define __CHRONO_CPU_HPP

#ifdef _WIN32
#include <windows.h>
#else

#include <ctime>


#endif

namespace equalizer {
    
    class ChronoCPU {
        private:
            #ifdef _WIN32
            LARGE_INTEGER m_frequency;
            LARGE_INTEGER m_start;
            LARGE_INTEGER m_stop;
            #else
            timeval m_start;
            timeval m_stop;
            #endif
            
            bool m_started;
        
        public:
            ChronoCPU();
            
            ~ChronoCPU();
            
            void start();
            
            void stop();
            
            float elapsedTime();
    };
}

#endif

