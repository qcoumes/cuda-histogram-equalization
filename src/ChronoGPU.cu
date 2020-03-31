/*
* File: chronoGPU.cu
* Author: Maxime MARIA
*/

#include <ChronoGPU.hpp>

#include <Utils.hpp>


namespace equalizer {
    
    ChronoGPU::ChronoGPU() :
            m_started(false) {
        HANDLE_CUDA_ERROR(cudaEventCreate(&m_start));
        HANDLE_CUDA_ERROR(cudaEventCreate(&m_end));
    }
    
    
    ChronoGPU::~ChronoGPU() {
        if (m_started) {
            stop();
            std::cerr << "ChronoGPU::~ChronoGPU(): chrono wasn't turned off!" << std::endl;
        }
        HANDLE_CUDA_ERROR(cudaEventDestroy(m_start));
        HANDLE_CUDA_ERROR(cudaEventDestroy(m_end));
    }
    
    
    void ChronoGPU::start() {
        if (!m_started) {
            HANDLE_CUDA_ERROR(cudaEventRecord(m_start, 0));
            m_started = true;
        }
        else {
            std::cerr << "ChronoGPU::start(): chrono is already started!" << std::endl;
        }
    }
    
    
    void ChronoGPU::stop() {
        if (m_started) {
            HANDLE_CUDA_ERROR(cudaEventRecord(m_end, 0));
            HANDLE_CUDA_ERROR(cudaEventSynchronize(m_end));
            m_started = false;
        }
        else {
            std::cerr << "ChronoGPU::stop(): chrono wasn't started!" << std::endl;
        }
    }
    
    
    float ChronoGPU::elapsedTime() {
        float time = 0.f;
        HANDLE_CUDA_ERROR(cudaEventElapsedTime(&time, m_start, m_end));
        return time;
    }
}
