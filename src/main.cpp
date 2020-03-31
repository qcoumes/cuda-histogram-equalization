#include <getopt.h>
#include <iostream>
#include <vector>

#include <Equalizer.hpp>
#include <Utils.hpp>
#include <Image.hpp>


bool equalizer::VERBOSE = false;


static void usage() {
    std::cout << std::endl;
    std::cout << "Usage:" << std::endl;
    std::cout << "\t  equalizer [OPTIONS] INPUT_IMAGE OUTPUT_IMAGE" << std::endl << std::endl;
}


static void help() {
    usage();
    
    std::cout << "Adjust image's contrast by via the histogram equalization method." << std::endl << std::endl;
    
    std::cout << "Mandatory arguments: " << std::endl << std::endl;
    
    std::cout << "\t INPUT_IMAGE" << std::endl;
    std::cout << "\t\t Path to the image the histogram equalization will be applied on." << std::endl << std::endl;
    
    std::cout << "\t OUTPUT_IMAGE" << std::endl;
    std::cout << "\t\t Path to the output equalized imag.e" << std::endl << std::endl;
    
    std::cout << "Optional arguments (must be before mandatory arguments): " << std::endl << std::endl;
    
    std::cout << "\t -g" << std::endl;
    std::cout << "\t\t Execute the program on the GPU (default) use with '-c' to execute on both the GPU and the CPU."
              << std::endl;
    
    std::cout << "\t -c" << std::endl;
    std::cout << "\t\t Execute the program on the CPU use with '-g' to execute on both the GPU and the CPU."
              << std::endl;
    
    std::cout << "\t -v" << std::endl;
    std::cout << "\t\t Displays information and the time taken by each step of the execution." << std::endl;
    
    std::cout << "\t -h" << std::endl;
    std::cout << "\t\t Displays this help" << std::endl << std::endl;
}


int main(int argc, char **argv) {
    bool cpu = false, gpu = false;
    std::string inputPath, outputPath;
    char c;
    
    while ((c = static_cast<char>(getopt(argc, argv, "vcgh"))) != -1) {
        switch (c) {
            case 'g':
                gpu = true;
                break;
            case 'c':
                cpu = true;
                break;
            case 'v':
                equalizer::VERBOSE = true;
                break;
            case 'h':
                help();
                exit(EXIT_SUCCESS);
            case '?':
                std::cerr << "equalizer: invalid option -- '" << optopt << "'" << std::endl
                          << "Try 'equalizer -h' for more information." << std::endl;
                exit(EXIT_FAILURE);
            default:
                break;
        }
    }
    
    if (argc - optind <= 0) {
        std::cerr << "equalizer: missing file operand" << std::endl
                  << "Try 'equalizer -h' for more information." << std::endl;
        exit(EXIT_FAILURE);
    }
    else if (argc - optind == 1) {
        std::cerr << "missing destination file operand after '" << argv[argc - 1] << "'" << std::endl
                  << "Try 'equalizer -h' for more information." << std::endl;
        exit(EXIT_FAILURE);
    }
    
    inputPath = argv[argc - 2];
    outputPath = argv[argc - 1];
    
    std::vector<float4> input, output;
    uint32_t width = 0, height = 0;
    
    equalizer::Image::decodePNG(inputPath, input, width, height);
    output.resize(input.size());
    if (cpu) {
        equalizer::equalizeCpu(input.data(), output.data(), width, height);
    }
    if (gpu || !cpu) {
        equalizer::equalizeGpu(input.data(), output.data(), width, height);
    }
    equalizer::Image::encodePNG(outputPath, output, width, height);
}
