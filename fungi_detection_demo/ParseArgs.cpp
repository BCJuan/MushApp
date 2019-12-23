/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   ParseArgs.cpp
 * Author: openvino
 * 
 * Created on December 4, 2019, 1:24 PM
 */

#include "ParseArgs.h"

#include <gflags/gflags.h>
#include <functional>
#include <iostream>
#include <fstream>
#include <random>
#include <string>
#include <memory>
#include <vector>
#include <time.h>
#include <limits>
#include <chrono>
#include <algorithm>

#include <format_reader_ptr.h>
#include <inference_engine.hpp>
#include <ext_list.hpp>

#include <samples/common.hpp>
#include <samples/slog.hpp>
//#include <samples/args_helper.hpp>
#include "object_detection_sample_ssd.h"


ParseArgs::ParseArgs() {
}

ParseArgs::ParseArgs(const ParseArgs& orig) {
}

ParseArgs::~ParseArgs() {
}

/**
* \brief This function show a help message
*/
static void showUsage() {
    std::cout << std::endl;
    std::cout << "object_detection_sample_ssd [OPTION]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << std::endl;
    std::cout << "    -h                      " << help_message << std::endl;
    std::cout << "    -i \"<path>\"             " << image_message << std::endl;
    std::cout << "    -md \"<path>\"             " << detection_model_message << std::endl;
    std::cout << "    -mc \"<path>\"             " << classification_model_message << std::endl;
    std::cout << "      -l \"<absolute_path>\"    " << custom_cpu_library_message << std::endl;
    std::cout << "          Or" << std::endl;
    std::cout << "      -c \"<absolute_path>\"    " << custom_cldnn_message << std::endl;
    std::cout << "    -pp \"<path>\"            " << plugin_path_message << std::endl;
    std::cout << "    -d \"<device>\"           " << target_device_message << std::endl;
    std::cout << "    -pc                     " << performance_counter_message << std::endl;
    std::cout << "    -ni \"<integer>\"         " << iterations_count_message << std::endl;
    std::cout << "    -p_msg                  " << plugin_err_message << std::endl;
}


bool ParseAndCheckCommandLine(int argc, char *argv[]) 
{
    // ---------------------------Parsing and validation of input args--------------------------------------
    gflags::ParseCommandLineNonHelpFlags(&argc, &argv, true);
    
    if (FLAGS_h) 
    {
        showUsage();
        return false;
    }

    if (!FLAGS_silent)
        slog::info << "Parsing input parameters" << slog::endl;

    if (FLAGS_ni < 1) {
        throw std::logic_error("Parameter -ni should be greater than 0 (default: 1)");
    }

    if (FLAGS_i.empty()) {
        throw std::logic_error("Parameter -i is not set");
    }

    if (FLAGS_md.empty()) 
    {
        throw std::logic_error("Parameter -md is not set");
    }

    if (FLAGS_mc.empty()) 
    {
        throw std::logic_error("Parameter -mc is not set");
    }
    
    return true;
}