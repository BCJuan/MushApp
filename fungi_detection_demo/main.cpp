// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

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


#include <inference_engine.hpp>
#include <ext_list.hpp>

#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/args_helper.hpp>
//#include "object_detection_sample_ssd.h"

#include "ParseArgs.h"
#include "MushroomDetector.h"
#include "BoundingBox.h"
#include "MushroomClassifier.h"
#include "ImageUtils.h"
#include "InputImage.h"
//#include "object_detection_sample_ssd.h"

using namespace InferenceEngine;

ConsoleErrorListener error_listener;

#define BBOX_THICKNESS 2

DECLARE_bool(silent);
DECLARE_bool(h);
DECLARE_string(i);
DECLARE_string(md);
DECLARE_string(mc);
DECLARE_string(pp);
DECLARE_string(d);
DECLARE_bool(pc);
DECLARE_string(c);
DECLARE_string(l);
DECLARE_uint32(ni);
DECLARE_bool(p_msg);

void addBoundingBoxes(unsigned char *data, size_t height, size_t width, 
        std::vector<BoundingBox>& boxes, int thickness = 1) ;

/**
* \brief The entry point for the Inference Engine object_detection sample application
* \file object_detection_sample_ssd/main.cpp
* \example object_detection_sample_ssd/main.cpp
*/
int main(int argc, char *argv[]) 
{
    
    try 
    {
        
        /** This sample covers certain topology and cannot be generalized for any object detection one **/
        

        // --------------------------- 1. Parsing and validation of input args ---------------------------------
        if (!ParseAndCheckCommandLine(argc, argv)) {
            return 0;
        }
        
        if (!FLAGS_silent)
        {
            slog::info << "InferenceEngine: " << GetInferenceEngineVersion() << "\n";
        }
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 2. Read input -----------------------------------------------------------
        /** This vector stores paths to the processed images **/
        std::vector<std::string> imagesNames;
        parseInputFilesArguments(imagesNames);
        if (imagesNames.empty())
            throw std::logic_error("No suitable images were found");
        
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 3. Load Plugin for inference engine -------------------------------------
        if (!FLAGS_silent)
            slog::info << "Loading plugin" << slog::endl;
        
        InferencePlugin plugin = PluginDispatcher({ FLAGS_pp }).getPluginByDevice(FLAGS_d);
        if (FLAGS_p_msg) 
        {
            static_cast<InferenceEngine::InferenceEnginePluginPtr>(plugin)->SetLogCallback(error_listener);
        }

        /*If CPU device, load default library with extensions that comes with the product*/
        if (FLAGS_d.find("CPU") != std::string::npos) 
        {
            /**
            * cpu_extensions library is compiled from "extension" folder containing
            * custom MKLDNNPlugin layer implementations. These layers are not supported
            * by mkldnn, but they can be useful for inferring custom topologies.
            **/
            plugin.AddExtension(std::make_shared<Extensions::Cpu::CpuExtensions>());
        }

        if (!FLAGS_l.empty()) 
        {
            // CPU(MKLDNN) extensions are loaded as a shared library and passed as a pointer to base extension
            IExtensionPtr extension_ptr = make_so_pointer<IExtension>(FLAGS_l);
            plugin.AddExtension(extension_ptr);
            if (!FLAGS_silent)
                slog::info << "CPU Extension loaded: " << FLAGS_l << slog::endl;
        }

        if (!FLAGS_c.empty()) 
        {
            // clDNN Extensions are loaded from an .xml description and OpenCL kernel files
            plugin.SetConfig({ { PluginConfigParams::KEY_CONFIG_FILE, FLAGS_c } });
            
            if (!FLAGS_silent)
                slog::info << "GPU Extension loaded: " << FLAGS_c << slog::endl;
        }

        /** Setting plugin parameter for per layer metrics **/
        if (FLAGS_pc) {
            plugin.SetConfig({ { PluginConfigParams::KEY_PERF_COUNT, PluginConfigParams::YES } });
        }

        /** Printing plugin version **/
        if (!FLAGS_silent)
            printPluginVersion(plugin, std::cout);
        
        MushroomDetector detector;
        detector.load(FLAGS_md);
        detector.prepareInputs();
        detector.prepareOutputs();
        detector.prepareInference(plugin);
        
        MushroomClassifier classifier;
        classifier.load(FLAGS_mc);
        
        
        std::vector<InputImage> images;
        ImageUtils imageUtils;
        imageUtils.loadImages(imagesNames, images, detector.inputInfo);
        
        detector.setInput(&images);
        detector.runInference();
        
        vector<BoundingBox> result;
        
        detector.getOutput(result);

        std::vector<InputImage> cropImages;
        imageUtils.cropAndResizeBoxes(images[0], result, cropImages, classifier.resizeW, classifier.resizeH);
        
        classifier.prepareInputs(cropImages);
        classifier.prepareOutputs();
        classifier.prepareInference(plugin);
        
        classifier.setBoxes(result);
        classifier.setInput();
        classifier.runInference();
        classifier.getOutput();
        
        
        // -----------------------------------------------------------------------------------------------------

        // --------------------------- 11. Process output -------------------------------------------------------
//        size_t batchSize = 1; // number of input images
//
//        for (size_t batch_id = 0; batch_id < batchSize; ++batch_id) 
//        {
//            InputImage img = images[batch_id];
//            
//            addBoundingBoxes(img.m_original.get(), img.m_h, img.m_w, result, BBOX_THICKNESS);
//            const std::string image_path = "out_" + std::to_string(batch_id) + ".bmp";
//            
//            if (writeOutputBmp(image_path, img.m_original.get(), img.m_h, img.m_w)) 
//            {
//                slog::info << "Image " + image_path + " created!" << slog::endl;
//            }
//            else 
//            {
//                throw std::logic_error(std::string("Can't create a file: ") + image_path);
//            }
//        }
        // -----------------------------------------------------------------------------------------------------
        if (!FLAGS_silent)
        {
            std::cout << std::endl << "total inference time: " << detector.total << std::endl;
            std::cout << "Average running time of one iteration: " << detector.total / static_cast<double>(FLAGS_ni) << " ms" << std::endl;
            std::cout << std::endl << "Throughput: " << 1000 * static_cast<double>(FLAGS_ni) * 1 / detector.total << " FPS" << std::endl;
            std::cout << std::endl;
        }

        /** Show performance results **/
        if (FLAGS_pc) 
        {
            printPerformanceCounts(detector.infer_request, std::cout);
        }
    }
    catch (const std::exception& error) {
        slog::err << error.what() << slog::endl;
        return 1;
    }
    catch (...) {
        slog::err << "Unknown/internal exception happened." << slog::endl;
        return 1;
    }

    slog::info << "Execution successful" << slog::endl;
    return 0;
}


void addBoundingBoxes(unsigned char *data, size_t height, size_t width, 
        std::vector<BoundingBox>& boxes, int thickness ) 
{
    std::vector<Color> colors = {  // colors to be used for bounding boxes
        { 128, 64,  128 },
        { 232, 35,  244 },
        { 70,  70,  70 },
        { 156, 102, 102 },
        { 153, 153, 190 },
        { 153, 153, 153 },
        { 30,  170, 250 },
        { 0,   220, 220 },
        { 35,  142, 107 },
        { 152, 251, 152 },
        { 180, 130, 70 },
        { 60,  20,  220 },
        { 0,   0,   255 },
        { 142, 0,   0 },
        { 70,  0,   0 },
        { 100, 60,  0 },
        { 90,  0,   0 },
        { 230, 0,   0 },
        { 32,  11,  119 },
        { 0,   74,  111 },
        { 81,  0,   81 }
    };
    
//    if (boxes.size() % 4 != 0 || boxes.size() / 4 != classes.size()) {
//        return;
//    }

    for (size_t i = 0; i < boxes.size(); i++) 
    {
        BoundingBox box = boxes[i];
        
        int x = box.m_x;
        int y = box.m_y;
        int w = box.m_w;
        int h = box.m_h;

        int cls = box.m_label % colors.size();  // color of a bounding box line

        if (x < 0) x = 0;
        if (y < 0) y = 0;
        if (w < 0) w = 0;
        if (h < 0) h = 0;

        if (static_cast<std::size_t>(x) >= width) { x = width - 1; w = 0; thickness = 1; }
        if (static_cast<std::size_t>(y) >= height) { y = height - 1; h = 0; thickness = 1; }

        if (static_cast<std::size_t>(x + w) >= width) { w = width - x - 1; }
        if (static_cast<std::size_t>(y + h) >= height) { h = height - y - 1; }

        thickness = std::min(std::min(thickness, w / 2 + 1), h / 2 + 1);

        size_t shift_first;
        size_t shift_second;
        for (int t = 0; t < thickness; t++) {
            shift_first = (y + t) * width * 3;
            shift_second = (y + h - t) * width * 3;
            for (int ii = x; ii < x + w + 1; ii++) {
                data[shift_first + ii * 3] = colors.at(cls).red();
                data[shift_first + ii * 3 + 1] = colors.at(cls).green();
                data[shift_first + ii * 3 + 2] = colors.at(cls).blue();
                data[shift_second + ii * 3] = colors.at(cls).red();
                data[shift_second + ii * 3 + 1] = colors.at(cls).green();
                data[shift_second + ii * 3 + 2] = colors.at(cls).blue();
            }
        }

        for (int t = 0; t < thickness; t++) {
            shift_first = (x + t) * 3;
            shift_second = (x + w - t) * 3;
            for (int ii = y; ii < y + h + 1; ii++) {
                data[shift_first + ii * width * 3] = colors.at(cls).red();
                data[shift_first + ii * width * 3 + 1] = colors.at(cls).green();
                data[shift_first + ii * width * 3 + 2] = colors.at(cls).blue();
                data[shift_second + ii * width * 3] = colors.at(cls).red();
                data[shift_second + ii * width * 3 + 1] = colors.at(cls).green();
                data[shift_second + ii * width * 3 + 2] = colors.at(cls).blue();
            }
        }
    }
}