/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   MushroomClassifier.cpp
 * Author: openvino
 * 
 * Created on December 4, 2019, 7:36 PM
 */

#include "MushroomClassifier.h"
#include "MushroomClassificationResult.h"


#include <gflags/gflags.h>
#include <format_reader_ptr.h>
#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/classification_results.h>

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
DECLARE_uint32(nt);

MushroomClassifier::MushroomClassifier() {
}

MushroomClassifier::MushroomClassifier(const MushroomClassifier& orig) {
}

MushroomClassifier::~MushroomClassifier() {
}

void MushroomClassifier::load(std::string modelFile)
{
    std::string binFileName = fileNameNoExt(modelFile) + ".bin";
    
    if (!FLAGS_silent)
        slog::info << "Loading network files:"
            "\n\t" << modelFile <<
            "\n\t" << binFileName <<
            slog::endl;

    CNNNetReader networkReader;
    /** Read network model **/
    networkReader.ReadNetwork(modelFile);

    /** Extract model name and load weights **/
    networkReader.ReadWeights(binFileName);
    network = networkReader.getNetwork();

    /** Taking information about all topology inputs **/
    InputsDataMap inputInfo = network.getInputsInfo();
    if (inputInfo.size() != 1) 
        throw std::logic_error("Sample supports topologies only with 1 input");

    auto inputInfoItem = *inputInfo.begin();

    /** Specifying the precision and layout of input data provided by the user.
     * This should be called before load of the network to the plugin **/
    inputInfoItem.second->setPrecision(Precision::U8);
    inputInfoItem.second->setLayout(Layout::NCHW);

    resizeW = inputInfoItem.second->getTensorDesc().getDims()[3];
    resizeH = inputInfoItem.second->getTensorDesc().getDims()[2];
}


void MushroomClassifier::prepareInputs(vector<InputImage> images)
{
    if (!FLAGS_silent)
        slog::info << "Preparing input blobs" << slog::endl;

        /** Taking information about all topology inputs **/
        InputsDataMap inputInfo = network.getInputsInfo();
        if (inputInfo.size() != 1) throw std::logic_error("Sample supports topologies only with 1 input");

        auto inputInfoItem = *inputInfo.begin();

        /** Specifying the precision and layout of input data provided by the user.
         * This should be called before load of the network to the plugin **/
        inputInfoItem.second->setPrecision(Precision::U8);
        inputInfoItem.second->setLayout(Layout::NCHW);

        
        for (auto & i : images) 
        {
            imagesData.push_back(i.m_data);
        }
        if (imagesData.empty()) throw std::logic_error("Valid input images were not found!");

        /** Setting batch size using image count **/
        network.setBatchSize(imagesData.size());
        batchSize = network.getBatchSize();
        
        if (!FLAGS_silent)
            slog::info << "Batch size is " << std::to_string(batchSize) << slog::endl;

        // ------------------------------ Prepare output blobs -------------------------------------------------
        if (!FLAGS_silent)
            slog::info << "Preparing output blobs" << slog::endl;

        OutputsDataMap outputInfo(network.getOutputsInfo());
        // BlobMap outputBlobs;
        std::string firstOutputName;

        for (auto & item : outputInfo) {
            if (firstOutputName.empty()) {
                firstOutputName = item.first;
            }
            DataPtr outputData = item.second;
            if (!outputData) {
                throw std::logic_error("output data pointer is not valid");
            }

            item.second->setPrecision(Precision::FP32);
        }

        const SizeVector outputDims = outputInfo.begin()->second->getDims();

        bool outputCorrect = false;
        if (outputDims.size() == 2 /* NC */) {
            outputCorrect = true;
        } else if (outputDims.size() == 4 /* NCHW */) {
            /* H = W = 1 */
            if (outputDims[2] == 1 && outputDims[3] == 1) outputCorrect = true;
        }

        if (!outputCorrect) {
            throw std::logic_error("Incorrect output dimensions for classification model");
        }
}


void MushroomClassifier::prepareOutputs()
{
    if (!FLAGS_silent)
        slog::info << "Preparing output blobs" << slog::endl;

        OutputsDataMap outputInfo(network.getOutputsInfo());
        // BlobMap outputBlobs;
        

        for (auto & item : outputInfo) {
            if (firstOutputName.empty()) {
                firstOutputName = item.first;
            }
            DataPtr outputData = item.second;
            if (!outputData) {
                throw std::logic_error("output data pointer is not valid");
            }

            item.second->setPrecision(Precision::FP32);
        }

        const SizeVector outputDims = outputInfo.begin()->second->getDims();

        bool outputCorrect = false;
        if (outputDims.size() == 2 /* NC */) {
            outputCorrect = true;
        } else if (outputDims.size() == 4 /* NCHW */) {
            /* H = W = 1 */
            if (outputDims[2] == 1 && outputDims[3] == 1) outputCorrect = true;
        }

        if (!outputCorrect) {
            throw std::logic_error("Incorrect output dimensions for classification model");
        }
}

void MushroomClassifier::prepareInference(InferencePlugin& plugin)
{
    
    // -----------------------------------------------------------------------------------------------------
    if (!FLAGS_silent)
        slog::info << "Loading model to the plugin" << slog::endl;

    ExecutableNetwork executable_network = plugin.LoadNetwork(network, {});
    //inputInfoItem.second = {};
    outputInfo = {};
//    network = {};
//    networkReader = {};
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 5. Create infer request -------------------------------------------------
    infer_request = executable_network.CreateInferRequest();
}

void MushroomClassifier::setInput()
{
    if (!FLAGS_silent)
        slog::info << "setting Input" << slog::endl;
    
    InputsDataMap inputInfo = network.getInputsInfo();
    
    if (!FLAGS_silent)
        printf("Info Size: %d\n", (int) inputInfo.size());
    
    for (const auto & item : inputInfo) 
    {
            /** Creating input blob **/
            Blob::Ptr input = infer_request.GetBlob(item.first);

            /** Filling input tensor with images. First b channel, then g and r channels **/
            size_t num_channels = input->getTensorDesc().getDims()[1];
            size_t image_size = input->getTensorDesc().getDims()[2] * input->getTensorDesc().getDims()[3];

            if (!FLAGS_silent)
                printf("Img channels: %d size: %d*%d\n", (int)num_channels, 
                    (int) input->getTensorDesc().getDims()[2], (int) input->getTensorDesc().getDims()[3]);
            
            auto data = input->buffer().as<PrecisionTrait<Precision::U8>::value_type*>();

            /** Iterate over all input images **/
            for (size_t image_id = 0; image_id < imagesData.size(); ++image_id) {
                /** Iterate over all pixel in image (b,g,r) **/
                for (size_t pid = 0; pid < image_size; pid++) {
                    /** Iterate over all channels **/
                    for (size_t ch = 0; ch < num_channels; ++ch) {
                        /**          [images stride + channels stride + pixel id ] all in bytes            **/
                        data[image_id * image_size * num_channels + ch * image_size + pid ] = imagesData.at(image_id).get()[pid*num_channels + ch];
                    }
                }
            }
        }
        inputInfo = {};
}

void MushroomClassifier::runInference()
{
    if (!FLAGS_silent)
        slog::info << "Starting inference (" << FLAGS_ni << " iterations)" << slog::endl;

        typedef std::chrono::high_resolution_clock Time;
        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        typedef std::chrono::duration<float> fsec;

        double total = 0.0;
        /** Start inference & calc performance **/
        for (size_t iter = 0; iter < FLAGS_ni; ++iter) {
            auto t0 = Time::now();
            infer_request.Infer();
            auto t1 = Time::now();
            fsec fs = t1 - t0;
            ms d = std::chrono::duration_cast<ms>(fs);
            total += d.count();
        }
}

void MushroomClassifier::setBoxes(vector<BoundingBox>& boxes)
{
    m_boxes = boxes;    
}

void MushroomClassifier::getOutput()
{
    if (!FLAGS_silent)
        slog::info << "Processing output blobs" << slog::endl;
    
    if (!FLAGS_silent)
        printf("Batch size: %d\n", (int) batchSize);

        const Blob::Ptr output_blob = infer_request.GetBlob(firstOutputName);

        /** Validating -nt value **/
        const size_t resultsCnt = output_blob->size() / batchSize;
        if (FLAGS_nt > resultsCnt || FLAGS_nt < 1) {
            slog::warn << "-nt " << FLAGS_nt << " is not available for this network (-nt should be less than " \
                      << resultsCnt+1 << " and more than 0)\n            will be used maximal value : " << resultsCnt;
            FLAGS_nt = resultsCnt;
        }

        /** Read labels from file (e.x. AlexNet.labels) **/
        std::string labelFileName = fileNameNoExt(FLAGS_mc) + ".labels";
        std::vector<std::string> labels;

        std::ifstream inputFile;
        inputFile.open(labelFileName, std::ios::in);
        if (inputFile.is_open()) {
            std::string strLine;
            while (std::getline(inputFile, strLine)) {
                trim(strLine);
                labels.push_back(strLine);
            }
        }
std::vector<std::string> imageNames;
for (size_t i=0; i < batchSize; i++)
    imageNames.push_back("single input");

    MushroomClassificationResult classificationResult(output_blob, imageNames,
                                                  batchSize, FLAGS_nt,
                                                  labels);
    
    classificationResult.m_boxes = m_boxes;
        classificationResult.print();

        
}