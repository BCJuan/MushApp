/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   MushroomDetector.cpp
 * Author: openvino
 * 
 * Created on December 4, 2019, 2:30 PM
 */

#include "MushroomDetector.h"
#include "InputImage.h"

#include <gflags/gflags.h>
#include <format_reader_ptr.h>
#include <samples/common.hpp>
#include <samples/slog.hpp>

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

MushroomDetector::MushroomDetector() {
}

MushroomDetector::MushroomDetector(const MushroomDetector& orig) {
}

MushroomDetector::~MushroomDetector() {
}

void MushroomDetector::load(std::string modelFile)
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
}


void MushroomDetector::prepareInputs()
{
    if (!FLAGS_silent)
        slog::info << "Preparing input blobs" << slog::endl;

    /** Taking information about all topology inputs **/
    InputsDataMap inputsInfo(network.getInputsInfo());

        /** SSD network has one input and one output **/
        if (inputsInfo.size() != 1 && inputsInfo.size() != 2) 
            throw std::logic_error("Sample supports topologies only with 1 or 2 inputs");

        /**
         * Some networks have SSD-like output format (ending with DetectionOutput layer), but
         * having 2 inputs as Faster-RCNN: one for image and one for "image info".
         *
         * Although object_datection_sample_ssd's main task is to support clean SSD, it could score
         * the networks with two inputs as well. For such networks imInfoInputName will contain the "second" input name.
         */
        
        

        inputInfo = nullptr;

        SizeVector inputImageDims;
        /** Stores input image **/

        /** Iterating over all input blobs **/
        for (auto & item : inputsInfo) 
        {
            /** Working with first input tensor that stores image **/
            if (item.second->getInputData()->getTensorDesc().getDims().size() == 4) 
            {
                imageInputName = item.first;

                inputInfo = item.second;

                if (!FLAGS_silent)
                    slog::info << "Batch size is " << std::to_string(network.getBatchSize()) << slog::endl;

                /** Creating first input blob **/
                Precision inputPrecision = Precision::U8;
                item.second->setPrecision(inputPrecision);
            } 
            else if (item.second->getInputData()->getTensorDesc().getDims().size() == 2) 
            {
                imInfoInputName = item.first;

                Precision inputPrecision = Precision::FP32;
                item.second->setPrecision(inputPrecision);
                if ((item.second->getTensorDesc().getDims()[1] != 3 && item.second->getTensorDesc().getDims()[1] != 6)) {
                    throw std::logic_error("Invalid input info. Should be 3 or 6 values length");
                }
            }
        }

        if (inputInfo == nullptr) 
        {
            inputInfo = inputsInfo.begin()->second;
        }
}


void MushroomDetector::prepareOutputs()
{
    if (!FLAGS_silent)
        slog::info << "Preparing output blobs" << slog::endl;

    OutputsDataMap outputsInfo(network.getOutputsInfo());

    
    DataPtr outputInfo;
    for (const auto& out : outputsInfo) 
    {
        if (out.second->creatorLayer.lock()->type == "DetectionOutput") 
        {
            outputName = out.first;
            outputInfo = out.second;
        }
    }

    if (outputInfo == nullptr) 
    {
        throw std::logic_error("Can't find a DetectionOutput layer in the topology");
    }

    const SizeVector outputDims = outputInfo->getTensorDesc().getDims();

    maxProposalCount = outputDims[2];
    objectSize = outputDims[3];

    if (objectSize != 7) 
    {
        throw std::logic_error("Output item should have 7 as a last dimension");
    }

    if (outputDims.size() != 4) 
    {
        throw std::logic_error("Incorrect output dimensions for SSD model");
    }

    /** Set the precision of output data provided by the user, should be called before load of the network to the plugin **/
    outputInfo->setPrecision(Precision::FP32);
}

void MushroomDetector::prepareInference(InferencePlugin& plugin)
{
    // --------------------------- 7. Loading model to the plugin ------------------------------------------
    if (!FLAGS_silent)
        slog::info << "Loading model to the plugin" << slog::endl;

    ExecutableNetwork executable_network = plugin.LoadNetwork(network, {});
    // -----------------------------------------------------------------------------------------------------

    // --------------------------- 8. Create infer request -------------------------------------------------
    infer_request = executable_network.CreateInferRequest();
    // -----------------------------------------------------------------------------------------------------

}

void MushroomDetector::setInput(std::vector<InputImage>* images)
{
    // --------------------------- 9. Prepare input --------------------------------------------------------
        /** Collect images data ptrs **/
    m_images = images;
        
//        for (auto & i : images) 
//        {
//            
//            FormatReader::ReaderPtr reader(i.c_str());
//            if (reader.get() == nullptr) 
//            {
//                slog::warn << "Image " + i + " cannot be read!" << slog::endl;
//                continue;
//            }
//            /** Store image data **/
//            std::shared_ptr<unsigned char> originalData(reader->getData());
//            std::shared_ptr<unsigned char> data(reader->getData(inputInfo->getTensorDesc().getDims()[3], inputInfo->getTensorDesc().getDims()[2]));
//            
//            if (data.get() != nullptr) 
//            {
//                originalImagesData.push_back(originalData);
//                imagesData.push_back(data);
//                imageWidths.push_back(reader->width());
//                imageHeights.push_back(reader->height());
//            }
//        }
        
        if (images->empty())
            throw std::logic_error("Valid input images were not found!");

        batchSize = network.getBatchSize();
        
        if (!FLAGS_silent)
            slog::info << "Batch size is " << std::to_string(batchSize) << slog::endl;
        
        if (batchSize != images->size()) 
        {
            if (!FLAGS_silent)
                slog::warn << "Number of images " + std::to_string(images->size()) + \
                    " doesn't match batch size " + std::to_string(batchSize) << slog::endl;
            batchSize = std::min(batchSize, images->size());
            
            if (!FLAGS_silent)
                slog::warn << "Number of images to be processed is "<< std::to_string(batchSize) << slog::endl;
        }

        /** Creating input blob **/
        Blob::Ptr imageInput = infer_request.GetBlob(imageInputName);

        /** Filling input tensor with images. First b channel, then g and r channels **/
        size_t num_channels = imageInput->getTensorDesc().getDims()[1];
        size_t image_size = imageInput->getTensorDesc().getDims()[3] * imageInput->getTensorDesc().getDims()[2];

        unsigned char* data = static_cast<unsigned char*>(imageInput->buffer());

        /** Iterate over all input images **/
        for (size_t image_id = 0; image_id < std::min(images->size(), batchSize); ++image_id) 
        {
            /** Iterate over all pixel in image (b,g,r) **/
            for (size_t pid = 0; pid < image_size; pid++) {
                /** Iterate over all channels **/
                for (size_t ch = 0; ch < num_channels; ++ch) 
                {
                    /**          [images stride + channels stride + pixel id ] all in bytes            **/
                    data[image_id * image_size * num_channels + ch * image_size + pid] = m_images->at(image_id).m_data.get()[pid*num_channels + ch];
                }
            }
        }

        if (imInfoInputName != "") 
        {
            InputsDataMap inputsInfo(network.getInputsInfo());
            
            Blob::Ptr input2 = infer_request.GetBlob(imInfoInputName);
            auto imInfoDim = inputsInfo.find(imInfoInputName)->second->getTensorDesc().getDims()[1];

            /** Fill input tensor with values **/
            float *p = input2->buffer().as<PrecisionTrait<Precision::FP32>::value_type*>();

            for (size_t image_id = 0; image_id < std::min(m_images->size(), batchSize); ++image_id) 
            {
                p[image_id * imInfoDim + 0] = static_cast<float>(inputsInfo[imageInputName]->getTensorDesc().getDims()[2]);
                p[image_id * imInfoDim + 1] = static_cast<float>(inputsInfo[imageInputName]->getTensorDesc().getDims()[3]);
                for (size_t k = 2; k < imInfoDim; k++) 
                {
                    p[image_id * imInfoDim + k] = 1.0f;  // all scale factors are set to 1.0
                }
            }
        }
        // -----------------------------------------------------------------------------------------------------

}

void MushroomDetector::runInference()
{
    // --------------------------- 10. Do inference ---------------------------------------------------------
    if (!FLAGS_silent)
        slog::info << "Start inference (" << FLAGS_ni << " iterations)" << slog::endl;

        typedef std::chrono::high_resolution_clock Time;
        typedef std::chrono::duration<double, std::ratio<1, 1000>> ms;
        typedef std::chrono::duration<float> fsec;

        total = 0.0;
        /** Start inference & calc performance **/
        for (size_t iter = 0; iter < FLAGS_ni; ++iter) 
        {
            auto t0 = Time::now();
            infer_request.Infer();
            auto t1 = Time::now();
            fsec fs = t1 - t0;
            ms d = std::chrono::duration_cast<ms>(fs);
            total += d.count();
        }
}


void MushroomDetector::getOutput(vector<BoundingBox>& result)
{
    if (!FLAGS_silent)
        slog::info << "Processing output blobs" << slog::endl;

    const Blob::Ptr output_blob = infer_request.GetBlob(outputName);
    const float* detection = static_cast<PrecisionTrait<Precision::FP32>::value_type*>(output_blob->buffer());

    std::vector<std::vector<int> > boxes(batchSize);
    std::vector<std::vector<int> > classes(batchSize);

    /* Each detection has image_id that denotes processed image */
    for (int curProposal = 0; curProposal < maxProposalCount; curProposal++) 
    {
        auto image_id = static_cast<int>(detection[curProposal * objectSize + 0]);
        if (image_id < 0) 
        {
            break;
        }

        float confidence = detection[curProposal * objectSize + 2];
        auto label = static_cast<int>(detection[curProposal * objectSize + 1]);
        auto xmin = static_cast<int>(detection[curProposal * objectSize + 3] * (*m_images)[image_id].m_w);
        auto ymin = static_cast<int>(detection[curProposal * objectSize + 4] * (*m_images)[image_id].m_h);
        auto xmax = static_cast<int>(detection[curProposal * objectSize + 5] * (*m_images)[image_id].m_w);
        auto ymax = static_cast<int>(detection[curProposal * objectSize + 6] * (*m_images)[image_id].m_h);

        if (!FLAGS_silent)
            std::cout << "[" << curProposal << "," << label << "] element, prob = " << confidence <<
            "    (" << xmin << "," << ymin << ")-(" << xmax << "," << ymax << ")" << " batch id : " << image_id;

        if (confidence > 0.5) 
        {
            /** Drawing only objects with >50% probability **/
            classes[image_id].push_back(label);
            boxes[image_id].push_back(xmin);
            boxes[image_id].push_back(ymin);
            boxes[image_id].push_back(xmax - xmin);
            boxes[image_id].push_back(ymax - ymin);
            
            if (!FLAGS_silent)
                std::cout << " WILL BE PRINTED!";
            
            result.push_back(BoundingBox(xmin, ymin, xmax - xmin, ymax - ymin, label));
        }
        std::cout << std::endl;
    }
}