/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   MushroomDetector.h
 * Author: openvino
 *
 * Created on December 4, 2019, 2:30 PM
 */

#ifndef MUSHROOMDETECTOR_H
#define MUSHROOMDETECTOR_H

#include <inference_engine.hpp>
#include <string>
#include <vector>
#include "BoundingBox.h"
#include "InputImage.h"

using namespace std;
using namespace InferenceEngine;

/**
 * Single image 
 */
class MushroomDetector {
public:
    MushroomDetector();
    MushroomDetector(const MushroomDetector& orig);
    virtual ~MushroomDetector();
    
public:
    void load(std::string model);
    void prepareInputs();
    void prepareOutputs();
    void prepareInference(InferencePlugin& plugin);
    void setInput(std::vector<InputImage>* images);
    void runInference();
    void getOutput(vector<BoundingBox>& result);
    
private:
    CNNNetwork network;
    int maxProposalCount;
    int objectSize;

    size_t batchSize;
    
    std::string imageInputName;
    std::string imInfoInputName;
    std::string outputName;
    
public:
    std::vector<InputImage>* m_images;
//    std::vector<std::shared_ptr<unsigned char>> imagesData;
//    std::vector<std::shared_ptr<unsigned char>> originalImagesData;
//    std::vector<size_t> imageWidths;
//    std::vector<size_t> imageHeights;
    double total;
    InferRequest infer_request;
    InputInfo::Ptr inputInfo;
};

#endif /* MUSHROOMDETECTOR_H */

