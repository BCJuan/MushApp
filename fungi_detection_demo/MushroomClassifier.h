/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   MushroomClassifier.h
 * Author: openvino
 *
 * Created on December 4, 2019, 7:36 PM
 */

#ifndef MUSHROOMCLASSIFIER_H
#define MUSHROOMCLASSIFIER_H


#include <inference_engine.hpp>
#include <string>
#include <vector>
#include "BoundingBox.h"
#include "InputImage.h"

using namespace std;
using namespace InferenceEngine;


class MushroomClassifier {
public:
    MushroomClassifier();
    MushroomClassifier(const MushroomClassifier& orig);
    virtual ~MushroomClassifier();
    
public:
    void load(std::string modelFile);
    void prepareInputs(vector<InputImage> images);
    void prepareOutputs();
    void prepareInference(InferencePlugin& plugin);
    void setInput();
    void runInference();
    void getOutput();
    void setBoxes(vector<BoundingBox>& boxes);
    
private:
    CNNNetwork network;
    OutputsDataMap outputInfo;
    std::vector<std::shared_ptr<unsigned char>> imagesData;
    std::string firstOutputName;
    InferRequest infer_request;
    size_t batchSize;
    vector<BoundingBox> m_boxes;
public:
    
    int resizeW;
    int resizeH;
};

#endif /* MUSHROOMCLASSIFIER_H */

