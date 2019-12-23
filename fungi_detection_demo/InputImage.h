/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   InputImage.h
 * Author: openvino
 *
 * Created on December 4, 2019, 9:05 PM
 */

#ifndef INPUTIMAGE_H
#define INPUTIMAGE_H


#include <inference_engine.hpp>
#include <string>
#include <vector>
#include "BoundingBox.h"
#include "InputImage.h"
#include <samples/ocv_common.hpp>

using namespace std;
using namespace InferenceEngine;

class InputImage {
public:
    InputImage(cv::Mat& img, std::shared_ptr<unsigned char> original, std::shared_ptr<unsigned char> data, int w, int h);
    InputImage(const InputImage& orig);
    virtual ~InputImage();

public:
    cv::Mat m_img;
    std::shared_ptr<unsigned char> m_original;
    std::shared_ptr<unsigned char> m_data;
    int m_w;
    int m_h;
};

#endif /* INPUTIMAGE_H */

