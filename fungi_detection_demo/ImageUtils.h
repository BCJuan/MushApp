/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   ImageUtils.h
 * Author: openvino
 *
 * Created on December 4, 2019, 9:06 PM
 */

#ifndef IMAGEUTILS_H
#define IMAGEUTILS_H

#include "InputImage.h"


#include <inference_engine.hpp>
#include <string>
#include <vector>
#include "BoundingBox.h"
#include "InputImage.h"

using namespace std;
using namespace InferenceEngine;

class ImageUtils {
public:
    ImageUtils();
    ImageUtils(const ImageUtils& orig);
    virtual ~ImageUtils();
    
public:
    void loadImages(std::vector<std::string>& filenames, std::vector<InputImage>& images, InputInfo::Ptr& inputInfo);
    void cropAndResizeBoxes(InputImage& base, std::vector<BoundingBox>& boxes, std::vector<InputImage>& cropImages, int rw, int rh);
    
private:

};

#endif /* IMAGEUTILS_H */

