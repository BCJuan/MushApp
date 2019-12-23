/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   ImageUtils.cpp
 * Author: openvino
 * 
 * Created on December 4, 2019, 9:06 PM
 */

#include "ImageUtils.h"

#include <gflags/gflags.h>
#include <format_reader_ptr.h>
#include <samples/common.hpp>
#include <samples/slog.hpp>
#include <samples/ocv_common.hpp>

DECLARE_bool(silent);

ImageUtils::ImageUtils() {
}

ImageUtils::ImageUtils(const ImageUtils& orig) {
}

ImageUtils::~ImageUtils() {
}

/**
 * 
 * @param filenames
 * @param images
 * @param inputInfo
 */
void ImageUtils::loadImages(std::vector<std::string>& filenames, std::vector<InputImage>& images, InputInfo::Ptr& inputInfo)
{
    // --------------------------- 9. Prepare input --------------------------------------------------------
        /** Collect images data ptrs **/
        
        for (auto & filename : filenames) 
        {
            cv::Mat img = cv::imread(filename);
            
            int w = img.size().width;
            int h = img.size().height;
            size_t size = w*h*img.channels();
            
            std::shared_ptr<unsigned char> originalData;
            
            originalData.reset(new unsigned char[size], std::default_delete<unsigned char[]>());
            for (size_t id = 0; id < size; ++id) 
            {
                originalData.get()[id] = img.data[id];
            }
                       
            
            int rw = inputInfo->getTensorDesc().getDims()[3];
            int rh = inputInfo->getTensorDesc().getDims()[2];
            
            cv::Mat resized(img); 
            cv::resize(img, resized, cv::Size(rw, rh));
            
            size = rw*rh*resized.channels();
            
            std::shared_ptr<unsigned char> data;
            
            data.reset(new unsigned char[size], std::default_delete<unsigned char[]>());
            for (size_t id = 0; id < size; ++id) 
            {
                data.get()[id] = resized.data[id];
            }
            
            InputImage inputImg(img, originalData, data, w, h);
            images.push_back(inputImg);
            
//            FormatReader::ReaderPtr reader(filename.c_str());
//            if (reader.get() == nullptr) 
//            {
//                slog::warn << "Image " + filename + " cannot be read!" << slog::endl;
//                continue;
//            }
//            /** Store image data **/
//            std::shared_ptr<unsigned char> originalData(reader->getData());
//            std::shared_ptr<unsigned char> data(reader->getData(inputInfo->getTensorDesc().getDims()[3], inputInfo->getTensorDesc().getDims()[2]));
//            
//            if (data.get() != nullptr) 
//            {
//                InputImage img(originalData, data, reader->width(), reader->height());
//                images.push_back(img);
////                originalImagesData.push_back(originalData);
////                imagesData.push_back(data);
////                imageWidths.push_back(reader->width());
////                imageHeights.push_back(reader->height());
//            }
        }
        
        
}

void ImageUtils::cropAndResizeBoxes(InputImage& base, std::vector<BoundingBox>& boxes, std::vector<InputImage>& cropImages, int rw, int rh)
{
    for (size_t i=0; i < boxes.size(); i++)
    {
        BoundingBox box = boxes[i];
        
        cv::Rect roi;
        roi.x = box.m_x;
        roi.y = box.m_y;
        roi.width = box.m_w;
        roi.height = box.m_h;
        
        if (!FLAGS_silent)
            printf("[ INFO ] Going to crop (%d, %d) (%d, %d) from img (%d, %d) and resize to (%d, %d)\n", roi.x, roi.y, roi.width, roi.height, base.m_img.size().width, base.m_img.size().height, rw, rh);
    
        cv::Mat img = base.m_img(roi);
        
        int w = img.size().width;
        int h = img.size().height;
        size_t size = w*h*img.channels();

        std::shared_ptr<unsigned char> originalData;

        originalData.reset(new unsigned char[size], std::default_delete<unsigned char[]>());
        for (size_t id = 0; id < size; ++id) 
        {
            originalData.get()[id] = img.data[id];
        }


        

        cv::Mat resized(img); 
        cv::resize(img, resized, cv::Size(rw, rh));

        size = rw*rh*resized.channels();

        std::shared_ptr<unsigned char> data;

        data.reset(new unsigned char[size], std::default_delete<unsigned char[]>());
        for (size_t id = 0; id < size; ++id) 
        {
            data.get()[id] = resized.data[id];
        }

        InputImage inputImg(img, originalData, data, w, h);
        cropImages.push_back(inputImg);
    }
}