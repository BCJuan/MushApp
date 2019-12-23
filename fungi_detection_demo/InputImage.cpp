/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   InputImage.cpp
 * Author: openvino
 * 
 * Created on December 4, 2019, 9:05 PM
 */

#include "InputImage.h"

InputImage::InputImage(cv::Mat& img, std::shared_ptr<unsigned char> original, std::shared_ptr<unsigned char> data, int w, int h)
{
    m_img = img;
    m_original = original;
    m_data = data;
    m_w = w;
    m_h = h;
}

InputImage::InputImage(const InputImage& orig) 
{
    m_img = orig.m_img;
    m_original = orig.m_original;
    m_data = orig.m_data;
    m_w = orig.m_w;
    m_h = orig.m_h;
}

InputImage::~InputImage() 
{
}

