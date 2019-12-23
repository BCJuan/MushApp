/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   BoundingBox.cpp
 * Author: openvino
 * 
 * Created on December 4, 2019, 3:29 PM
 */

#include "BoundingBox.h"

BoundingBox::BoundingBox(int x, int y, int w, int h, int lbl)
{
    m_x = x;
    m_y = y;
    m_w = w;
    m_h = h;
    m_label = lbl;
}

BoundingBox::BoundingBox(const BoundingBox& orig) 
{
    m_x = orig.m_x;
    m_y = orig.m_y;
    m_w = orig.m_w;
    m_h = orig.m_h;
    m_label = orig.m_label;
}

BoundingBox::~BoundingBox() {
}

