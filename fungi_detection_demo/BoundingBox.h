/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   BoundingBox.h
 * Author: openvino
 *
 * Created on December 4, 2019, 3:29 PM
 */

#ifndef BOUNDINGBOX_H
#define BOUNDINGBOX_H

class BoundingBox {
public:
    BoundingBox(int x, int y, int w, int h, int lbl);
    BoundingBox(const BoundingBox& orig);
    virtual ~BoundingBox();

    int m_x;
    int m_y;
    int m_w;
    int m_h;
    int m_label;
    
private:

};

#endif /* BOUNDINGBOX_H */

