/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   MushroomClassificationResult.h
 * Author: openvino
 *
 * Created on December 6, 2019, 6:21 PM
 */

#ifndef MUSHROOMCLASSIFICATIONRESULT_H
#define MUSHROOMCLASSIFICATIONRESULT_H

#include <inference_engine.hpp>

#include <samples/common.hpp>
#include <string>
#include <vector>
#include <iostream>
#include <utility>

#include <ie_blob.h>
#include "BoundingBox.h"

using namespace std;
using namespace InferenceEngine;

/**
 * @class ClassificationResult
 * @brief A ClassificationResult creates an output table with results
 */
class MushroomClassificationResult {
private:
    const std::string _classidStr = "classid";
    const std::string _probabilityStr = "probability";
    const std::string _labelStr = "label";
    size_t _nTop;
    InferenceEngine::Blob::Ptr _outBlob;
    const std::vector<std::string> _labels;
    const std::vector<std::string> _imageNames;
    const size_t _batchSize;

public:
    vector<BoundingBox> m_boxes;
    
    void printHeader() 
    {
        std::cout << _classidStr << " " << _probabilityStr;
        if (!_labels.empty())
            std::cout << " " << _labelStr;
        std::string classidColumn(_classidStr.length(), '-');
        std::string probabilityColumn(_probabilityStr.length(), '-');
        std::string labelColumn(_labelStr.length(), '-');
        std::cout << std::endl << classidColumn << " " << probabilityColumn;
        if (!_labels.empty())
            std::cout << " " << labelColumn;
        std::cout << std::endl;
    }

public:
    explicit MushroomClassificationResult(InferenceEngine::Blob::Ptr output_blob,
                                  std::vector<std::string> image_names = {},
                                  size_t batch_size = 1,
                                  size_t num_of_top = 10,
                                  std::vector<std::string> labels = {}) :
            _nTop(num_of_top),
            _outBlob(std::move(output_blob)),
            _labels(std::move(labels)),
            _imageNames(std::move(image_names)),
            _batchSize(batch_size) 
    {
        if (_imageNames.size() != _batchSize) 
        {
            throw std::logic_error("Batch size should be equal to the number of images.");
        }
    }

    /**
    * @brief prints formatted classification results
    */
    void print() 
    {
        /** This vector stores id's of top N results **/
        std::vector<unsigned> results;
        TopResults(_nTop, *_outBlob, results);

        /** Print the result iterating over each batch **/
        //std::cout << std::endl << "Top " << _nTop << " results:" << std::endl << std::endl;
        
        std::cout << "JSON_START" << std::endl;
        std::cout << "{" << std::endl;
                
        for (unsigned int image_id = 0; image_id < _batchSize; ++image_id) 
        {
            //std::cout << "Image " << _imageNames[image_id] << std::endl << std::endl;
            std::cout << " \"subimage_"<< image_id<<"\" : " << std::endl;
            std::cout << " { \"pos\" : { "  ;
            std::cout << " \"x\" : " << m_boxes[image_id].m_x << "," ;
            std::cout << " \"y\" : " << m_boxes[image_id].m_y << "," ;
            std::cout << " \"w\" : " << m_boxes[image_id].m_w << "," ;
            std::cout << " \"h\" : " << m_boxes[image_id].m_h << "},"  << endl;
            
            //printHeader();
            

            for (size_t id = image_id * _nTop, cnt = 0; id < (image_id + 1) * _nTop; ++cnt, ++id)
            {
                std::cout.precision(7);
                /** Getting probability for resulting class **/
                const auto result = _outBlob->buffer().
                        as<InferenceEngine::PrecisionTrait<InferenceEngine::Precision::FP32>::value_type*>()
                [results[id] + image_id * (_outBlob->size() / _batchSize)];

                std::cout << "  \"prediction_"<< cnt <<"\" : { "  ;
                std::cout << "\"id\" : " << results[id] << ",";
                std::cout << "\"prob\" : " << result << ",";
                
                
                if (!_labels.empty()) {
                    std::cout << "\"label\" : \"" <<  _labels[results[id]] << "\"  ";
                }
                else
                {
                    std::cout << "\"label\" : \"??\"" ;
                }
                std::cout << "}";
                
                if (id < ((image_id + 1) * _nTop)-1)
                    std::cout << ",";
                
                std::cout << std::endl;
            }
            
            std::cout << "  }"  ;
            
            if (image_id < _batchSize-1)
                std::cout << ","  ;
            std::cout << std::endl;
        }
        
        std::cout << "}"  << std::endl;
        std::cout << "JSON_END" << std::endl;
    }
    
    
};


#endif /* MUSHROOMCLASSIFICATIONRESULT_H */

