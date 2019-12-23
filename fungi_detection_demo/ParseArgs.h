/*
 * To change this license header, choose License Headers in Project Properties.
 * To change this template file, choose Tools | Templates
 * and open the template in the editor.
 */

/* 
 * File:   ParseArgs.h
 * Author: openvino
 *
 * Created on December 4, 2019, 1:24 PM
 */

#ifndef PARSEARGS_H
#define PARSEARGS_H

bool ParseAndCheckCommandLine(int argc, char *argv[]) ;

class ParseArgs {
public:
    ParseArgs();
    ParseArgs(const ParseArgs& orig);
    virtual ~ParseArgs();
private:

};

#endif /* PARSEARGS_H */

