#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Oct  6 12:04:18 2019

@author: juan
"""

import os
import shutil

count = 0
for i in os.listdir("./images_app/test/"):
    for j in os.listdir(os.path.join("./images_app/test/", i)):
        zeros = "0"*(8-len(str(count)))
        shutil.copy(os.path.join("./images_app/test/", i, j),
                    os.path.join("./images_pic_video/",
                                 "ILSVRC2012_val_" + zeros +
                                 str(count) + ".JPEG"))
        count += 1
        break
    if count > 200:
        break
