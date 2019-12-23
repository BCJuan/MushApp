// Copyright (C) 2018-2019 Intel Corporation
// SPDX-License-Identifier: Apache-2.0
//

#pragma once

#include <string>
#include <vector>
#include <gflags/gflags.h>
#include <iostream>

#ifdef _WIN32
#include <os/windows/w_dirent.h>
#else
#include <dirent.h>
#endif

/* thickness of a line (in pixels) to be used for bounding boxes */
#define BBOX_THICKNESS 2

/// @brief message for help argument
static const char help_message[] = "Print a usage message.";

/// @brief message for images argument
static const char image_message[] = "Required. Path to an .bmp image.";

/// @brief message for plugin_path argument
static const char plugin_path_message[] = "Optional. Path to a plugin folder.";

/// @brief message for model argument
static const char detection_model_message[] = "Required. Path to an .xml file with a trained model for detection.";

static const char classification_model_message[] = "Required. Path to an .xml file with a trained model for classification.";

/// @brief message for plugin argument
static const char plugin_message[] = "Plugin name. For example MKLDNNPlugin. If this parameter is pointed, " \
"the sample will look for this plugin only";

/// @brief message for assigning cnn calculation to device
static const char target_device_message[] = "Optional. Specify the target device to infer on; CPU, GPU, FPGA, HDDL or MYRIAD is acceptable. " \
"Sample will look for a suitable plugin for device specified";

/// @brief message for performance counters
static const char performance_counter_message[] = "Optional. Enables per-layer performance report";

/// @brief message for iterations count
static const char iterations_count_message[] = "Optional. Number of iterations. Default value is 1";

/// @brief message for clDNN custom kernels desc
static const char custom_cldnn_message[] = "Required for GPU custom kernels. "\
"Absolute path to the .xml file with the kernels descriptions.";

/// @brief message for user library argument
static const char custom_cpu_library_message[] = "Required for CPU custom layers. " \
"Absolute path to a shared library with the kernels implementations.";

/// @brief message for plugin messages
static const char plugin_err_message[] = "Optional. Enables messages from a plugin";


/// @brief message for top results number
static const char ntop_message[] = "Number of top results. Default value is 10";

static const char silent_message[] = "Silent";

DEFINE_bool(silent, false, silent_message);

/// \brief Define flag for showing help message <br>
DEFINE_bool(h, false, help_message);

/// \brief Define parameter for set image file <br>
/// It is a required parameter
DEFINE_string(i, "", image_message);

/// \brief Define parameter for set model file <br>
/// It is a required parameter
DEFINE_string(md, "frozen.xml", detection_model_message);

/// \brief Define parameter for set model file <br>
/// It is a required parameter
DEFINE_string(mc, "squeezenet_cpu.xml", classification_model_message);

/// \brief Define parameter for set path to plugins <br>
/// Default is ./lib
DEFINE_string(pp, "", plugin_path_message);

/// \brief device the target device to infer on <br>
DEFINE_string(d, "CPU", target_device_message);

/// \brief Enable per-layer performance report
DEFINE_bool(pc, false, performance_counter_message);

/// @brief Define parameter for clDNN custom kernels path <br>
/// Default is ./lib
DEFINE_string(c, "", custom_cldnn_message);

/// @brief Absolute path to CPU library with user layers <br>
/// It is a optional parameter
DEFINE_string(l, "", custom_cpu_library_message);

/// @brief Iterations count (default 1)
DEFINE_uint32(ni, 1, iterations_count_message);

/// @brief Enable plugin messages
DEFINE_bool(p_msg, false, plugin_err_message);

/// @brief Top results number (default 10) <br>
DEFINE_uint32(nt, 10, ntop_message);