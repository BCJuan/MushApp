# MushApp

Mushroom Recognition application based on detection and classifcation. It is composed of two parts:

+ Pytorch transfer learning and training
+ C++ deployment with OpenVINO

The final application consists in a web service built for an Android application. 

## Pytorch

The general pipeline for the overall model is the following: an image is input in the detection network, the mushrooms are detected. The detections are cropped with sfae margins and resized. FInally, the resized images are used as input for the classification network, which indicates which are the most probable mushroom species.

![General Pipeline image](./images/fig3.png?raw=true "General Pipeline for Mushroom Recognition")

### Detection

For detection, a network pretrained on the [Open Images Dataset v4](https://storage.googleapis.com/openimages/web/index.html) has been used. More specifically, the two networks used have been a [Faster R-CNN](https://arxiv.org/abs/1506.01497?context=cs.CV) and a [SSD](https://arxiv.org/abs/1512.02325).

![Detection Image](./images/fig2.png?raw=true "Mushroom Detection")

The reason to choose Open Images v4 is because it includes a mushroom class. In the case of the pretrained models those were obtained from the [Object Detection Model Zoo](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md) and the [OpenVINO Model Zoo](https://software.intel.com/en-us/openvino-toolkit/documentation/pretrained-models).

### Classification

For classification, four models from the Pytorch Model Zoo are retrained with the [FCGVx Mushroom Challenge Dataset](https://www.kaggle.com/c/fungi-challenge-fgvc-2018) for detecting over 1.5k classes of Funghi.

The four networks used are: Resnet, DenseNet, SqueezeNet and MobileNet V2.

To train a classifier, just run: `python retrain.py`. 

The command line options let you choose among the four networks, the percentage of layers frozen during retraining and the number of epochs. It offers also the posssibility to test the network. 

Run `python retrain.py -h` for more details.

## FPGA & OpenVINO 

### Converting the models

After training the models with Pytorch, they are converted first to ONNX and then ported to OpenVINO .xml and .bin files with its own converter.

The onnx conversion can be found at `python_transfer_learning/retrain.py` as the `export_to_onnx` function.

To convert the models from ONNX, in the case of the classifier, or from `.pb` files, for the detection models, two different command line arguments sets are needed. In the case of the detection models:

```
<INSTALL_DIR>/deployment_tools/model_optimizer/mo_tf.py --input_model=<MODEL_FOLDER>/frozen_inference_graph.pb --tensorflow_use_custom_operations_config <INSTALL_DIR>/deployment_tools/model_optimizer/extensions/front/tf/<MODEL_SUPPORT>_support.json --tensorflow_object_detection_api_pipeline_config <MODEL_FOLDER>/pipeline.config --reverse_input_channels
```
More help at [OpenVINO Docs](https://docs.openvinotoolkit.org/latest/_docs_MO_DG_prepare_model_convert_model_tf_specific_Convert_Object_Detection_API_Models.html)

In the case of the ONNX model the flags needed were: `--reverse_input_channels --scale --mean`

### Compiling 

We are assuming that we work in a Ubuntu system with user name `openvino`.

The OpenVino samples were copied (from `/opt/intel/openvino/inference_engine`) into the `/home/openvino/inference_engine` folder

1) Download the repository 
```
cd /home/openvino/inference_engine/samples
git clone https://github.com/BCJuan/MushApp.git (FIX this so that the final directory is /home/openvino/inference_engine/samples/fungi_detection_demo)
```

2) Download the network files from the github releases

3) Compile the samples (including the fungi_detection_demo)
```
./build_samples.sh
```

### Running

1) Be sure that OpenVino Starter Kit board is up and running with `aocl diagnose`
   
2) Setup the OpenCL environment
```
 source /opt/intel/openvino/deployment_tools/terasic_demo/setup_board_osk.sh
```
3) Execute the application

```
/home/openvino/inference_engine_samples_build/intel64/Release/fungi_detection_demo -i llanaguera.bmp -md frozen.xml -mc squeezenet_cpu.xml -silent -d "HETERO:CPU,FPGA"
```

### Demo Output

Will be a JSON text describing the location of the detected mushrooms in the image together with a  list of the most probable classes

## See Also
