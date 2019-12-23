# MushApp

This application demonstrates the use of two networks, one to do mushroom detection using transfer learning from
a pre-trained network, and the second one to do more accurate mushroom classification.

## Compiling 

we are assuming that we work in a Ubuntu system with user openvino.
The OpenVino samples were copied (from /opt/intel/openvino/inference_engine) into the /home/openvino/inference_engine folder

1) Download the repository 

cd /home/openvino/inference_engine/samples
git clone https://github.com/BCJuan/MushApp.git (FIX this so that the final directory is /home/openvino/inference_engine/samples/fungi_detection_demo)

2) Download the network files from the github releases

3) Compile the samples (including the fungi_detection_demo)
./build_samples.sh



## Running

1) Be sure that OpenVino Starter Kit board is up and running with aocl diagnose
   
2) Setup the OpenCL environment
 source /opt/intel/openvino/deployment_tools/terasic_demo/setup_board_osk.sh

3) Execute the application

/home/openvino/inference_engine_samples_build/intel64/Release/fungi_detection_demo -i llanaguera.bmp -md frozen.xml -mc squeezenet_cpu.xml -silent -d "HETERO:CPU,FPGA"


## Demo Output

Will be a JSON text describing the location of the detected mushrooms in the image together with a 
list of the most probable classes

## See Also
