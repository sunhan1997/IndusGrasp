# IndusGrasp
This is the official PyTorch Implementation of **IndusGrasp: A Novel Robotic Grasp Detection Method Using Synthetic
Data for Disordered Industrial Scenarios**


## Video
- The video is available in [YouTube](https://youtu.be/lmlCMYdMw5g)

## The pipeline:

<img src="image/all.png" width="500" height="500"/><br/>

The Comparison studies in the real-world environment
<img src="image/compare.png" width="950" height="240"/><br/>


## Requirements: Software
  ```shell
cd bop_toolkit

pip install -r requirements.txt -e .

sudo apt-get install libglfw3-dev libglfw3  

sudo apt-get install libassimp-dev   

pip install  cython cyglfw3 pyassimp==3.3 imgaug progressbar

pip install  -r requments.txt
  ```

## Generate dataset
- Using the `tools/pybullet_dataset/pybullet_dataset.py` to create render poses with your own 3D mesh(xx.urdf)

- Using the `tools/creat_grasp.py` to create grasp representation with your own 3D mesh(xx.ply)

- Using the `dataset_generate.py` to create your dataset

## Train the IndusGrasp
`python train.py`

## Test the IndusGrasp
`python test_with_detection.py ` use the maskrcnn in torchvision to detect the object firstly.
`python test_without_detection.py ` just detect the grasp pose.


## Notice
This version use the pyrender (from [bop_toolkit](https://github.com/thodan/bop_toolkit)) to replace the opengl.
There maybe some problems in the maskrcnn train/test with cuda, but this model is 
small, and you can run it with cpu. In addition, this the grasp sampling method of this project is not the best solution.
For better performance, you can use this [easy-dexnet_vis](https://github.com/sunhan1997/easy-dexnet_vis) to sample grasps.

