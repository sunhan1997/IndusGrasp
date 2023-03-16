# IndusGrasp
This is the official PyTorch Implementation of **IndusGrasp: A Novel Robotic Grasp Detection Method Using Synthetic
Data for Disordered Industrial Scenarios**


## Video
- The video is available in [YouTube](https://youtu.be/lmlCMYdMw5g)

## The pipeline:

<img src="image/all.png" width="500" height="500"/><br/>

The Comparison studies in the real-world environment
<img src="image/compare.png" width="900" height="240"/><br/>


## Requirements: Software
  ```shell
sudo apt-get install libglfw3-dev libglfw3  

sudo apt-get install libassimp-dev   

pip install --pre --upgrade PyOpenGL PyOpenGL_accelerate 

pip install  cython cyglfw3 pyassimp==3.3 imgaug progressbar

pip install  -r requments.txt`
  ```

## Generate dataset
- Using the `tools/pybullet_dataset/pybullet_dataset.py` to create render poses with your own 3D mesh(xx.urdf)

- Using the `tools/creat_grasp.py` to create grasp representation with your own 3D mesh(xx.ply)

- Using the `dataset.py` to create your dataset

## Train the IndusGrasp
`python train.py`

## Test the IndusGrasp
`python test.py `


