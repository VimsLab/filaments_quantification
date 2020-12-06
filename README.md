filaments_quantification
-------
This project includes segmenation, seperation, reconnection three steps as stated in the paper.
The proposed method aovid reconnecting filaments at intersection and thus acheives a better result than existing methods.

This repo is based on the following two papers:

* Liu, Y., Treible, W., Kolagunda, A., Nedo, A., Saponaro, P., Caplan, J. and Kambhamettu, C., Densely Connected Stacked U-Network for Filament Segmentation in Microscopy Images, ECCV Workshops, 2018. [link](http://openaccess.thecvf.com/content_eccv_2018_workshops/w33/html/Liu_Densely_Connected_Stacked_U-network_for_Filament_Segmentation_in_Microscopy_Images_ECCVW_2018_paper.html)
* Yi Liu, Abhishek Kolagunda, Wayne Treible, Alex Nedo, Jeffrey Caplan, Chandra Kambhamettu, Intersection To Overpass: Instance Segmentation on Filamentous Structures with An Orientation-Aware Neural Network and erminus Pairing Algorithm, CVPR Bioimaging Workshop, 2019. [link](http://openaccess.thecvf.com/content_CVPRW_2019/paper/BIC/Liu_Intersection_to_Overpass_Instance_Segmentation_on_Filamentous_Structures_With_an_CVPRW_2019_paper.pdf)

![avatar](https://i-yliu.github.io/images/pipeLine.png)
Environment Configuration
----------------------
Anaconda enviroment file **mtquant.yml**

Matlab for reonnection step (Python version coming out very soon)

Dataset
-------
We have collected three dataset and listed as follows

**Binary Segmentation**
* Microtubule dataset 
* Actin dataset

**Instance Label**
 * Microtubule instance dataset 

Usage: 
------------
**Binary Segmentation**
* Modify config.yaml file, 
  * Mode **ts** for timeSeries tif data Evaluation
  * Mode **train** for training
  * Mode **predict** for None timeSeries data evaluation

**Separation by Orientation**
* Modify config.yaml file, 
  * Mode **train** for training, Training should be conducted on customed synthetic dataset
  * Mode **predict** for None timeSeries data evaluation 
  
**Reconnecting filaments**
* run runMain_v2.sh
* If not using Slurm, check description in slurm_run_mat.py 



