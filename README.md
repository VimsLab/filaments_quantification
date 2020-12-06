filaments_quantification
-------
This project includes segmenation, seperation, reconnection three steps as stated in the paper.
The proposed method aovid reconnecting filaments at intersection and thus acheives a better result than existing methods.

This repo is based on the following two papers:

* Liu, Y., Treible, W., Kolagunda, A., Nedo, A., Saponaro, P., Caplan, J. and Kambhamettu, C., Densely Connected Stacked U-Network for Filament Segmentation in Microscopy Images, ECCV Workshops, 2018. [link](http://openaccess.thecvf.com/content_eccv_2018_workshops/w33/html/Liu_Densely_Connected_Stacked_U-network_for_Filament_Segmentation_in_Microscopy_Images_ECCVW_2018_paper.html)
* Yi Liu, Abhishek Kolagunda, Wayne Treible, Alex Nedo, Jeffrey Caplan, Chandra Kambhamettu, Intersection To Overpass: Instance Segmentation on Filamentous Structures with An Orientation-Aware Neural Network and erminus Pairing Algorithm, CVPR Bioimaging Workshop, 2019. [link](http://openaccess.thecvf.com/content_CVPRW_2019/paper/BIC/Liu_Intersection_to_Overpass_Instance_Segmentation_on_Filamentous_Structures_With_an_CVPRW_2019_paper.pdf)

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
* * **ts** for timeSeries tif data Evaluation


Usage: Evaluation
-----------------
Modify cfgs/config_ssn_cls.yaml with *.pth file from cls/ folder with highest accuracy.<br>
python voting_evaluate_cls.py<br>

Note: You can use our model cls/model_cls_ssn_iter_70917_acc_0.925446.pth as the checkpoint in config_ssn_cls.yaml, and with majority voting you will get an accuracy of 93.51%. Due to randomness the accuracy might vary. <br>

This code has been heaviy borrowed from https://github.com/Yochengliu/Relation-Shape-CNN/ and https://github.com/erikwijmans/Pointnet2_PyTorch <br>


To cite our paper please use below bibtex.
  
```BibTex
        @InProceedings{Sheshappanavar_2020_CVPR_Workshops,
            author = {Venkanna Sheshappanavar, Shivanand and Kambhamettu, Chandra},
            title = {A Novel Local Geometry Capture in PointNet++ for 3D Classification},
            booktitle = {The IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR) Workshops},
            month = {June},
            year = {2020}
        }  
```



