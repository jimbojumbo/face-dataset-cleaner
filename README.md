# face-dataset-cleaner

An implementation to clean large scale public face dataset

This is an unofficial implementation based on the paper [A Community Detection Approach to Cleaning Extremely Large Face Database](https://www.hindawi.com/journals/cin/2018/4512473/#B3)

To do the experiment, first prepare your face-dataset and LFW embedding files using a pre-trained face recognition network.

Use the **lfw_far_thresholding.py** to determine the similarity threshold between different face images.

Then run the **dataset_adjacency_build.py** to save the image pair similarity information in csv files, which will then be used in **dataset_cleaner.py** to build the graphs and do small community cleaning.

A small tool is provided to move original images to a separate folder according to the clean data list.

A first version of cleaned VGGFace2 training and testing image lists can be downloaded at [Google Drive](https://drive.google.com/open?id=1rHKzmeWCaiJ34HViWU2XvYnpbBKulHdO)
