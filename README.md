# Informative Object Annotations (IOTA) - CVPR 2019
Code for selecting informative labels of images based on a known distribution of labels.

### Paper
See our [CVPR 2019 Paper](http://openaccess.thecvf.com/content_CVPR_2019/papers/Bracha_Informative_Object_Annotations_Tell_Me_Something_I_Dont_Know_CVPR_2019_paper.pdf)

### Citation
If you find IOTA useful in your research, please consider citing:
```
@inproceedings{bracha2019informative,
  title={Informative Object Annotations: Tell Me Something I Don't Know},
  author={Bracha, Lior and Chechik, Gal},
  booktitle={Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition},
  pages={12507--12515},
  year={2019}
}
```

### Download code
```
git clone ssh://git@github.com:liorbracha/iota.git
```

### To install the anaconda environment
```
$ conda install --yes --file requirements.txt

to install in a “single package mode”:
$ while read requirement; do conda install --yes $requirement; done < requirements.txt
```

### IOTA-10K ground truth data 
```
wget https://chechiklab.biu.ac.il/~brachalior/IOTA/data/iota10K/iota_raw.csv.tar.gz
```
### Open Images Dataset (OID)
Download [Image-Level Labels](https://storage.googleapis.com/openimages/web/download.html) from the Open Images Dataset to data/oid 

To change the default location of the data or results directory: 
```
export RES_DIR=/where/results/are/written
export OID_DIR=/where/data/is/located/
``` 
    # data
    #   |_ ground_truth
    #   |_oid
    #       |_classes
    #       |_train
    #       |_validation
    #       |_test
    # results
    #   |_models
    #   |_counts
