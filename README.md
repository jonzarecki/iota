

# Informative-Annotations

Code for selecting informative labels of images based on a known distribution of labels.

### To install   
```
$ conda install --yes --file requirements.txt

or iterate over the file and install each package in “single package mode”

$ while read requirement; do conda install --yes $requirement; done < 
requirements.txt

```

### To download IOTA-10K ground truth data 
```
wget https://chechiklab.biu.ac.il/~brachalior/IOTA/data/iota10K/iota_raw.csv
.tar.gz

```
### Open Images Dataset (OID)
Download [Image-Level Labels](https://storage.googleapis.com/openimages/web/download.html) to iota/Data/oid 
or 
link to your data folder by running from iota/Data:
```
ln -s [your data dir] oid

``` 

