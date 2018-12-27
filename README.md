

# Informative-Annotations

Code for selecting informative labels of images based on a known distribution of labels.

### To install:
```
yes | conda create -n iota python=2.7 pyqt=4.11
source activate iota
```
```
yes | conda install pandas scikit-image ipython jupyter
yes | conda install scikit-learn
yes | conda install -c anaconda seaborn
yes | conda install -c anaconda scikit-image
yes | conda install -c omnia termcolor 
```

You may also need to update or downgrade matplotlib
```
conda install matplotlib=2.0.2

```

Install environment 
```
$ conda install --yes --file requirements.txt

or iterate over the file and install each package in “single package mode”

$ while read requirement; do conda install --yes $requirement; done < 
requirements.txt

```

Download IOTA-10K ground truth data 
```
wget https://chechiklab.biu.ac.il/~brachalior/IOTA/data/iota10K/iota_raw.csv
.tar.gz

```

Save OID data to iota/Data/oid or link to your data folder via 
```
ln -s [your data dir] oid

``` 
