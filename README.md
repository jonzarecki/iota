

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
Directories and settings:
Download the OID to iota/Data/oid directory or run from iota/Data    
```
ln -s [path to data] oid
```
