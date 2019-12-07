conda env remove --name iota
conda create --name iota --yes

#export OID_DIR=/cortex/data/images/OID/oid/  # in lab
export OID_DIR=/media/yonatanz/yz/OID/oid
conda config --env --add channels ankurankan
conda config --env --add channels conda-forge
conda install --yes --file requirements.txt
#conda create  --name iota --yes --file requirements.txt
conda activate iota && while read requirement; do conda install -c conda-forge -c ankurankan --yes $requirement; done < requirements.txt
conda activate iota && pip install pandas==0.23.4