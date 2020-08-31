#/bin/bash

# Contributed by Aaron Soellinger
# Usage (*nix):
# $ mkdir VoxCeleb1; cd VoxCeleb1; /bin/bash path/to/dl_script.sh "yourusername" "yourpassword"
# Note: I found my username and password in an email titled "VoxCeleb dataset" 

U=$1
P=$2
wget http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partaa --user "$U" --password "$P" &
wget http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partab --user "$U" --password "$P" &
wget http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partac --user "$U" --password "$P" &
wget http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_dev_wav_partad --user "$U" --password "$P" &
wget http://www.robots.ox.ac.uk/~vgg/data/voxceleb/vox1a/vox1_test_wav.zip --user "$U" --password "$P" &
vox1_dev* > vox1_dev_wav.zip
unzip vox1_dev_wav.zip -d "dev" &
unzip vox1_test_wav.zip -d "test"
rm vox1_dev_wav_part*
rm wget*
wget http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/vox1_meta.csv
wget http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt
wget http://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/iden_split.txt
