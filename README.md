# FruitSyntheticDataset

This code is accompanied by a tutorial at https://amizorach.medium.com/creating-synthetic-data-for-machine-learning-dab5728f6411

For the already created Dataset simply use the files in the dataset template_folder
you can find a colab notebook in the notebook folder

Preparations:

Create a virtual environment or conda environment

pip install -r requirements.txt

-- This should install the required packages for creating the dataset

Unfortunately, I am not able to install pascal-voc  from the pip
You can try running
pip install pascal-voc

If this does not work run the following

unzip pascal_voc.tar.gz


Running

python fruit_dataset_creator.py -h
to get the full options for the dataset creator

for example, you can run the following

python fruit_dataset_creator.py -c 10 -s 250
To create 10 images of 250X250 pixels
