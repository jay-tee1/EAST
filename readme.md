# EAST: An Efficient and Accurate Scene Text Detector

### Introduction
This is a tensorflow re-implementation of [EAST: An Efficient and Accurate Scene Text Detector](https://arxiv.org/abs/1704.03155v2). Furthermore this repository includes all necessary files for the thesis "Identification of Trailers and Containers in
Inland Ports by Drone Images".

### Contents
1. [Installation](#installation)
2. [Test](#train)
3. [Train](#test)

### Installation
1. For sucessful application TensorFlow 1.x is needed. In addition it requires a python version of 3.6.x.

### Test
To test the East detector eval.py has to be run. The provided command shows an example usage. It is important to note, that the model consits of 4 files. One data, one meta and one index file with a preceding name like "model.ckpt-50502". The checkpoint file has to be adjusted depending on the desired model for testing. With said model name the checkpoint file would need to hold following contents: 

model_checkpoint_path: "model.ckpt-50502" \\
all_model_checkpoint_paths: "model.ckpt-50502"

This adjustment is a necessity for running the python script successfully.The example training command is following: 

```
python eval.py --output_dir=./outputs/INSERT_OUTPUT_DIR --checkpoint_path=./models/INSERT_DIR_OF_MODEL --test_data_path=./test_samples/INSERT_TESTDATA_DIR
```

An image with detected boudning boxes as well as a text file holding the coordinates and confidences of those are written to the output directory.

### Train
The follwoing command was run for training and can be used for further training as well:

```
python multigpu_train.py --gpu_list=0 --checkpoint_path=./models/PATH_TOCHECKPOINT --training_data_path=./training_samples/ --geometry=RBOX --learning_rate=0.0001 --batch_size_per_gpu=10 --save_checkpoint_steps= 1000 --restore True
```
The PATH_TO_CHECKPOINT would need to be adapted to the pretrained model checkpoint, on which the training should be continued.
For training purposes each image has to be accompanied with a '.txt' file and the same preceding name as the image itself, holding the annotated bounding boxes. A python script for obtaining those file sis included with 'extract_data.py'.







