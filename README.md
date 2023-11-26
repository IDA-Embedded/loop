# Loop Detector for ESP-EYE

This repository contains code for a bell sound event detector to run on an ESP-EYE. This is the default workshop project case for IDA Embedded's TinyML workshop series November 2023-January 2024. The application continuously samples the microphone of the ESP-EYE and uses a machine learning model to detect a bell sound in the audio stream.

The repository has two projects:

* A Python project in the `Python` folder. This project reads data and labels from the `Data` folder, trains a machine learning model using TensorFlow and generates C code from the resulting model directly into the microcontroller project in the `ESP-32` folder.
* A microcontroller project in the `ESP-32` folder. This project contains the actual application code.

**NOTE**: You must run the Python project before you can build the ESP-32 project, because the generated ML model files (`model.c` and `model.h`) are not included in the repository.

## Training data format

Training data should be placed in the `Data` folder. Each recording should provide two files:

* A .wav file with 16 bit mono audio in 16 kHz. Its file name should be `audio_`*\<id\>*`.wav`, where *\<id\>* can be any unique string that identifies the recording.
* A .txt file with labels indicating the starting and ending times of each bell sound in the recording. Each row contains a starting time in seconds, the ending time in seconds and a label name, separated by a tab character. This conforms to Audacity's label file format. Its file name should be `label_`*\<id\>*`.txt`, where *\<id\>* is the recording identifier.

## Getting started - Python and TensorFlow

To be written.

## Getting started - ESP-32

To be written.

## Licence

To be written.
