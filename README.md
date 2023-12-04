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

* It's generally recommended to use python virtual environments when working with different python projects to handle dependencies/packages.


[Python documentation](https://docs.python.org/3.10/library/venv.html)

[Real python guide on venv](https://realpython.com/python-virtual-environments-a-primer/)

### Linux setup:

> Python version 3.10 or newer is required

1. Install python environment: (Debian package manager in this example. Use yours)
    ```sh
        sudo apt install python3-venv
    ```
2. Create virtual environment and activate enviroment: (Source to activate the virtual environment)
    ```sh
        python3 -m venv venv && source ./venv/bin/activate
    ```
> Note: If you restart your terminal session, you have to reactivate your virtual enviroment.

3. Upgrade pip and install dependencies:
    ```sh
        pip install --upgrade pip && pip install -r requirements.txt
    ```
4. Run scripts
    ```sh
        python3 main.py
    ```

## Getting started - ESP-32

To be written.

## Licence

To be written.
