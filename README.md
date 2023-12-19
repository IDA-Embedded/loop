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

## More raw data
Raw recordings from Loop fitness can be found here : https://anyvej11.dk/loop  

## Workshop slides

* Workshop 1, pt1: [TinyML Introduction](https://docs.google.com/presentation/d/1IYGDZZqRrE4ZvU1GUY1HpiN73ZtuhXn9buOHMAe0r3Y)
* Workshop 1, pt2: [TinyML HW/SW](https://docs.google.com/presentation/d/1Or0dlfwEeps-LqslExg2k3uU2bS3sLQUAZxr2nw6lvo)
* Workshop 2: [TinyML Data prep, model and deployment](https://docs.google.com/presentation/d/1cRfUvB82cQw7qil6pJezRVyu-Q3wIUR1_X0M9sBXE_U)
* Workshop 3, pt1: [Production Model Developemnt](https://docs.google.com/presentation/d/1JoLjX3rePVxhLUhv33q94o3sm2obSHIeEd-6BmAlmUA)
* Workshop 3, pt2: [Model optimization through Quantization](https://docs.google.com/presentation/d/e/2PACX-1vQSEskWAp7qakSjyGhIgF6iSSweX-2eQtBJbIA1a7KJkjtK_I6owz_Y1vjZUSEVK18cLWoZSUAUCjEt/pub?)
* Workshop 3, pt3: [ESP32 Code, profile and optimizing](https://docs.google.com/presentation/d/1u-gXyeAKtL2nef_diZLR5ulNL9ifhgIZVbEZjKQID7E)

## Workshop videos
Videos are uploaded as "unlisted" on youtube, use the links below :

* Workshop 1 : https://youtu.be/4ktrJHBaMmQ
* Workshop 2 : https://youtu.be/sm6M5oDPHAM
* Workshop 3 : https://youtu.be/zw0UyR8W64w

## Getting started - Python and TensorFlow

* It's generally recommended to use python virtual environments when working with different python projects to handle dependencies/packages.

[Python documentation](https://docs.python.org/3.10/library/venv.html)

[Real python guide on venv](https://realpython.com/python-virtual-environments-a-primer/)

### Linux setup:

> Python version 3.10 or 3.11 is required

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

### Mac M setup

> Python version 3.11 is required

1. Same steps as Linux but replace the `requirements` file in step `3`.

    ```sh
        pip install --upgrade pip && pip install -r requirements_mac_m.txt
    ```



## Getting started - ESP-32

To be written.

## Licence

To be written.
