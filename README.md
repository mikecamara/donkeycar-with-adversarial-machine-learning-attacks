

# donkeycar: a python self driving library

[![Build Status](https://travis-ci.org/autorope/donkeycar.svg?branch=dev)](https://travis-ci.org/autorope/donkeycar)
[![CodeCov](https://codecov.io/gh/autoropoe/donkeycar/branch/dev/graph/badge.svg)](https://codecov.io/gh/autorope/donkeycar/branch/dev)
[![PyPI version](https://badge.fury.io/py/donkeycar.svg)](https://badge.fury.io/py/donkeycar)
[![Py versions](https://img.shields.io/pypi/pyversions/donkeycar.svg)](https://img.shields.io/pypi/pyversions/donkeycar.svg)

Donkeycar is minimalist and modular self driving library for Python. It is
developed for hobbyists and students with a focus on allowing fast experimentation and easy
community contributions.

#### Quick Links
* [Donkeycar Updates & Examples](http://donkeycar.com)
* [Build instructions and Software documentation](http://docs.donkeycar.com)
* [Discord / Chat](https://discord.gg/PN6kFeA)

![donkeycar](./docs/assets/build_hardware/donkey2.png)

#### Use Donkey if you want to:
* Make an RC car drive its self.
* Compete in self driving races like [DIY Robocars](http://diyrobocars.com)
* Experiment with autopilots, mapping computer vision and neural networks.
* Log sensor data. (images, user inputs, sensor readings)
* Drive your car via a web or game controller.
* Leverage community contributed driving data.
* Use existing CAD models for design upgrades.

### Get driving.
After building a Donkey2 you can turn on your car and go to http://localhost:8887 to drive.

### Modify your cars behavior.
The donkey car is controlled by running a sequence of events

```python
#Define a vehicle to take and record pictures 10 times per second.

import time
from donkeycar import Vehicle
from donkeycar.parts.cv import CvCam
from donkeycar.parts.tub_v2 import TubWriter
V = Vehicle()

IMAGE_W = 160
IMAGE_H = 120
IMAGE_DEPTH = 3

#Add a camera part
cam = CvCam(image_w=IMAGE_W, image_h=IMAGE_H, image_d=IMAGE_DEPTH)
V.add(cam, outputs=['image'], threaded=True)

#warmup camera
while cam.run() is None:
    time.sleep(1)

#add tub part to record images
tub = TubWriter(path='./dat', inputs=['image'], types=['image_array'])
V.add(tub, inputs=['image'], outputs=['num_records'])

#start the drive loop at 10 Hz
V.start(rate_hz=10)
```

See [home page](http://donkeycar.com), [docs](http://docs.donkeycar.com)
or join the [Discord server](http://www.donkeycar.com/community.html) to learn more.


# The car

Donkeycar S1 platform was used in the experiment. Donkeycar is an open source, easy-to-use, and well-documented Python library that can be used with a self-driving 1:10 scale remote control car.

The scaled autonomous vehicle used in the experiment is shown in the figure below comes pre-assembled with a Raspberry Pi 4, frontal camera, remote-controlled car chassis, battery, and a sensor hat. The frontal camera was the only sensor used in the competition. It was used to collect the data, which means collecting labeled images used to train a CNN supervised classifier autopilot using behavioral cloning.

![thecar](https://user-images.githubusercontent.com/8085864/156641685-faf4b0a0-f444-4144-bbdc-90a55569a5db.png)

Donkeycar provides tools to collect and label expert demonstrations, clean inaccurate records from the dataset, and train models using deep neural networks. The Donkeycar autopilot machine learning models are created using a form of supervised learning called behavioral cloning. The behavior of an expert is recorded and associated with labeled data and then used to train the model. The model is the output of the neural network and consists of an array of decision-making algorithms.

The Donkeycar software was built upon open source deep learning libraries such as OpenCV for image processing and Keras for training and running the machine learning models. Keras is a lightweight python neural network library that runs on TensorFlow, which the Donkeycar libraries use to train and run autopilot models. From creating to driving autopilot models, Donkeycar offers a variety of tools that allow fast experimentation. 

As shown in the figure below, creating models with Donkeycar involves three steps:

a) To collect data using the car.
b) To transfer the data to a host computer for cleaning and training the model.
c) Transferring the autopilot model back to the car ready to be tested.

![Screen Shot 2022-03-03 at 10 09 49 pm](https://user-images.githubusercontent.com/8085864/156644820-5b8f0f42-7fcc-414f-88c7-2fed0b81ab63.png)

To run the following solution, you must have the Donkeycar library installed both in the vehicle itself and your personal computer or a simple Google Colab repository. 

If you would like more information on to create a Donkeycar from scratch you can read [my blog post](https://softwareengineering.netlify.app/donkey-car/ )in which I go in detail about how to install the libraries in both the vehicle and your computer, you can always check the [Donkeycar official documentation](https://docs.donkeycar.com/). 


## Running the experiment code

With the Donkeycar installed, run the models available in this [repository](https://github.com/mikecamara/donkeycar-with-adversarial-machine-learning-attacks).


Run the linear model available in this repo and copy it to your local Donkeycar folder on your computer. Make sure you put inside the folder `mycar/models`. Then to copy it to the Raspberry Pi in your car, use the following command. The rsync command is used to sync the folder models from your computer with the folder models from your Raspberry Pi. 

````
rsync -rv --progress --partial ~/mycar/models/tub633-and-attack-08-5k.h5 pi@raspberrypi:~/mycar/models/
````

With the model transferred to the car, you are ready to experiment with the self-driving model.

Make sure you connect to your Raspberry Pi from your computer by connecting via SSH. 

````
ssh pi@raspberrypi
````

Then connected to the security shell of your Raspberry Pi you can run the following command to run a model. 

````
python manage.py drive --model ~/mycar/models/tub633-and-attack-08-5k.h5
````

It is essential to notice that you don't have to train these models added to this repo. They had already been trained. The process of training a neural network involves providing examples of labeled data as input and applying mathematical and statistical concepts such as feed-forward and error backpropagation. Those functions will gradually reduce the error between the predicted output and the desired output across several iterations over the dataset. Each cycle of complete training iterations is called an epoch. The training session is composed of as many epochs needed until the model can no longer improve its prediction performance.

