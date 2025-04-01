# Traffic Sign Recognition Using Deep Learning

## Overview

This project implements a neural network using TensorFlow to classify traffic signs from images. The model is trained on the German Traffic Sign Recognition Benchmark (GTSRB) dataset, which includes images of 43 different types of road signs. The AI is capable of identifying traffic signs in real-world scenarios, aiding in the development of autonomous driving systems.

## Installation

To set up the environment and install dependencies, run:

```sh
pip3 install -r requirements.txt
```

Ensure that you have downloaded the GTSRB dataset and placed it inside the directory. The GTSRB dataset can be downloaded from the following link: [GTSRB Dataset](https://drive.google.com/drive/folders/1-Nq13KV-b20Dsb9P5IPxh-DFOKQpYsCY?usp=sharing).

## Training the Model

To train the model, execute the following command:

```sh
python traffic.py gtsrb
```

The model trains for 10 epochs by default, improving accuracy with each epoch. Training results are displayed in real-time, showcasing the loss and accuracy metrics.

## Model Architecture

The model is a convolutional neural network (CNN) built using TensorFlow and Keras. The architecture consists of:

- Multiple convolutional layers with ReLU activation
- Max-pooling layers for dimensionality reduction
- Fully connected dense layers for classification
- A softmax activation function for final prediction
- Dropout layers to prevent overfitting

## Running Inference

To classify a new traffic sign using the trained model, run:

```sh
python traffic.py gtsrb model.h5
```

This command loads the trained model and classifies images from the dataset.

## References

- Data provided by: [J. Stallkamp, M. Schlipsing, J. Salmen, and C. Igel. The German Traffic Sign Recognition Benchmark: A multi-class classification competition. In Proceedings of the IEEE International Joint Conference on Neural Networks, pages 1453–1460. 2011](https://ieeexplore.ieee.org/document/6033395)

This project provides a foundational AI model for traffic sign recognition, contributing to autonomous vehicle research and computer vision advancements.

## License
This project is part of an Harvard's CS80 AI coursework assignment exploring Neural Networks.

---

## Experimentation

### Convolutional and Pooling
Initially, the model was tested with dropout set to 0.5 and a single hidden layer with 128 units.

By experimenting with different numbers of layers, I first applied a single convolution and pooling step, achieving an accuracy of **0.90**. Adding a second convolutional network improved accuracy to **0.94**. In this second test, I used **64 filters**, given that fewer pixels were being processed due to the initial pooling step.

I also adjusted the kernel and filter size from the default **2x2 to 3x3**, which maintained an accuracy of **0.94** but significantly improved the training speed.

Using Conv2D filters with **2x2 and 4x4** sizes resulted in decreased accuracy, with values of **0.927 and 0.93**, respectively. Thus, I reverted the filter size to **3x3**.

### Hidden Layers
Doubling the number of units to **256** led to a slight drop in accuracy to **0.92**. To counter this, I reduced the units by **50% to 64**, which increased the accuracy to **0.957**. Further, I experimented with **two hidden layers of 64 neurons each**, leading to a higher accuracy of **0.968**.

### Dropout
Since overfitting was less of a concern due to the two convolutional steps, I reduced dropout to **0.3**, achieving an accuracy of **0.96**. When I increased dropout to **0.6**, accuracy improved further to **0.97**.

### Final Results
After fine-tuning the architecture, the final model achieved a **maximum accuracy of 0.97**.
