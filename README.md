# Weed Detection using Convolutional Neural Networks

This project implements a weed detection system using Convolutional Neural Networks (CNNs) with TensorFlow and Keras. The entire project is contained within a single Jupyter notebook.

## Table of Contents
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Model Architecture](#model-architecture)
- [Data Augmentation](#data-augmentation)
- [Contributing](#contributing)
- [License](#license)

## Prerequisites

- Python 3.x
- Jupyter Notebook
- TensorFlow 2.x
- OpenCV
- NumPy

## Installation

1. Clone this repository:
   ```
   git clone https://github.com/yourusername/weed-detection.git
   cd weed-detection
   ```

2. Install the required packages:
   ```
   pip install jupyter tensorflow opencv-python numpy
   ```

## Usage

1. Prepare your dataset:
   - Place your images in a directory named `datas/` in the same directory as the notebook.
   - Organize images into subdirectories, one for each class of weed.

2. Open the Jupyter notebook:
   ```
   jupyter notebook weed_detection.ipynb
   ```

3. Run the cells in the notebook sequentially:
   - The first cell will train the model and save it as `weed.h5`.
   - The second cell will perform real-time detection using your webcam.

4. To stop the webcam detection, press 'q' when the opencv window is in focus.

## Project Structure

- `weed_detection.ipynb`: Jupyter notebook containing both the model training and real-time detection code.
- `datas/`: Directory containing the training images (you need to create this).
- `weed.h5`: Trained model file (generated after running the training cell).

## Model Architecture

The CNN architecture used in this project consists of:
- 3 Convolutional layers with ReLU activation
- 3 MaxPooling layers
- Flatten layer
- Dense layer with 512 units and ReLU activation
- Dropout layer (50% dropout rate)
- Output Dense layer with softmax activation (3 classes)

## Data Augmentation

To improve model generalization, we use data augmentation techniques including:
- Random rotation
- Width and height shifts
- Shearing
- Zooming
- Horizontal flipping

## Contributing

Contributions to this project are welcome. Please fork the repository and submit a pull request with your changes.

## License

[MIT License](https://opensource.org/licenses/MIT)
