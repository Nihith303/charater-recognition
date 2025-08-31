# Handwritten Character Recognition using Deep Learning

This project implements a deep learning model for recognizing handwritten characters (both digits and letters) using a Convolutional Neural Network (CNN). The application provides a user-friendly interface for drawing characters and getting real-time predictions.

## Features

- Handwritten digit and letter recognition (0-9, A-Z)
- Real-time prediction using a trained CNN model
- Interactive drawing canvas
- Simple and intuitive user interface

## Dataset

The model is trained on a combination of two datasets:

1. **MNIST Database** (Modified National Institute of Standards and Technology database)
   - Contains 60,000 training images and 10,000 testing images
   - Handwritten digits (0-9)
   - 28x28 pixel grayscale images
   - Download: [MNIST Dataset](http://yann.lecun.com/exdb/mnist/)

2. **EMNIST Dataset** (Extended MNIST)
   - Extends MNIST to include handwritten letters (A-Z, a-z)
   - Maintains the same format as MNIST (28x28 pixel grayscale images)
   - Download: [NIST EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)

### Dataset Setup Instructions

1. Create a directory called `sample_data` in the project root:
   ```
   mkdir sample_data
   ```

2. Place the following dataset files in the `sample_data` directory:
   - `mnist_train.csv` - MNIST training data
   - `mnist_test.csv` - MNIST test data
   - `A_Z Handwritten Data.csv` - EMNIST letters data (available from Kaggle or other sources)

3. The directory structure should look like this:
   ```
   handwritten-character-recognition-deep-learning/
   ├── sample_data/
   │   ├── mnist_train.csv
   │   ├── mnist_test.csv
   │   └── A_Z Handwritten Data.csv
   ├── model/
   ├── main.py
   ├── training.py
   ├── README.md
   └── requirements.txt
   ```

## Requirements

- Python 3.7+
- TensorFlow 2.x
- Keras
- NumPy
- Pandas
- Pillow
- scikit-learn
- pywin32 (for Windows GUI)
- tkinter (usually comes with Python)

## Installation

1. Clone the repository:
   ```bash
   git clone [your-repository-url]
   ```

2. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

1. Run the training script (optional, if you want to retrain the model):
   ```bash
   python training.py
   ```
   Note: Make sure to place the dataset files in the `sample_data/` directory:
   - `mnist_train.csv`
   - `mnist_test.csv`
   - `A_Z Handwritten Data.csv`

2. Run the application:
   ```bash
   python main.py
   ```

3. Use the application:
   - Draw a character (digit or letter) on the black canvas
   - Click "Predict" to see the model's prediction
   - Click "Clear" to clear the canvas and start over

## Model Architecture

The model uses a deep CNN with the following layers:
- Multiple Conv2D layers with BatchNormalization
- MaxPooling2D for downsampling
- Dropout layers for regularization
- Dense layers for classification
- Softmax activation for output (36 classes: 0-9, A-Z)

## Performance

The model achieves high accuracy on both training and test sets, with data augmentation techniques applied to improve generalization.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is open source and available under the MIT License.

## Acknowledgments

- [MNIST Database](http://yann.lecun.com/exdb/mnist/)
- [EMNIST Dataset](https://www.nist.gov/itl/products-and-services/emnist-dataset)
- Keras and TensorFlow teams for the deep learning framework
