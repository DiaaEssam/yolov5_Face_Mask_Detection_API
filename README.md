# Face Mask Detection

This project implements face mask detection using YOLOv5 and provides functionalities for detecting masks in both images and videos.

## Table of Contents

- [Files](#files)
- [Installation](#installation)
- [Usage](#usage)
- [Contributing](#contributing)
- [License](#license)

## Files

The project consists of the following files:

- `API.py`: Flask API script for serving the face mask detection model.
- `face-mask-detection-yolo-v5.ipynb`: Jupyter Notebook containing the code for training and evaluating the face mask detection model.
- `detect.py`: Python script with functions for detecting face masks in images and videos.

## Installation

To use this project, you need to install the following dependencies:

- Python (3.10.12)
- NumPy
- Pandas
- Matplotlib
- TensorFlow
- OpenCV
- Seaborn
- PIL
- Flask
- Flasgger

You can install the required dependencies by running the following command:

```
pip install -r requirements.txt
```

## Usage

To use the face mask detection functionalities, you can import the `detect_image` and `detect_video` functions from `detect.py` into your own Python script or Jupyter Notebook. These functions take the current directory path and the model as inputs and perform face mask detection on images and videos, respectively.

Here's an example of how to use the `detect_image` function:

```python
from detect import detect_image

current_directory = "/path/to/directory"
model = load_model()

detect_image(current_directory, model)
```

And here's an example of how to use the `detect_video` function:

```python
from detect import detect_video

current_directory = "/path/to/directory"
model = load_model()

detect_video(current_directory, model)
```

## Contributing

Contributions to this project are welcome. If you find any issues or have suggestions for improvements, please open an issue or submit a pull request.

## License

This project is licensed under the [MIT License](LICENSE).
