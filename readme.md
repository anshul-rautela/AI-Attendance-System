# Face Recognition Attendance System

This project is a facial recognition-based attendance system that uses AI and machine learning to identify individuals and log their attendance automatically. It captures faces via a webcam, recognizes them using pre-trained encodings, and logs the attendance into a CSV file.

## Features

- **Facial Recognition**: Identifies known faces using the `face_recognition` and `dlib` libraries.
- **CSV Logging**: Attendance is saved in a CSV file with the format `YYYY-MM-DD_HH-MM-SS_attendance.csv`.
- **Confidence Filtering**: Only logs faces with a confidence level of 50% or higher.
- **Daily Logging**: Ensures attendance is logged only once per person per day.
- **Webcam Integration**: Captures real-time frames from a webcam for face recognition.

## Requirements

Make sure the following libraries are installed:

- Python 3
- `numpy`
- `opencv-python`
- `face_recognition`
- `dlib`
- `csv`

You can install the required libraries using pip:
```bash
pip install numpy opencv-python face_recognition dlib
```

## Directory Structure

- `lfw_funneled/`: Directory containing subfolders of individuals' images for training known faces.
  - Example structure:
    ```
    lfw_funneled/
    ├── Person1/
    │   ├── image1.jpg
    │   └── image2.jpg
    ├── Person2/
    │   ├── image1.jpg
    │   └── image2.jpg
    ```
- `program.py`: The main script file containing all the functionality.

## How to Run

1. **Prepare the Directory**:
   - Place images of known individuals in the `lfw_funneled/` directory. Each individual's images should be in a separate subfolder named after the individual.

2. **Run the Script**:
   ```bash
   python program.py
   ```

3. **Use the Webcam**:
   - The script will access your webcam to capture frames. Detected faces will be shown in real-time, with their names and confidence levels.
   - Press `q` to quit the webcam view.

4. **Check Attendance**:
   - Attendance will be logged in a CSV file with the timestamped name, e.g., `2025-01-27_14-30-00_attendance.csv`.

## Code Overview

### `AttendanceLogger` Class
- **Purpose**: Handles attendance logging to a CSV file.
- **Key Methods**:
  - `create_csv_if_not_exists()`: Creates a new CSV file if it doesn't already exist.
  - `log_attendance(known_faces)`: Logs attendance for recognized faces with confidence >= 50%.

### `load_known_faces()` Function
- **Purpose**: Loads known faces and their encodings from the specified directory.
- **Details**: Uses `dlib` and `face_recognition` to encode face images.

### `run_face_recognition()` Function
- **Purpose**: Captures frames from the webcam, detects faces, matches them with known faces, and logs attendance.
- **Details**: Uses real-time facial recognition and overlays results on the webcam feed.

## Example Output

When a known face is detected:
```plaintext
Logged attendance for John Doe with 65.23%
```

Real-time display:
- Green rectangle: Known face
- Red rectangle: Unknown face
- Status bar showing the number of detected faces

## Notes

- **Accuracy**: Ensure the images in the `lfw_funneled` directory are clear and well-lit for better recognition accuracy.
- **Performance**: Reducing the image resolution (e.g., resizing frames) improves processing speed.
- **Error Handling**: The script includes basic error handling for missing directories, invalid images, and face encoding issues.

## License

This project is open-source and free to use under the MIT License.
