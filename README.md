# LookIn: Face Recognition for  Attendance System using KNN model

## Overview
LookIn is a Python-based attendance system that uses face recognition to mark attendance automatically. 
It integrates liveness detection (blink detection) to ensure authenticity, supports timetable management, and provides analytics and reporting features. 
The system uses OpenCV, face_recognition, dlib, and Tkinter for the GUI.

## Prerequisites
- Python 3.8 or higher
- A webcam
- Git (optional, for downloading the shape predictor)

## Installation Steps

### 1. Clone or Download the Repository
If you have Git installed, clone the repository:
```bash
git clone <repository-url>
cd <repository-directory>
```
Alternatively, download the project as a ZIP file and extract it.

### 2. Set Up a Virtual Environment (Recommended)
Create and activate a virtual environment to manage dependencies:
```bash
python -m venv venv
```
- On Windows:
  ```bash
  venv\Scripts\activate
  ```
- On macOS/Linux:
  ```bash
  source venv/bin/activate
  ```

### 3. Install Required Python Packages
Install the necessary Python libraries using pip:
```bash
pip install opencv-python numpy face_recognition dlib pandas matplotlib pillow
```
**Note**: Installing `dlib` may require additional dependencies:
- On Windows, you may need to install CMake and Visual Studio Build Tools.
- On macOS/Linux, ensure you have `libpng` and `libjpeg` installed:
  ```bash
  # macOS
  brew install libpng libjpeg
  # Ubuntu
  sudo apt-get install libpng-dev libjpeg-dev
  ```

For `face_recognition`, ensure `dlib` is installed first. If you face issues, consider installing a precompiled `dlib` wheel:
```bash
pip install dlib --verbose
```

### 4. Download the Shape Predictor File
The system requires the `shape_predictor_68_face_landmarks.dat` file for facial landmark detection. Follow these steps to obtain it:
1. Visit the official dlib GitHub repository or download from [this link](http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2).
2. Download the `shape_predictor_68_face_landmarks.dat.bz2` file.
3. Extract the `.bz2` file to obtain `shape_predictor_68_face_landmarks.dat`. You can use tools like `bunzip2` or 7-Zip:
   - On macOS/Linux:
     ```bash
     bunzip2 shape_predictor_68_face_landmarks.dat.bz2
     ```
   - On Windows, use 7-Zip or another extraction tool.
4. Move the extracted `shape_predictor_68_face_landmarks.dat` file to the root directory of your project (where the main script is located).

**Alternative**: Download directly from a trusted source and place it in the project directory:
```bash
wget http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2
bunzip2 shape_predictor_68_face_landmarks.dat.bz2
mv shape_predictor_68_face_landmarks.dat <project-directory>
```

### 5. Create Required Directories
Create a directory to store student face images:
```bash
mkdir student_faces
```
Place sample student images (named as `<roll_number>.jpg`, e.g., `A001.jpg`) in the `student_faces` directory for testing.

### 6. (Optional) Create a Timetable File
The system uses a `timetable.json` file to manage class schedules. You can create it manually or through the GUI. Example format:
```json
{
  "Mon": [
    {"class": "Math", "start": "08:00", "end": "09:00"},
    {"class": "Physics", "start": "09:00", "end": "10:00"}
  ],
  "Tue": []
}
```
Save this as `timetable.json` in the project directory, or use the "Edit Timetable" feature in the app to create it.

## Execution Steps
1. Ensure all dependencies are installed and `shape_predictor_68_face_landmarks.dat` is in the project directory.
2. Run the main script:
   ```bash
   python main.py
   ```
   Replace `main.py` with the actual name of your script file.
3. The GUI will open with the following features:
   - **Start/Stop**: Begin or end the attendance marking process using the webcam.
   - **Register Face**: Add a new student's face by capturing an image via webcam.
   - **Edit Timetable**: Manage class schedules.
   - **Analytics**: View attendance statistics (summary, roll-wise, subject-wise).
   - **Report**: Generate a detailed attendance report as a CSV file.
   - **Export**: Export all attendance data to a single CSV file.

## Usage Notes
- **Face Registration**: Ensure good lighting and a single face in the frame when registering a new student.
- **Attendance Marking**: The system requires the user to blink (detected via liveness check) to confirm attendance.
- **Timetable**: Define class schedules in the timetable to enable automatic attendance marking for specific classes.
- **Logs**: Check `attendance.log` in the project directory for system activity and errors.

## Troubleshooting
- **Webcam Issues**: Ensure the webcam is connected and accessible. Try changing `camera_index` in the `CONFIG` dictionary (e.g., from `0` to `1`).
- **Missing Shape Predictor**: Verify that `shape_predictor_68_face_landmarks.dat` is in the project root.
- **Dependency Errors**: Ensure all libraries are installed in the virtual environment. Use `pip list` to check.
- **Performance**: If the system is slow, reduce the webcam resolution or frame rate in the `CONFIG` dictionary.

## Acknowledgments
- Built using [dlib](http://dlib.net/), [face_recognition](https://github.com/ageitgey/face_recognition), and [OpenCV](https://opencv.org/).
- Shape predictor file provided by dlib.

## License
This project is for educational purposes. Ensure compliance with privacy laws when using face recognition in real-world applications.
