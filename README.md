# Gender Detection Web App

## Objective 
The objective of this project is to develop and deploy a web application using Flask that leverages a machine learning algorithm, specifically Support Vector Machine (SVM), to recognize and classify gender based on input image.

## About the Project :
This project is a gender detection system designed to classify a person’s gender based on facial images. The system leverages machine learning techniques such as Eigenfaces for feature extraction, Principal Component Analysis (PCA) for dimensionality reduction, and a Support Vector Machine (SVM) for classification. Flask is used to deploy the model as a web application.

### **Technologies and Concepts**
Here are the key technologies and concepts used in the project:

**Eigenfaces for Feature Extraction**
- Eigenfaces is a technique that represents facial images as a combination of significant features (principal components) derived from the dataset.
- It reduces the facial data to a compressed format while retaining the essential features required for gender classification.
**Principal Component Analysis (PCA)**
- PCA is a dimensionality reduction technique used to reduce the high-dimensional facial data into a smaller set of principal components.
- It eliminates noise and redundant information, making the classification process faster and more accurate.

**Support Vector Machine (SVM)**
- SVM is a supervised learning algorithm used to classify data by finding the optimal hyperplane that separates different classes.
- In this project, the SVM classifier is trained on the principal components extracted from the facial images to distinguish between male and female classes.
**Flask for Deployment**
- Flask is a lightweight Python web framework used to deploy the gender detection model as a web application.
- It handles user interactions, processes input images, and displays predictions in a simple web interface.

## Workflow
- Data Preprocessing
![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

- Dimensionality Reduction
- Feature Extraction
![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)
- Model Training
- Web Application Development
- Prediction and Results
![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)


## Features
- **Face Detection**: Identifies faces in the uploaded image.
- **Gender Prediction**: Uses a machine learning model to classify the gender as Male or Female.
- **User-Friendly Interface**: Upload images easily through the web app interface.

## Dataset :
For this python project, I had used the [IMDB-WIKI-500k](https://data.vision.ee.ethz.ch/cvl/rrothe/imdb-wiki/) data; the dataset is available in the public domain and you can find it here. This dataset contain 4762 female images and 5399 male images. Total images 10161

## How It Works
1. **Upload an Image**: Users can upload any image containing one or more faces.
2. **Face Detection**: The app uses a face detection model to locate faces in the image.
3. **Gender Prediction**: For each detected face, the gender classification model predicts whether the person is Male or Female.
4. **Results**: The image is displayed back to the user with bounding boxes around detected faces and their corresponding gender predictions.

## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)

## Screenshots

![App Screenshot](https://via.placeholder.com/468x300?text=App+Screenshot+Here)
## Installation and Setup
### Prerequisites
- Python (3.8 or later)
- pip (Python package installer)
- Virtual environment (optional but recommended)

### Steps
1. Clone the repository:
   ```bash
   git clone https://github.com/pranayganvir/gender-recognition-flask.git
   cd gender-recognition-flask
   ```
2. Create a virtual environment (optional):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```
3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
4. Run the application:
   ```bash
   python app.py
   ```
5. Open the application in your browser at `http://localhost:5000`.

## Usage
1. Open the web app in your browser.
2. Click the **Upload Image** button and select an image file.
3. Wait for the app to process the image.
4. View the results with detected faces and their predicted genders.


## Future Improvements
- Add age prediction along with gender detection.
- Enable real-time face detection and gender prediction through a webcam.
- Support for multiple languages in the interface.

## Contributing
Contributions are welcome! Feel free to fork this repository and submit pull requests. Please ensure your contributions align with the goals of this project.



## Acknowledgements
- Face detection: OpenCV
- Gender classification model inspiration:Scikit-learn

---
Feel free to reach out for questions or suggestions!
