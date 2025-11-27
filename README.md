# Skin Cancer Detection Using CNN
A deep learning project for classifying skin lesions as benign or malignant using Convolutional Neural Networks (CNN). This model helps in early detection of skin cancer by analyzing dermoscopic images.
### Overview:
Skin cancer is one of the most common cancers globally. Early detection significantly improves treatment success.
This project uses a CNN model trained on labelled skin lesion images to automatically classify them.
### Problem Statement:
Skin cancer is difficult to detect at early stages and relies heavily on expert diagnosis. Manual analysis of skin lesions is time-consuming and prone to human error. Therefore, there is a need for an automated system that can accurately classify skin lesions from images.
### Features:
* Image preprocessing (resizing, normalization, augmentation)
* CNN model built with TensorFlow/Keras
* Training, validation, and testing pipeline
* Accuracy & loss visualization
* Streamlit/Flask web app for real-time predictions
* Saved trained model files (.h5)-
### Requirements:
#### Operating System:
  * Windows 10/11 (64-bit) / Ubuntu / macOS
#### Programming Language:
  * Python 3.8 or later
  * Libraries / Dependencie
  * TensorFlow / Keras
  * NumPy
  * Pandas
  * Matplotlib / Seaborn
  * OpenCV (opencv-python)
  * Pillow
  * Scikit-learn
  * Streamlit or Flask (for web app)
#### Tools:
  * VS Code (recommended)
  * Git & GitHub for version control
#### Hardware:
  * Minimum: 4 GB RAM, CPU
  * Recommended: 8+ GB RAM, NVIDIA GPU for faster training
### System Architecture:
<img width="940" height="529" alt="image" src="https://github.com/user-attachments/assets/29b230aa-48d8-4225-a291-0e9270ca918a" />


### Output:

<img width="940" height="512" alt="image" src="https://github.com/user-attachments/assets/cdc3ca62-81d9-42e2-b326-8fc72fd79ff3" />
<img width="940" height="512" alt="image" src="https://github.com/user-attachments/assets/83b8b4ac-255b-4e32-91f9-f4bf7c7643ec" />


### Result:
The CNN-based skin cancer detection model achieved strong performance on the dermatoscopic image dataset, with an overall accuracy of ~90%. Key metrics such as precision, recall, and F1-score indicate that the model effectively distinguishes between benign and malignant skin lesions.
   * Sensitivity (Melanoma Detection): 92%
   * Specificity: 88%

These results show the model’s ability to minimize false negatives, which is essential for early cancer detection.
While the model performs well, accuracy decreases slightly on images from diverse skin types, highlighting the need for more varied training data. The system is designed to support dermatologists—not replace clinical expertise—ensuring ethical, fair, and interpretable AI-assisted diagnosis.
### Reference and Research:
[1] Shi Wang, Melika Hamian. “Skin Cancer Detection Based on Extreme Learning Machine and a Developed Version of Thermal Exchange Optimization”, Computational Intelligence and Neuroscience, 2021.

[2] Arslan Javaid, Muhammad Sadiq, Faraz Akram, “Skin Cancer Classification Using Image Processing and Machine Learning”, IEEE, 2021.

[3] Ahmed Wasif Reza, Samia Islam, “Skin Cancer Detection Using Convolutional Neural Network (CNN)”, ResearchGate, Conference Paper, 2019.

[4] S. Subha, Dr. D. C. Joy Winnie Wise, S. Srinivasan, M. Preetham, B. Soundarlingam, “Detection and Differentiation of Skin Cancer from Rashes”, Proceedings of the International Conference on Electronics and Sustainable Communication System (ICESC), 2020.

[5] Dubal, P., Bhatt, S., Joglekar, C., & Patii, S. (2017). “Skin cancer detection and classification.” 6th International Conference on Electrical Engineering and Informatics, 2017.

