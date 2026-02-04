# Blood Cell Segmentation Using Watershed Algorithm

## Overview
This project demonstrates blood cell segmentation from microscopic blood images using classical image processing techniques implemented in Python with OpenCV. The work focuses on understanding how traditional computer vision methods can be applied in biomedical image analysis to segment cells that are small, densely packed, or partially overlapping.

The project is learning-oriented and aims to build strong foundational knowledge in biomedical image processing rather than proposing a clinical-ready solution.

## Objective
The main objective of this project is to design and implement a marker-based watershed segmentation pipeline for blood cell detection. Through this project, core concepts such as contrast enhancement, noise reduction, thresholding, morphological operations, distance transform, and watershed segmentation are explored and practically applied.

## Methodology
The image processing pipeline follows these steps:

1. Load the microscopic blood image and convert it to grayscale  
2. Enhance local contrast using CLAHE (Contrast Limited Adaptive Histogram Equalization)  
3. Reduce noise using Gaussian blurring  
4. Apply Otsu’s thresholding and binary inversion  
5. Perform morphological opening to remove small noise  
6. Use distance transform to identify sure foreground regions  
7. Apply marker-based watershed segmentation to separate blood cells  

The watershed boundaries are overlaid on the original image for visual interpretation of segmentation results.

## Technologies Used
- Python  
- OpenCV  
- NumPy  
- Matplotlib  

## Project Structure
Blood-Cell-Segmentation/
│
├── data/
│ └── download.jpg
│
├── src/
│ └── blood_cell_watershed.py
│
├── requirements.txt
└── README.md

## How to Run
1. Clone the repository  
2. Install dependencies:
   ```bash
      pip install -r requirements.txt

## Results

The output displays the grayscale image, binary segmentation, distance transform, and final watershed-based blood cell boundaries. This approach improves separation compared to simple thresholding methods.

## Limitations

Sensitive to image quality and illumination

May struggle with heavily overlapping cells

Not suitable for clinical deployment

## Future Work

Blood cell counting

Cell type differentiation (RBC, WBC, platelets)

Comparison with deep learning segmentation models

## Author

HAREEM
Biomedical Engineering Undergraduate
Interested in Biomedical Image Processing and Computer Vision
    

