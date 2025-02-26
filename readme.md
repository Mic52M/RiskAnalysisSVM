

# README

## Project: Identification of At-Risk Data via Kernelized SVMs

This project implements a method to analyze and visualize risk levels in a dataset using Support Vector Machines (SVMs) with linear and polynomial kernels. The analysis identifies "at-risk" data points, assigns them to specific risk levels, and visualizes the results in risk maps and histograms. This document provides a detailed explanation of the project's goals, methodology, and usage of the provided code.

---

## 1. Project Goals

The primary objectives of this project are:
1. **Risk Identification:** Assign risk levels to data points in a dataset using SVMs.
2. **Kernel Comparison:** Evaluate the effectiveness of linear and polynomial kernels in capturing data complexity.
3. **Visualization:** Generate risk maps and histograms to represent the risk distribution across the dataset.
4. **Comparative Analysis:** Compare the outputs of the two SVM models to highlight differences in risk classification.

---

## 2. Dataset and Preprocessing

### Dataset
The project uses the following datasets:
- **Discrimination.csv:** Features of individuals including both numerical and categorical variables.
- **y_origin_bin.csv:** Binary labels associated with the dataset, indicating classification categories.

### Preprocessing
1. **Feature Selection:** Four critical features were selected:
   - `Age`: Numerical variable.
   - `RaceDesc`: Categorical variable.
   - `Sex`: Categorical variable.
   - `Pay Rate`: Numerical variable.
2. **Standardization:** Numerical variables were standardized using Z-score scaling.
3. **One-Hot Encoding:** Categorical variables were transformed into binary vectors.
4. **Train-Test Split:** The dataset was split into training (80%) and testing (20%) sets for model training and evaluation.

---

## 3. Methodology

### Risk Level Assignment
The methodology assigns data points to five distinct risk levels using an iterative approach:
1. **Initial Training:** The SVM model is trained on the dataset.
2. **Support Vector Identification:** Points closest to the decision boundary (support vectors) are assigned to the current risk level.
3. **Point Removal:** Support vectors are removed from the dataset.
4. **Re-training:** The model is retrained on the remaining points, and the process repeats until all points are assigned to a risk level.

Risk levels range from:
- **Level 1 (High Risk):** Points closest to the decision boundary.
- **Level 5 (Low Risk):** Points farthest from the decision boundary.

### Visualization
Two types of visualizations are generated:
1. **Risk Maps:** Visualize risk levels in a 2D space using Principal Component Analysis (PCA) for dimensionality reduction.
2. **Histograms:** Show the distribution of data points across the five risk levels.

### Comparative Analysis
A comparison between the linear and polynomial SVM models is conducted:
- Points with unchanged risk levels are marked in gray.
- Points with changed risk levels are highlighted in red.

---

## 4. Code Structure

### File Organization
- `Discrimination.csv`: Input dataset containing features.
- `y_origin_bin.csv`: Input dataset containing binary labels.
- `script.py`: Main Python script implementing the methodology.

### Key Functions
1. **assign_risk_levels(model, X, max_levels=5):**
   - Assigns risk levels to data points using the iterative SVM approach.
2. **plot_risk_map(X, risk_levels, title, file_name):**
   - Generates and saves risk maps.
3. **plot_risk_histogram(risk_levels, title, file_name):**
   - Generates and saves histograms of risk levels.

### Outputs
The script generates the following files:
- **linear_risk_map.png:** Risk map for the linear SVM.
- **linear_istogramma.png:** Histogram of risk levels for the linear SVM.
- **poly_risk_map.png:** Risk map for the polynomial SVM.
- **poly_istogramma.png:** Histogram of risk levels for the polynomial SVM.
- **risk_comparison.png:** Comparison map showing changed and unchanged risk levels.

---

## 5. How to Run the Code

### Prerequisites
- Python 3.7+
- Required libraries: `pandas`, `numpy`, `matplotlib`, `sklearn`

### Steps
1. Clone or download the repository.
2. Place the input datasets (`Discrimination.csv` and `y_origin_bin.csv`) in the same directory as the script.
3. Run the script using the command:
   ```bash
   python script.py
   ```
4. The output visualizations will be saved in the same directory.

---

## 6. Theoretical Background

### Support Vector Machines (SVMs)
SVMs are supervised learning models used for classification and regression tasks. The model identifies a hyperplane that best separates data points into different classes.

#### Kernels
- **Linear Kernel:** Assumes linear separability of classes. It is computationally efficient but limited for datasets with complex relationships.
- **Polynomial Kernel:** Captures non-linear relationships by transforming input data into a higher-dimensional space.

### Principal Component Analysis (PCA)
PCA is a dimensionality reduction technique used to project high-dimensional data into a 2D space for visualization while retaining the maximum variance.

---

## 7. Results and Observations

1. **Linear SVM:**
   - Risk levels are distributed linearly, reflecting the model's limitation in capturing non-linear relationships.
   - Points closer to the decision boundary are classified as high risk.

2. **Polynomial SVM:**
   - Captures complex, non-linear relationships in the dataset.
   - Produces a more detailed and granular risk distribution.

3. **Comparative Analysis:**
   - Points with unchanged risk levels are mostly in low-risk regions.
   - Changed points highlight areas where the polynomial kernel identifies relationships missed by the linear kernel.

---

## 8. Possible Extensions

1. **Confidence Score Analysis:**
   - Include confidence levels in risk assignment to better quantify uncertainty.
2. **Additional Kernels:**
   - Explore other kernels like RBF or sigmoid to analyze their impact on risk classification.
3. **Application to Other Datasets:**
   - Test the methodology on datasets with different feature distributions.

---

## 9. Conclusion

This project demonstrates the effectiveness of kernelized SVMs in identifying and visualizing risk levels in a dataset. The polynomial kernel provides a significant improvement in capturing non-linear relationships, offering a more comprehensive risk analysis compared to the linear model.
