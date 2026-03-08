# Concrete Compressive Strength Predictor

An end-to-end Machine Learning web application designed to predict the 28-day compressive strength of concrete based on specific mix design parameters. 

## Machine Learning Architecture & Methodology
Due to the constraints of the provided dataset (n=111 observations), a highly rigorous small-data validation strategy was required to prevent model overfitting:
* Validation: Implemented Repeated K-Fold Cross-Validation (5 splits, 3 repeats).
* Comparative Analysis: Evaluated 6 diverse algorithms: Linear Regression, Ridge, Lasso, Random Forest, XGBoost, Support Vector Regression (SVR), and Multi-Layer Perceptron (MLP).
* Selection: Lasso Regression (alpha=0.1) was selected as the optimal production model. Complex tree ensembles (XGBoost) and deep learning models (MLP) suffered from severe overfitting on the 111-row dataset, whereas the regularized linear approach generalized successfully.

## Repository Structure
civil-mix/
|-- assests/
|   |-- infographics/
|   |   |-- r2_comparison.png
|   |   |-- rmse_comparison.png
|   |-- models/
|       |-- lasso_model.pkl
|       |-- scaler.pkl
|-- datasets/
|   |-- merged_dataset.csv
|   |-- original_dataset.xlsx
|-- main.py
|-- train.py
|-- requirements.txt
|-- README.md

## Installation & Usage (Standard Python)

1. Clone the repository and navigate to the folder:
git clone https://github.com/arnavgupta-py/civil-mix.git
cd civil-mix

2. Create a virtual environment (Recommended):
python -m venv venv
venv\Scripts\activate

3. Install the required dependencies:
pip install -r requirements.txt

4. Launch the application:
streamlit run main.py

The application will automatically open in your default web browser at http://localhost:8501.

## Technology Stack
* Frontend/Deployment: Streamlit
* Machine Learning: Scikit-Learn
* Data Processing: Pandas, NumPy
* Visualization: Matplotlib, Seaborn

## Experimentation & Model Training
The complete research phase, including Exploratory Data Analysis (EDA), feature scaling, and the comparative cross-validation of all 6 models, was conducted in a Jupyter Notebook provided in the directory as experiment.ipynb.
