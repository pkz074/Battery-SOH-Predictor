README - PulseBat Project (Part 1 & Part 2)
----------------------------------------------

Project Overview
==============================================
This project implements a Linear Regression model to predict the State of Health (SOH) of lithium-ion batteries using the PulseBat dataset.  
It tests different sorting and preprocessing methods (by Cell ID, SOC, etc.) and evaluates their impact on model training.  
The dataset is divided into training (80%) and testing (20%) splits to ensure consistent model evaluation across all sorting strategies.  
The model’s performance is compared using regression metrics, and the final classification of battery health is performed using a threshold-based rule.


Features
==============================================
- Loads and cleans the PulseBat dataset automatically.
- Applies multiple sorting strategies:
  - original (as-is)
  - by_ID (sorted by cell number)
  - by_SOC (sorted by state-of-charge)
  - random (random shuffle)
- Trains and evaluates a Linear Regression model under each sorting method.
- Compares model performance using R², Mean Squared Error (MSE), and Mean Absolute Error (MAE).
- Uses the original dataset for the final demonstration of the regression equation and classification.
- Classifies batteries as “Healthy” or “Unhealthy” based on a user-defined SOH threshold (default: 0.6).

Files Included
===============================================
Files & Description 

PulseBat Dataset.xlsx: Dataset containing battery SOC, SOH, and voltage data 
SOH_Prediction.py: Main program file 


Requirements
===============================================
Install the following Python packages before running the script:

pip install pandas numpy scikit-learn openpyxl


How to Run the Program
===============================================
1. Place the following files in the same directory:
   - SOH_Prediction.py
   - PulseBat Dataset.xlsx

2. Open a terminal or command prompt in that directory.

3. Run the program using:
   python SOH_Prediction.py

4. When prompted, enter an SOH threshold value (for example, 0.6) or press Enter to use the default.

5. The program will:
   - Load and preprocess the dataset.
   - Train and evaluate a Linear Regression model using each sorting method.
   - Display R², MSE, and MAE for each sorting method.
   - Train a final model using the original dataset.
   - Print the regression equation for predicting SOH.
   - Display a sample classification of “Healthy” vs “Unhealthy” batteries.
   - Show the overall classification accuracy.


Conclusion
==============================================
This project demonstrates that sorting order has minimal effect on the linear regression model’s ability to predict battery SOH. 
The final model trained on the original dataset achieves consistent and accurate predictions, classifying batteries as “Healthy” or “Unhealthy” based on the SOH threshold.
