# Machine Learning Classification for Predictive Maintenance

Project Highlights
Objective: Predict machine failure (binary classification).

Key Skills: 
- Feature Engineering
- Class Imbalance Handling (SMOTE)
- Pipeline Optimization
- Hyperparameter Tuning.

Business Impact: Identifies critical failure patterns, allowing for data-driven maintenance scheduling.

Technical Stack
Programming Language: Python

Libraries: Scikit-Learn, Pandas, NumPy, Matplotlib, Seaborn

Imbalance Handling: Imbalanced-learn (SMOTE)

Environment: Jupyter Notebook

Data Strategy & Preprocessing
The project utilizes a dataset with 10,000+ records featuring real-world industrial metrics.

- Feature Selection: Critically evaluated features to prevent Data Leakage and Overfitting (e.g., removing Product_ID and UDI).

- Numerical Features: Process Temperature, Air Temperature, Rotational Speed, Torque, and Tool Wear.

- Pipeline Engineering: Implemented StandardScaler and ColumnTransformer within a Scikit-learn Pipeline for reproducible data scaling and transformation.

Modeling & Performance

Multiple classification algorithms were evaluated and optimized using GridSearchCV with Stratified K-Fold Cross-Validation:

- Random Forest Classifier

- Logistic Regression

- Support Vector Machine (SVM)

- K-Nearest Neighbors (KNN)

- Decision Tree Classifier

Handling Class Imbalance
To address the rare occurrence of machine failures (minority class), SMOTE (Synthetic Minority Over-sampling Technique) was integrated into the training pipeline to improve the model's sensitivity to failure events.

Key Results
The models were rigorously tested, including a Leakage Detection Test (shuffling labels) to ensure the model learned actual patterns rather than noise.

- Precision / Recall: Focused on maximizing recall for the "Failure" class to ensure no downtime event goes undetected.

- Metric Monitoring: Evaluated using Confusion Matrices and F1-scores to balance detection accuracy and false alarm rates.
