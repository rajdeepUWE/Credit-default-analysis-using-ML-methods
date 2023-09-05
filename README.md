# Credit-default-analysis-using-ML-methods

View the .ipynb file for the complete code. Refer to my medium post here https://rajdeepsarkar95.medium.com/mastering-credit-default-analysis-a-comprehensive-model-comparison-a35c2b04db06
for complete explanation of the code. 

The dataset is available here, use Colab or Jupyter notebook to run the code. 

 analysis of credit default prediction using various machine learning models. Let's break down the code step by step and then discuss the model performance.

Importing Libraries: The code starts by importing necessary Python libraries for data analysis and machine learning, such as pandas, numpy, seaborn, matplotlib, plotly, and various machine learning libraries like scikit-learn, CatBoost, and XGBoost.

Mounting Google Drive: The code mounts Google Drive to access a CSV file containing the credit risk dataset. It uses the google.colab library for this purpose.

Loading the Dataset: The code loads the credit risk dataset from Google Drive into a pandas DataFrame called base_credit.

Data Exploration and Preprocessing: It performs some data exploration and preprocessing steps, including checking for missing values, dropping rows with missing values, and encoding categorical variables using Label Encoding and One-Hot Encoding.

Splitting the Data: The data is split into training and testing sets.

Model Building: Several machine learning models are trained and evaluated on the dataset. The models used include:

Naive Bayes (GaussianNB)
Decision Tree
XGBoost
CatBoost
Random Forest
K-Nearest Neighbors (KNN)
Support Vector Machine (SVM)
Multi-Layer Perceptron (Neural Network)
Model Evaluation: For each model, accuracy, confusion matrix, and classification report metrics are calculated and printed. These metrics provide insights into how well each model performs in predicting credit defaults.

Now, let's discuss which model performed the best based on accuracy:

Naive Bayes (GaussianNB): Accuracy ~22%
Decision Tree: Accuracy ~87%
XGBoost: Accuracy ~91%
CatBoost: Accuracy ~93%
Random Forest: Accuracy ~90%
K-Nearest Neighbors (KNN): Accuracy ~85%
Support Vector Machine (SVM): Accuracy ~88%
Multi-Layer Perceptron (Neural Network): Accuracy ~88%
Based on accuracy alone, CatBoost performed the best with an accuracy of approximately 93%. It outperformed the other models in this specific analysis. However, it's important to note that accuracy is not the only metric to consider when evaluating a model. Depending on the specific problem and business goals, other metrics like precision, recall, and F1-score might be more important.

Additionally, it's crucial to perform further analysis, such as hyperparameter tuning and cross-validation, to ensure the selected model's robustness and generalizability. The choice of the best model can also depend on the specific requirements of the credit risk prediction problem and the trade-offs between different evaluation metrics.
