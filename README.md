# Student Performance Analysis - MLOps Project

## Overview

The "Student Performance Analysis" project is a comprehensive end-to-end machine learning solution designed to predict students' math scores based on various demographic and socio-economic factors. This project demonstrates the application of MLOps principles, covering the entire machine learning lifecycle from data ingestion to model deployment. It serves as a practical example of how to structure and implement machine learning projects in a production environment, with a focus on reproducibility, scalability, and maintainability.

## Project Structure

- **notebooks:** Contains Jupyter notebooks for exploratory data analysis (EDA) and model training.
- **src:** This directory is organized into several subdirectories, each focusing on a specific component of the MLOps pipeline:
  - **data:**
    - **`data_ingestion.py`**: Script responsible for loading and splitting the dataset into training and test sets, ensuring that the data is properly formatted and ready for processing.
  - **features:**
    - **`build_features.py`**: Contains functions for feature engineering and data preprocessing, including handling missing values, encoding categorical variables, and scaling numerical features.
  - **models:**
    - **`train_model.py`**: Script to train various machine learning models using the preprocessed data. This script also handles the saving of trained models for future use.
    - **`predict_model.py`**: Script that loads the saved models and uses them to make predictions on new data, ensuring consistent and efficient predictions.
    - **`evaluate_model.py`**: Contains functions for evaluating the performance of the trained models, computing metrics like accuracy, mean squared error (MSE), and R-squared, and comparing different models.
  - **utils:**
    - **`logger.py`**: Provides custom logging functionality to track the progress and status of various processes within the pipeline, making it easier to trace errors and debug issues.
    - **`helpers.py`**: Contains utility functions that support the main scripts, such as data splitting, performance metric calculation, and other repetitive tasks.

- **templates:** HTML templates for the web interface used in the deployment phase.
- **Dockerfile:** Configuration file for containerizing the application, ensuring that it can be easily deployed in various environments.
- **requirements.txt:** Lists all the Python dependencies required for the project, ensuring that the environment can be replicated easily.

## Experiment Notebook

The `Experiment.ipynb` notebook is the primary environment for exploratory data analysis and experimentation. It includes:

- **Data Loading:** Initial data exploration, including loading the dataset and examining basic statistics.
- **Exploratory Data Analysis (EDA):** Visualization and analysis of key features, such as gender, ethnicity, parental education, lunch type, and test preparation, to understand their impact on students' math scores.
- **Data Preprocessing:** Handling missing values, encoding categorical variables into numerical format, and scaling features to prepare the data for modeling.
- **Model Selection:** Training and evaluating multiple machine learning models, including Linear Regression, Decision Trees, and Random Forests, to find the best-performing model.
- **Hyperparameter Tuning:** Optimizing model performance through techniques like Grid Search and Cross-Validation.

### Tools Required

All tools required go here. You would require the following tools to develop and run the project:

- A text editor or an IDE (like VsCode)
- Github Account [For Code Upload]
- Anaconda or Python [ For Create Virtual Environment ]

## Running the App

- if not clone the project on the local system 1st clone
- open cmd and go to project directory `cd projectDir`
THEN `git clone https://github.com/shahil04/end_to_end_ml_project.git`
- CREATE Virtual Environment Using Conda

### Installation

All installation steps go here.

- Installing an Anaconda via a .exe file [Set the environment Path ](by Default it is done when installed)
  - Create a project folder
  - Open CMD and RUN -->`conda activate`
  - RUN `conda create -name myenv python=3.9 -y`
  - RUN `conda activate myenv`
  Inside the myenv you install all libraries to run the project
  after this you simply [clone the GitHub repository and Run requireiment.txt]
  - Run on the same Folder where your project requirements.txt file available
  - Like --> [(myenv) S:\new\final_project\ml_end_to_end_project>] this is my cmd path
  - Run `python install -r requirements.txt`

- ### Run on Local system

  - Open the  terminal
  - activate the conda environment
  `conda activate myenv`
  - go to the  Project directory/folder like me
      `(myenv) S:\new\final_project\ml_end_to_end_project>`
  - RUN `python app.py`
  - Go to Browser paste localhost `http://127.0.0.1:5000/`
  - Awesome Project run on your localhost

## Development

This section gives some insight basic overview of Development.

### Life cycle of Machine Learning Project

#### Do EDA Task --> In Experiment.ipynb

- Understanding the Problem Statement
- Data Collection
- Data Checks to perform
- Exploratory data analysis
- Data Pre-Processing
- Model Training
- Choose the best model

### 1. Problem statement

- This project understands how the student's performance (test scores) is affected by other variables such as Gender, Ethnicity, Parental level of education, Lunch and Test preparation course.

### 2. Data Collection

- Dataset Source - <https://www.kaggle.com/datasets/spscientist/students-performance-in-exams?datasetId=74977>
- The data consists of 8 columns and 1000 rows.

### Dataset information

- gender : sex of students  -> (Male/female)
- race/ethnicity : ethnicity of students -> (Group A, B,C, D,E)
- parental level of education : parents' final education ->(bachelor's degree,some college,master's degree,associate's degree,high school)
- lunch : having lunch before test (standard or free/reduced)
- test preparation course : complete or not complete before test
- math score
- reading score
- writing score

### 3. Data Checks to perform

- Check Missing values
- Check Duplicates
- Check data type
- Check the number of unique values in each column
- Check statistics of the data set
- Check various categories present in the different categorical column

- Basic info
  - `[df.shape, df.isnull().sum(), df.duplicated().sum(), df.dtypes, df.info(), df.columns,  ]`
  - [checking the count of the number of the unique values of each column --> `df.nunique()`]
  - [check stats of data -->`df.describe()`]
  - checking the get unique value  of each column --> `df.unique()`

### 4. Exploring Data ( Visualization )

- `Matplotlib and Seaborn`[Histogram, Kernel Distribution Function (KDE), pie, bar, Boxplot(check outliers), pairplot]
- Multivariate analysis using pieplot
- Feature Wise Visualization
- UNIVARIATE ANALYSIS
- BIVARIATE ANALYSIS

### 5. MODELING

- Importing Sklearn, Pandas, Numpy, Matplotlib, Seaborn etc.
- Preparing X and Y variables
- Transform the into a numerical datatype to perform Models
- Create Column Transformer
- preprocessing data using OneHotEncoder, StandardScaler
- Create an Evaluate Function to give all metrics after model Training
  - [mae, rmse, r2_square, mse] For  regression Problem

- Create Model lists and run using a loop at once so that there no-repeat same task for all model

  - ```[models ={
                    - "Linear Regression": LinearRegression(),
                    "Lasso": Lasso(),
                    "Ridge": Ridge(),
                    "K-Neighbors Regressor": KNeighborsRegressor(),
                    "Decision Tree": DecisionTreeRegressor(),
                    "Random Forest Regressor": RandomForestRegressor(),
                    "XGBRegressor": XGBRegressor(), 
                    "CatBoosting Regressor": CatBoostRegressor(verbose=False),
                    "AdaBoost Regressor": AdaBoostRegressor()
                }]
                ```

- Results [Choose the best model with the help of the evaluate Function, especially using R2Score]
  - Now predict the model `lin_model.predict(X_test)`
  - Plot y_pred and y_test [visualize data] `plt.scatter(y_test,y_pred)`
    - `sns.regplot(x=y_test,y=y_pred,ci=None,color ='red');`
    - Difference between Actual and Predicted Values
      - `pd.DataFrame({'Actual Value':y_test,'Predicted Value':y_pred,'Difference':y_test-y_pred})`
