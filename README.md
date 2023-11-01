## Electricity Price Prediction

This repository contains code for predicting electricity prices using various machine learning models. The code loads electricity price-related data from a CSV file, preprocesses it, trains different regression models, and provides predictions for electricity prices.

## Dependencies

Make sure you have the following libraries installed in your Python environment:

1. Pandas: Pandas is a powerful library for data manipulation and analysis. You can use it to read datasets from various file formats like CSV, Excel, SQL databases, and more. To download datasets, you may need to use other libraries or websites to fetch data files.

   To read a CSV file using Pandas:
  
   import pandas as pd
   df = pd.read_csv('Electricity.csv')
   

2. NumPy: NumPy is commonly used for numerical operations. You can use it to create and manipulate arrays, which are often used for storing and working with data.

3. Scikit-learn: Scikit-learn is a machine learning library that includes several datasets for practice and experimentation. You can load them using the `datasets` module.

   from sklearn import datasets
   iris = datasets.load_iris()
   

4. Seaborn and Matplotlib: These libraries are commonly used for data visualization, but they also include sample datasets that you can load for plotting and analysis.

   import seaborn as sns
   tips = sns.load_dataset('tips')

Before running the code, ensure that you have the following libraries installed:

- NumPy
- Pandas
- Matplotlib
- Seaborn
- Scikit-learn

You can install these libraries using pip:

bash
pip install numpy pandas matplotlib seaborn scikit-learn

## Dataset 
Link: https://www.kaggle.com/datasets/chakradharmattapalli/electricity-price-prediction

The above given is the link for the dataset.the dataset available at the kaggle website.
Here is a general explanation of the dataset attributes based on common features found in the electricity price prediction data.

1.DateTime: String, defines date and time of sample.
2.Holiday: String, gives name of holiday if day is a bank holiday.
3.HolidayFlag: Integer, 1 if day is a bank holiday, zero otherwise.
4.DayOfWeek: Integer (0-6), 0 monday, day of week.
5.WeekOfYear: Integer, running week within year of this date.
6.Day Integer: Day of the date.
7.Month Integer: Month of the date.
8.Year Integer: Year of the date.
9.PeriodOfDay integer: Denotes half hour period of day (0-47).
10.ForecastWindProduction: The forecasted wind production for this period.
11.SystemLoadEA: The national load forecast for this period.
12.SMPEA: The price forecast for this period.
13.ORKTemperature: The actual temperature measured at Cork airport.
14.ORKWindspeed: The actual windspeed measured at Cork airport.
15.CO2Intensity: The actual CO2 intensity in (g/kWh) for the electricity produced.
16.ActualWindProduction: The actual wind energy production for this period.
17.SystemLoadEP2: The actual national system load for this period.

## Usage
 
1.LOAD THE DATASET 
Loading a dataset means reading data from an external source into a data structure within a programming environment for analysis, manipulation, and processing.The specific method for loading a dataset depends on the programming language and libraries you are using.

2.Data Preprocessing 
Clean the dataset by handling missing values, removing duplicates, and addressing outliers.
Create new features and transform existing ones to improve model performance.
Divide the data into training, validation, and test sets.

3.Data Visualization
Data visualization is a powerful way to explore, analyze, and communicate insights from your dataset. In the context of electricity price prediction, data visualization can help you understand the underlying patterns and trends in the data. 

4.Feature Selection 
Decide on the set of features (independent variables) that the model will use for making predictions. Features may include historical electricity prices, demand, supply, weather conditions, and other relevant variables.

5.Model Training and Evaluation 
Evaluate the model's performance using the validation dataset. Use appropriate regression evaluation metrics such as Mean Absolute Error (MAE), Mean Squared Error (MSE), Root Mean Squared Error (RMSE), and R-squared (R^2).
Train the selected machine learning model using the training dataset. Fit the model to the historical data to learn the underlying patterns and relationships.

6.Testing and Validation 
Validate the final model using the test dataset to ensure that it performs well on unseen data.

## Running the Code

You can run the code by executing it in the python environment.Ensure you have the all required dependencies installed and have your dataset in the case, ("Electricity.csv")in the same directory

python 
python electricity_price_prediction.py

## Contributions

Contributions and improvements to this code are welcome. Feel free to fork the repository, make changes, and submit a pull request.

## License

This code is provided under the [MIT License],[Apache License],[GNU General Public License (GPL).]

Enjoy Predicting electricity prices using different machine learning and deep learning models!!