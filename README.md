# DS-project-customer-churn

## Project Overview
This project predicts customer churn using machine learning algorithms. The goal is to identify customers who are likely to cancel their service, enabling proactive retention strategies.

## Techniques involved in project
1) Predictive modelling: - Data collection and loading
                         - Exploratory Data Analysis
                         - Data preprocessing and cleaning (for example, check for missing values)
                         - Feature engineering and selection (Feature scaling and encoding categorical variables)
                         - Data splitting (features and label)
                         - Models Training (models such as random forest, svm, xgboost)
                         - cross validation and hyperparameter tuning
                         - prediction made using models
                         - Models evaluation (accuracy of models)


## Objectives
- Predict behaviour to retain customers
- build predictive models to identify potential churners
- Compare performance of different machine learning algorithms


### Tools needed
- Anaconda Navigator
- Jupyter notebook
- Visual studio code (download jupyter with python extension )

📁 Data Loading & Saving
pd.read_csv('file.csv')         # Load CSV file
pd.read_excel('file.xlsx')      # Load Excel file  
pd.read_json('file.json')       # Load JSON file
pd.read_sql(query, connection)  # Load from SQL database
df.to_csv('file.csv')           # Save to CSV
df.to_excel('file.xlsx')        # Save to Excel
df.to_json('file.json')         # Save to JSON

📊 Data Exploration & Info
df.head()                       # First 5 rows
df.tail()                       # Last 5 rows
df.shape                        # (rows, columns)
df.info()                       # Data types, memory usage
df.describe()                   # Statistical summary
df.columns                      # Column names
df.dtypes                       # Data types of each column
df.nunique()                    # Count unique values per column
df.value_counts()               # Count occurrences
df.sample(n=5)                  # Random sample
df.size                         # Total number of elements
df.ndim                         # Number of dimensions
df.index                        # Index information
df.memory_usage()               # Memory usage per column

🧹 Data Cleaning & Missing Values
df.isnull().sum()               # Count missing values
df.isna().sum()                 # Same as isnull
df.notnull()                    # Non-null values
df.dropna()                     # Remove missing values
df.fillna(value)                # Fill missing values
df.fillna(method='ffill')       # Forward fill
df.fillna(method='bfill')       # Backward fill
df.duplicated().sum()           # Count duplicates
df.drop_duplicates()            # Remove duplicates
df.drop(columns=['col'])        # Drop columns
df.drop(index=[0, 1])           # Drop rows by index


🔄 Data Replacement & Manipulation
df.replace(old_value, new_value) # Replace values
df.replace({'col': {'old': 'new'}}) # Replace specific column values
df.replace([val1, val2], [new1, new2]) # Replace multiple values
df.replace(regex=True)          # Replace using regex
df.rename(columns={'old': 'new'}) # Rename columns
df.sort_values('col')           # Sort by column
df.sort_values(['col1', 'col2']) # Sort by multiple columns
df.reset_index()                # Reset index
df.set_index('col')             # Set column as index
df.reindex()                    # Reindex dataframe


🎯 Data Selection & Filtering
df.loc[row, col]                # Label-based selection
df.iloc[row, col]               # Position-based selection
df.at[row, col]                 # Fast single value access
df.iat[row, col]                # Fast single value access by position
df.query('col > 5')             # Query data
df[df['col'] > 5]               # Boolean indexing
df.filter(items=['col1', 'col2']) # Filter columns
df.where(condition)             # Where condition
df.mask(condition)              # Mask condition

🔢 Feature Engineering & Transformation
pd.get_dummies(df)              # One-hot encoding
df['col'].astype('category')    # Convert to category
df.corr()                       # Correlation matrix
df.cov()                        # Covariance matrix
df.groupby('col').mean()        # Group by operations
df.groupby('col').agg(['mean', 'sum']) # Multiple aggregations
df.pivot_table()                # Pivot table
df.melt()                       # Melt dataframe
df.transpose()                  # Transpose dataframe
df.apply(function)              # Apply function
df.map(function)                # Map function to series
df.transform(function)          # Transform data


📈 Data Visualization
df.hist()                       # Histograms
df.plot()                       # Basic plots
df.boxplot()                    # Box plots
df['col'].plot.bar()            # Bar plot
df.plot.scatter(x='col1', y='col2') # Scatter plot
df.plot.line()                  # Line plot
df.plot.area()                  # Area plot
df.plot.pie()                   # Pie chart


🔍 Data Analysis & Statistics
df.count()                      # Count non-null values
df.sum()                        # Sum of values
df.mean()                       # Mean values
df.median()                     # Median values
df.mode()                       # Mode values
df.std()                        # Standard deviation
df.var()                        # Variance
df.min()                        # Minimum values
df.max()                        # Maximum values
df.quantile(0.25)               # Quantiles
df.skew()                       # Skewness
df.kurt()                       # Kurtosis

📅 Date & Time Operations
pd.to_datetime(df['col'])       # Convert to datetime
df['col'].dt.year               # Extract year
df['col'].dt.month              # Extract month
df['col'].dt.day                # Extract day
df['col'].dt.dayofweek          # Day of week
df['col'].dt.strftime('%Y-%m-%d') # Format datetime

🤖 Machine Learning Shortcuts
# Essential imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score

# Data splitting
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Model training
model.fit(X_train, y_train)
predictions = model.predict(X_test)

# Feature importance
model.feature_importances_


📏 Model Evaluation
# Regression metrics
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
mean_squared_error(y_true, y_pred)
mean_absolute_error(y_true, y_pred)
r2_score(y_true, y_pred)

# Classification metrics  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
accuracy_score(y_true, y_pred)
precision_score(y_true, y_pred)
recall_score(y_true, y_pred)
f1_score(y_true, y_pred)

# Confusion matrix
from sklearn.metrics import confusion_matrix, classification_report
confusion_matrix(y_true, y_pred)
classification_report(y_true, y_pred)


🔍 Quick Data Checks
df.memory_usage()               # Memory usage
df.select_dtypes('object')      # Select text columns
df.select_dtypes('number')      # Select numeric columns
df.select_dtypes(include=['int64']) # Select specific types
df.isna().any()                 # Any missing values?
df.columns.tolist()             # Columns as list
df['col'].unique()              # Unique values in series
df['col'].nunique()             # Number of unique values
df.duplicated().any()           # Any duplicates?

⚡ One-Liners for Common Tasks
# Quick missing value percentage
(df.isnull().sum() / len(df)) * 100

# Quick correlation with target
df.corr()['target'].sort_values(ascending=False)

# Quick replace multiple values
df.replace({'Yes': 1, 'No': 0})

# Quick train-test split with stratification
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Quick model evaluation
print(f"Accuracy: {accuracy_score(y_test, predictions):.3f}")
print(f"MAE: {mean_absolute_error(y_test, predictions):.3f}")
print(f"R²: {r2_score(y_test, predictions):.3f}")

# You need to test multiple models and compare:
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

models = {
    'Random Forest': RandomForestClassifier(),
    'SVM': SVC(),
    'Logistic Regression': LogisticRegression(),
    'Decision Tree': DecisionTreeClassifier()
}

# Train and evaluate each model
results = {}
for name, model in models.items():
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    
    results[name] = {
        'accuracy': accuracy_score(y_test, y_pred),
        'f1_weighted': f1_score(y_test, y_pred, average='weighted'),
        'f1_class_1': f1_score(y_test, y_pred, average=None)[1],  # Class 1 F1
        'precision_class_1': precision_score(y_test, y_pred, average=None)[1],
        'recall_class_1': recall_score(y_test, y_pred, average=None)[1]
    }
    
# Quick feature selection by correlation
high_corr = df.corr()['target'].abs().sort_values(ascending=False)


🔧 Data Type Conversions
df.astype('int64')              # Convert data type
pd.to_numeric(df['col'])        # Convert to numeric
pd.to_categorical(df['col'])    # Convert to categorical
df['col'].astype('category')    # Convert to category
df['col'].astype(str)           # Convert to string

📋 Data Combining
pd.concat([df1, df2])           # Concatenate dataframes
pd.merge(df1, df2, on='col')    # Merge dataframes
df.join(df2)                    # Join dataframes
df.append(df2)                  # Append dataframes

   
