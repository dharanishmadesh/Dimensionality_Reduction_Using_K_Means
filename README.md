
#Human Activity Recognition using Smartphones

This project uses the UCI Human Activity Recognition (HAR) dataset to classify human activities based on data from smartphones. The dataset is used to predict activities like walking, walking upstairs, sitting, etc., based on accelerometer and gyroscope data collected from smartphone sensors.

The repository contains a Python script that downloads, processes, and loads the dataset, allowing users to perform machine learning tasks such as classification, clustering, or model evaluation.

## Dataset

The dataset used is the **Human Activity Recognition Using Smartphones** dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones). The dataset contains sensor data recorded from a Samsung Galaxy S smartphone.

- **Features**: 561 numeric features from the accelerometer and gyroscope sensors.
- **Target**: 6 human activities (Walking, Walking Upstairs, Walking Downstairs, Sitting, Standing, Laying).

## Project Structure

```
.
├── README.md                # Project overview and instructions
├── load_data.py             # Python script for loading the dataset
├── requirements.txt         # Python dependencies
└── examples/
    └── example_usage.py     # Example of how to use the dataset loader
```

## Installation

To run this project, you need to install the required Python packages.

1. Clone the repository:

```bash
git clone https://github.com/yourusername/human-activity-recognition.git
cd human-activity-recognition
```

2. Create a virtual environment (optional but recommended):

```bash
python -m venv venv
source venv/bin/activate   # On Windows use: venv\Scripts\activate
```

3. Install the required dependencies:

```bash
pip install -r requirements.txt
```

4. Install any additional dependencies (if not already included in `requirements.txt`):

```bash
pip install requests beautifulsoup4 pandas scikit-learn
```

## Usage

### 1. Load the Data

The `load_data.py` script provides a function to download and load the dataset directly from the UCI repository. It can load both training and test data.

#### Example Usage:

```python
from load_data import load_data

# Load the training data
X_train, y_train = load_data(data_type='train')

# Load the testing data
X_test, y_test = load_data(data_type='test')

# Print the first few rows of the training data
print(X_train.head())
print(y_train.head())
```

You can change `data_type='train'` to `data_type='test'` to load the test dataset instead.

### 2. Machine Learning Example

You can use this dataset with machine learning models such as KMeans, Logistic Regression, or others. Below is an example of how to use the `LogisticRegression` classifier on the dataset:

```python
from load_data import load_data
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Load data
X, y = load_data(data_type='train')

# Split the data into training and validation sets
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)

# Initialize the Logistic Regression model
model = LogisticRegression(max_iter=1000)

# Train the model
model.fit(X_train, y_train.values.ravel())

# Make predictions on the validation set
y_pred = model.predict(X_val)

# Evaluate the model
print("Accuracy:", accuracy_score(y_val, y_pred))
```

## Functions

### `load_data(data_type='train')`

Downloads and loads the dataset from the UCI HAR repository. You can load either the `train` or `test` data by passing the corresponding argument (`'train'` or `'test'`).

#### Parameters:
- `data_type`: A string, either `'train'` or `'test'`. Default is `'train'`.

#### Returns:
- `X`: The feature matrix (pandas DataFrame).
- `y`: The target labels (pandas DataFrame).

## Requirements

- Python 3.6 or higher
- Libraries: `requests`, `beautifulsoup4`, `pandas`, `scikit-learn`

You can install the required libraries using the provided `requirements.txt` file.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

