import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler, LabelEncoder

# Load the CSV file with different encodings
encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'cp1252']
file_path = r"C:\Users\dhami\OneDrive\Desktop\New folder\spam.csv"

df = None
for encoding in encodings:
    try:
        df = pd.read_csv(file_path, encoding=encoding)
        print(f"File successfully read with encoding: {encoding}")
        break
    except UnicodeDecodeError:
        print(f"Failed to read with encoding: {encoding}")
        continue

# Data Cleaning and EDA
if df is not None:
    print("\nCSV loaded")
    print("Column Names:", df.columns)  # Print column names

    print("Data Cleaning and Preprocessing...")
    # Handle missing values
    df.dropna(inplace=True)  # Remove rows with missing values

    # Check if there are numerical columns for scaling
    numerical_columns = df.select_dtypes(include=['int', 'float']).columns
    if not numerical_columns.empty:
        # Handle outliers (example)
        for column in numerical_columns:
            # Assuming outlier detection and removal using z-score
            z_scores = (df[column] - df[column].mean()) / df[column].std()
            df = df[(z_scores < 3) & (z_scores > -3)]

        # Feature scaling for numerical features
        scaler = StandardScaler()
        df[numerical_columns] = scaler.fit_transform(df[numerical_columns])

    # Check if there are categorical columns for encoding
    categorical_columns = df.select_dtypes(include=['object']).columns
    if not categorical_columns.empty:
        # Encode categorical features (example)
        label_encoder = LabelEncoder()
        df[categorical_columns] = df[categorical_columns].apply(label_encoder.fit_transform)

    print("Data Cleaning and Preprocessing completed.")

    # Exploratory Data Analysis (EDA)
    print("\nExploratory Data Analysis:")
    # Summary statistics
    print("Summary Statistics:")
    print(df.describe())
    
    # Correlation heatmap
    print("\nCorrelation Heatmap:")
    plt.figure(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, cmap='viridis', fmt=".2f")
    plt.title('Correlation Heatmap', fontsize=16)
    plt.xticks(fontsize=12)
    plt.yticks(fontsize=12)
    plt.show()

    # Histograms for numerical features
    print("\nHistograms for Numerical Features:")
    plt.figure(figsize=(12, 8))
    df.hist(figsize=(12, 8), bins=20, color='skyblue')
    plt.suptitle('Histograms for Numerical Features', fontsize=16)
    plt.tight_layout()
    plt.show()

else:
    print("\nCSV not loaded")
