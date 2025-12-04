import pandas as pd
import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns

# Set style
sns.set(style="whitegrid")

def ingest_data(file_path, output_dir, description_file=None):
    print(f"Ingesting {file_path}...")
    file_name = os.path.basename(file_path)
    
    # Load descriptions if provided
    descriptions = {}
    if description_file and os.path.exists(description_file):
        try:
            with open(description_file, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            # Skip header lines if any, assuming format: Column Description Unit
            # Based on the file content, it seems to be tab or multiple space separated.
            # We'll try to parse it simply.
            for line in lines:
                parts = line.strip().split()
                if len(parts) >= 2:
                    col_name = parts[0]
                    # Heuristic to find description and unit
                    # This is a bit tricky without a strict parser, but let's try to just grab the whole line
                    # or maybe just store the raw line for lookup.
                    # Actually, let's just store the whole line as description for now or try to parse better if needed.
                    # The file has "Column Description Unit" header.
                    # Let's assume the first word is the column name.
                    desc_text = " ".join(parts[1:])
                    descriptions[col_name] = desc_text
        except Exception as e:
            print(f"Warning: Could not read description file: {e}")

    # Determine file type and read
    if file_path.endswith('.xlsx'):
        df = pd.read_excel(file_path)
    elif file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    else:
        raise ValueError("Unsupported file type")

    # Document Schema
    schema_info = []
    schema_info.append(f"File: {file_name}")
    schema_info.append(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")
    schema_info.append("-" * 100)
    schema_info.append(f"{'Column':<25} {'Type':<15} {'Non-Null':<10} {'Sample':<20} {'Description'}")
    schema_info.append("-" * 100)

    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].count()
        sample = str(df[col].iloc[0]) if not df.empty else "N/A"
        # Truncate sample if too long
        if len(sample) > 20:
            sample = sample[:17] + "..."
            
        desc = descriptions.get(col, "")
        schema_info.append(f"{col:<25} {dtype:<15} {non_null:<10} {sample:<20} {desc}")

    # Save schema to file
    output_file = os.path.join(output_dir, "schema.txt")
    with open(output_file, "w", encoding="utf-8") as f:
        f.write("\n".join(schema_info))
    
    print(f"Schema saved to {output_file}")
    return df

def understand_and_clean(df, output_dir):
    print("Understanding and Cleaning Data...")
    stats_file = os.path.join(output_dir, "summary_stats.txt")
    cleaning_log = os.path.join(output_dir, "cleaning_log.txt")
    
    log_entries = []
    
    # Descriptive Statistics
    desc = df.describe()
    with open(stats_file, "w", encoding="utf-8") as f:
        f.write(desc.to_string())
    log_entries.append("Generated summary statistics.")

    # Duplicates
    initial_rows = len(df)
    duplicates = df.duplicated().sum()
    if duplicates > 0:
        df = df.drop_duplicates()
        log_entries.append(f"Removed {duplicates} duplicate rows.")
    else:
        log_entries.append("No duplicates found.")
        
    # Missing Values
    missing = df.isnull().sum().sum()
    if missing > 0:
        log_entries.append(f"Found {missing} missing values.")
    else:
        log_entries.append("No missing values found.")

    # Save cleaning log
    with open(cleaning_log, "w", encoding="utf-8") as f:
        f.write("\n".join(log_entries))
        
    return df

import re

def sanitize_filename(name):
    return re.sub(r'[^\w\-_]', '_', str(name))

def perform_univariate_analysis(df, output_dir):
    print("Performing Univariate Analysis...")
    plots_dir = os.path.join(output_dir, "plots")
    if not os.path.exists(plots_dir):
        os.makedirs(plots_dir)
        
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    for col in numerical_cols:
        safe_col = sanitize_filename(col)
        plt.figure(figsize=(10, 6))
        sns.histplot(df[col], kde=True)
        plt.title(f'Distribution of {col}')
        plt.savefig(os.path.join(plots_dir, f'dist_{safe_col}.png'))
        plt.close()
        
        plt.figure(figsize=(10, 6))
        sns.boxplot(x=df[col])
        plt.title(f'Boxplot of {col}')
        plt.savefig(os.path.join(plots_dir, f'box_{safe_col}.png'))
        plt.close()

def perform_multivariate_analysis(df, output_dir):
    print("Performing Multivariate Analysis...")
    plots_dir = os.path.join(output_dir, "plots")
    
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    # Correlation Matrix
    if len(numerical_cols) > 1:
        plt.figure(figsize=(12, 10))
        corr = df[numerical_cols].corr()
        sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
        plt.title('Correlation Matrix')
        plt.savefig(os.path.join(plots_dir, 'correlation_matrix.png'))
        plt.close()
        
    # Time Series if Date column exists
    date_cols = [col for col in df.columns if 'Date' in col or 'date' in col]
    if date_cols:
        date_col = date_cols[0]
        # Ensure datetime
        if not pd.api.types.is_datetime64_any_dtype(df[date_col]):
            try:
                df[date_col] = pd.to_datetime(df[date_col])
            except:
                pass
        
        if pd.api.types.is_datetime64_any_dtype(df[date_col]):
            # Plot numerical cols over time
            for col in numerical_cols:
                safe_col = sanitize_filename(col)
                plt.figure(figsize=(14, 7))
                sns.lineplot(x=df[date_col], y=df[col])
                plt.title(f'{col} over Time')
                plt.savefig(os.path.join(plots_dir, f'time_series_{safe_col}.png'))
                plt.close()

def detect_outliers(df, output_dir):
    print("Detecting Outliers...")
    outlier_file = os.path.join(output_dir, "outliers.txt")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    
    outlier_info = []
    
    for col in numerical_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
        count = outliers.shape[0]
        percentage = (count / len(df)) * 100
        
        outlier_info.append(f"Column: {col}")
        outlier_info.append(f"  IQR: {IQR:.2f}")
        outlier_info.append(f"  Bounds: [{lower_bound:.2f}, {upper_bound:.2f}]")
        outlier_info.append(f"  Outliers: {count} ({percentage:.2f}%)")
        outlier_info.append("-" * 20)
        
    with open(outlier_file, "w", encoding="utf-8") as f:
        f.write("\n".join(outlier_info))

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python analyze_file.py <file_path> <output_dir> [description_file]")
        sys.exit(1)
    
    file_path = sys.argv[1]
    output_dir = sys.argv[2]
    description_file = sys.argv[3] if len(sys.argv) > 3 else None
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    df = ingest_data(file_path, output_dir, description_file)
    df = understand_and_clean(df, output_dir)
    perform_univariate_analysis(df, output_dir)
    perform_multivariate_analysis(df, output_dir)
    detect_outliers(df, output_dir)
    print("Analysis Complete.")
