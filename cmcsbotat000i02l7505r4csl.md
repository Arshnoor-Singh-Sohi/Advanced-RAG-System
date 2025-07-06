---
title: "🐼 Complete Pandas Learning Guide"
datePublished: Sun Jul 06 2025 23:48:04 GMT+0000 (Coordinated Universal Time)
cuid: cmcsbotat000i02l7505r4csl
slug: complete-pandas-learning-guide
cover: https://cdn.hashnode.com/res/hashnode/image/upload/v1751845635646/293590dc-7e3b-4fef-9b85-ecc904038074.webp
tags: python, data-science, data-analysis, pandas

---

Welcome to the most comprehensive guide to pandas! This repository will take you from absolute beginner to pandas expert with hands-on examples, real-world datasets, and advanced techniques.

## 📋 Table of Contents

1. [What is Pandas?](https://claude.ai/chat/035264ec-b713-4af7-8bf6-3e51d1e631ea#what-is-pandas)
    
2. [Installation & Setup](https://claude.ai/chat/035264ec-b713-4af7-8bf6-3e51d1e631ea#installation--setup)
    
3. [Core Data Structures](https://claude.ai/chat/035264ec-b713-4af7-8bf6-3e51d1e631ea#core-data-structures)
    
4. [Data Loading & Saving](https://claude.ai/chat/035264ec-b713-4af7-8bf6-3e51d1e631ea#data-loading--saving)
    
5. [Data Exploration & Inspection](https://claude.ai/chat/035264ec-b713-4af7-8bf6-3e51d1e631ea#data-exploration--inspection)
    
6. [Data Selection & Indexing](https://claude.ai/chat/035264ec-b713-4af7-8bf6-3e51d1e631ea#data-selection--indexing)
    
7. [Data Cleaning & Preprocessing](https://claude.ai/chat/035264ec-b713-4af7-8bf6-3e51d1e631ea#data-cleaning--preprocessing)
    
8. [Data Manipulation & Transformation](https://claude.ai/chat/035264ec-b713-4af7-8bf6-3e51d1e631ea#data-manipulation--transformation)
    
9. [Grouping & Aggregation](https://claude.ai/chat/035264ec-b713-4af7-8bf6-3e51d1e631ea#grouping--aggregation)
    
10. [Merging & Joining Data](https://claude.ai/chat/035264ec-b713-4af7-8bf6-3e51d1e631ea#merging--joining-data)
    
11. [Time Series Analysis](https://claude.ai/chat/035264ec-b713-4af7-8bf6-3e51d1e631ea#time-series-analysis)
    
12. [Advanced Operations](https://claude.ai/chat/035264ec-b713-4af7-8bf6-3e51d1e631ea#advanced-operations)
    
13. [Performance Optimization](https://claude.ai/chat/035264ec-b713-4af7-8bf6-3e51d1e631ea#performance-optimization)
    
14. [Best Practices](https://claude.ai/chat/035264ec-b713-4af7-8bf6-3e51d1e631ea#best-practices)
    
15. [Real-World Projects](https://claude.ai/chat/035264ec-b713-4af7-8bf6-3e51d1e631ea#real-world-projects)
    

---

## What is Pandas?

**Pandas** (Python Data Analysis Library) is the most powerful and popular data manipulation and analysis library for Python. It provides fast, flexible, and expressive data structures designed to make working with structured data both easy and intuitive.

```plaintext
┌─────────────┐    ┌─────────────┐    ┌─────────────────┐
│   Raw Data  │───▶│   Pandas    │───▶│  Clean, Analyzed│
│  (CSV, SQL, │    │  DataFrame  │    │     Data        │
│   JSON...)  │    │             │    │                 │
└─────────────┘    └─────────────┘    └─────────────────┘
```

### Why Pandas?

* **Fast and efficient** - Built on NumPy for speed
    
* **Flexible data structures** - Series and DataFrame
    
* **Powerful data manipulation** - Filter, transform, aggregate
    
* **Easy data import/export** - CSV, Excel, SQL, JSON, and more
    
* **Missing data handling** - Built-in NaN support
    
* **Time series functionality** - Date/time operations
    
* **Integration** - Works seamlessly with other Python libraries
    

---

## Installation & Setup

### Prerequisites

* Python 3.7 or higher
    
* pip package manager
    

### Installation

```bash
# Basic installation
pip install pandas

# With optional dependencies for better performance
pip install pandas[performance]

# Full installation with all optional dependencies
pip install pandas[all]

# Specific version
pip install pandas==2.1.0
```

### Essential Imports

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Display settings for better output
pd.set_option('display.max_columns', None)
pd.set_option('display.width', None)
pd.set_option('display.max_colwidth', 50)
```

### Verify Installation

```python
print(pd.__version__)
print(pd.show_versions())  # Detailed version info
```

---

## Core Data Structures

Pandas has two primary data structures: **Series** (1-dimensional) and **DataFrame** (2-dimensional).

### Series - 1D Labeled Array

```python
import pandas as pd
import numpy as np

# Creating Series from different data types
# From a list
fruits = pd.Series(['apple', 'banana', 'orange', 'grape'])
print(fruits)

# From a dictionary
prices = pd.Series({
    'apple': 1.20,
    'banana': 0.80,
    'orange': 1.50,
    'grape': 2.00
})
print(prices)

# From NumPy array with custom index
temperatures = pd.Series(
    np.random.randint(20, 35, 7),
    index=['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun'],
    name='Temperature'
)
print(temperatures)
```

#### Series Operations

```python
# Basic information
print(f"Length: {len(prices)}")
print(f"Data type: {prices.dtype}")
print(f"Index: {prices.index}")
print(f"Values: {prices.values}")

# Statistical operations
print(f"Mean: {prices.mean():.2f}")
print(f"Max: {prices.max()}")
print(f"Min: {prices.min()}")
print(f"Standard deviation: {prices.std():.2f}")

# Accessing elements
print(f"Apple price: ${prices['apple']}")
print(f"First item: {prices.iloc[0]}")

# Boolean indexing
expensive_fruits = prices[prices > 1.0]
print("Expensive fruits:", expensive_fruits)
```

### DataFrame - 2D Labeled Data Structure

```python
# Creating DataFrame from dictionary
sales_data = {
    'Product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones'],
    'Price': [999.99, 25.50, 79.99, 299.99, 149.99],
    'Quantity': [50, 200, 150, 30, 80],
    'Category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Electronics']
}

df = pd.DataFrame(sales_data)
print(df)

# Creating DataFrame with custom index
df_indexed = pd.DataFrame(
    sales_data,
    index=['P001', 'P002', 'P003', 'P004', 'P005']
)
print(df_indexed)

# Creating DataFrame from list of dictionaries
employees = [
    {'Name': 'John', 'Age': 28, 'Department': 'IT', 'Salary': 75000},
    {'Name': 'Sarah', 'Age': 32, 'Department': 'HR', 'Salary': 65000},
    {'Name': 'Mike', 'Age': 29, 'Department': 'Finance', 'Salary': 70000},
    {'Name': 'Emily', 'Age': 26, 'Department': 'IT', 'Salary': 72000}
]

employees_df = pd.DataFrame(employees)
print(employees_df)
```

#### DataFrame Basic Information

```python
# Shape and structure
print(f"Shape: {df.shape}")  # (rows, columns)
print(f"Columns: {df.columns.tolist()}")
print(f"Index: {df.index.tolist()}")
print(f"Data types:\n{df.dtypes}")

# Quick overview
print(df.info())  # Comprehensive information
print(df.describe())  # Statistical summary for numeric columns
print(df.head())  # First 5 rows
print(df.tail(3))  # Last 3 rows
```

---

## Data Loading & Saving

Pandas can read from and write to various file formats.

### Reading Data

#### CSV Files

```python
# Basic CSV reading
df = pd.read_csv('sales_data.csv')

# Advanced CSV reading with parameters
df = pd.read_csv(
    'sales_data.csv',
    sep=',',                    # Delimiter
    header=0,                   # Row to use as column names
    index_col=0,               # Column to use as index
    usecols=['Name', 'Price'],  # Specific columns to read
    dtype={'Price': 'float64'}, # Specify data types
    na_values=['N/A', 'NULL'],  # Additional NA values
    skiprows=1,                 # Skip first row
    nrows=1000,                 # Read only first 1000 rows
    encoding='utf-8'            # File encoding
)

# Reading with date parsing
df = pd.read_csv(
    'time_series.csv',
    parse_dates=['Date'],       # Parse Date column as datetime
    date_format='%Y-%m-%d'      # Specify date format
)
```

#### Excel Files

```python
# Read Excel file
df = pd.read_excel('data.xlsx')

# Read specific sheet
df = pd.read_excel('data.xlsx', sheet_name='Sales')

# Read multiple sheets
sheets_dict = pd.read_excel('data.xlsx', sheet_name=None)

# Advanced Excel reading
df = pd.read_excel(
    'data.xlsx',
    sheet_name='Sheet1',
    header=1,                   # Use second row as header
    usecols='A:D',             # Read columns A through D
    skiprows=2,                # Skip first 2 rows
    nrows=100                  # Read only 100 rows
)
```

#### JSON Files

```python
# Read JSON
df = pd.read_json('data.json')

# Read JSON with specific orientation
df = pd.read_json('data.json', orient='records')

# Read nested JSON
df = pd.json_normalize(data)  # Flatten nested JSON
```

#### SQL Databases

```python
import sqlite3

# Connect to database
conn = sqlite3.connect('database.db')

# Read SQL query into DataFrame
df = pd.read_sql_query("SELECT * FROM sales", conn)

# Read entire table
df = pd.read_sql_table('sales', conn)

# With parameters
query = "SELECT * FROM sales WHERE date >= ? AND date <= ?"
df = pd.read_sql_query(query, conn, params=['2024-01-01', '2024-12-31'])

conn.close()
```

#### Other Formats

```python
# HTML tables
tables = pd.read_html('https://example.com/data.html')
df = tables[0]  # First table

# Parquet (fast binary format)
df = pd.read_parquet('data.parquet')

# Feather (fast binary format)
df = pd.read_feather('data.feather')

# HDF5 (hierarchical data format)
df = pd.read_hdf('data.h5', key='sales')

# Pickle (Python serialization)
df = pd.read_pickle('data.pkl')
```

### Saving Data

#### CSV Files

```python
# Basic CSV writing
df.to_csv('output.csv')

# Advanced CSV writing
df.to_csv(
    'output.csv',
    index=False,               # Don't write row indices
    columns=['Name', 'Price'], # Specific columns
    sep=';',                   # Custom separator
    na_rep='NULL',             # How to represent NaN
    float_format='%.2f',       # Float formatting
    encoding='utf-8'           # File encoding
)

# Append to existing file
df.to_csv('output.csv', mode='a', header=False)
```

#### Excel Files

```python
# Basic Excel writing
df.to_excel('output.xlsx', index=False)

# Write multiple sheets
with pd.ExcelWriter('output.xlsx') as writer:
    df1.to_excel(writer, sheet_name='Sales', index=False)
    df2.to_excel(writer, sheet_name='Inventory', index=False)
    df3.to_excel(writer, sheet_name='Customers', index=False)

# Advanced Excel writing with formatting
with pd.ExcelWriter('output.xlsx', engine='xlsxwriter') as writer:
    df.to_excel(writer, sheet_name='Data', index=False)
    
    # Get workbook and worksheet objects
    workbook = writer.book
    worksheet = writer.sheets['Data']
    
    # Add formatting
    header_format = workbook.add_format({'bold': True, 'bg_color': '#CCE5FF'})
    worksheet.set_row(0, None, header_format)
```

#### Other Formats

```python
# JSON
df.to_json('output.json', orient='records', indent=2)

# Parquet (recommended for large datasets)
df.to_parquet('output.parquet')

# Feather (fast read/write)
df.to_feather('output.feather')

# HDF5 (good for time series)
df.to_hdf('output.h5', key='data', mode='w')

# SQL
df.to_sql('table_name', conn, if_exists='replace', index=False)
```

---

## Data Exploration & Inspection

Understanding your data is the first step in any analysis.

### Basic Information

```python
# Dataset overview
print(f"Dataset shape: {df.shape}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Column information
print("Column info:")
print(df.info())

# Data types
print("\nData types:")
print(df.dtypes)

# Check for missing values
print("\nMissing values:")
print(df.isnull().sum())
print(f"Total missing values: {df.isnull().sum().sum()}")

# Unique values in each column
print("\nUnique values per column:")
for col in df.columns:
    print(f"{col}: {df[col].nunique()} unique values")
```

### Statistical Summary

```python
# Descriptive statistics for numeric columns
print("Descriptive statistics:")
print(df.describe())

# Include non-numeric columns
print("\nAll columns summary:")
print(df.describe(include='all'))

# Custom percentiles
print("\nCustom percentiles:")
print(df.describe(percentiles=[.1, .25, .5, .75, .9, .95, .99]))

# Individual column statistics
print(f"Mean price: ${df['Price'].mean():.2f}")
print(f"Median price: ${df['Price'].median():.2f}")
print(f"Price standard deviation: ${df['Price'].std():.2f}")
print(f"Price range: ${df['Price'].min():.2f} - ${df['Price'].max():.2f}")
```

### Data Distribution

```python
# Value counts for categorical data
print("Category distribution:")
print(df['Category'].value_counts())

# Percentage distribution
print("\nCategory percentage:")
print(df['Category'].value_counts(normalize=True) * 100)

# Check for duplicates
print(f"\nDuplicate rows: {df.duplicated().sum()}")

# Most common values
print("\nMost common products:")
print(df['Product'].value_counts().head())
```

### Sample Data Inspection

```python
# View random samples
print("Random sample:")
print(df.sample(5))

# View specific rows
print("Specific rows:")
print(df.iloc[10:15])

# View data with conditions
print("High-value products:")
print(df[df['Price'] > 100])
```

---

## Data Selection & Indexing

Pandas offers multiple ways to select and access data.

### Column Selection

```python
# Single column (returns Series)
prices = df['Price']
print(type(prices))  # <class 'pandas.core.series.Series'>

# Single column (returns DataFrame)
prices_df = df[['Price']]
print(type(prices_df))  # <class 'pandas.core.frame.DataFrame'>

# Multiple columns
subset = df[['Product', 'Price', 'Quantity']]
print(subset)

# Using dot notation (only if column name is valid Python identifier)
products = df.Product
print(products)
```

### Row Selection

```python
# Select by index position (iloc)
first_row = df.iloc[0]        # First row
last_row = df.iloc[-1]        # Last row
first_five = df.iloc[:5]      # First 5 rows
middle_rows = df.iloc[2:8]    # Rows 2 through 7

# Select by index label (loc)
# Assuming we have a custom index
df_indexed = df.set_index('Product')
laptop_row = df_indexed.loc['Laptop']

# Select multiple rows by label
selected_products = df_indexed.loc[['Laptop', 'Mouse']]
```

### Boolean Indexing

```python
# Simple conditions
expensive_items = df[df['Price'] > 100]
electronics = df[df['Category'] == 'Electronics']
in_stock = df[df['Quantity'] > 0]

# Multiple conditions with & (and) and | (or)
expensive_electronics = df[(df['Price'] > 100) & (df['Category'] == 'Electronics')]
cheap_or_accessories = df[(df['Price'] < 50) | (df['Category'] == 'Accessories')]

# Using isin() for multiple values
selected_products = df[df['Product'].isin(['Laptop', 'Monitor'])]

# String operations
products_with_m = df[df['Product'].str.startswith('M')]
products_containing_board = df[df['Product'].str.contains('board', case=False)]

# Negation
not_electronics = df[~(df['Category'] == 'Electronics')]
```

### Advanced Selection with query()

```python
# Query method for complex conditions
result = df.query('Price > 100 and Category == "Electronics"')

# Using variables in query
min_price = 50
result = df.query('Price > @min_price')

# Complex string operations
result = df.query('Product.str.contains("Laptop")')
```

### Multi-level Indexing

```python
# Creating multi-level index
df_multi = df.set_index(['Category', 'Product'])
print(df_multi)

# Selecting from multi-level index
electronics_data = df_multi.loc['Electronics']
laptop_data = df_multi.loc[('Electronics', 'Laptop')]

# Cross-section
all_laptops = df_multi.xs('Laptop', level='Product')
```

---

## Data Cleaning & Preprocessing

Real-world data is messy. Here's how to clean it effectively.

### Handling Missing Values

#### Detecting Missing Values

```python
# Check for missing values
print("Missing values per column:")
print(df.isnull().sum())

# Percentage of missing values
print("Percentage missing:")
print((df.isnull().sum() / len(df)) * 100)

# Visualize missing data pattern
import seaborn as sns
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 6))
sns.heatmap(df.isnull(), cbar=True, yticklabels=False, cmap='viridis')
plt.title('Missing Data Pattern')
plt.show()
```

#### Handling Missing Values

```python
# Remove rows with any missing values
df_cleaned = df.dropna()

# Remove rows where all values are missing
df_cleaned = df.dropna(how='all')

# Remove rows with missing values in specific columns
df_cleaned = df.dropna(subset=['Price', 'Quantity'])

# Remove columns with too many missing values
threshold = 0.5  # Remove columns with >50% missing
df_cleaned = df.dropna(thresh=int(threshold * len(df)), axis=1)

# Fill missing values
# With a constant value
df_filled = df.fillna(0)

# With column mean/median/mode
df['Price'].fillna(df['Price'].mean(), inplace=True)
df['Category'].fillna(df['Category'].mode()[0], inplace=True)

# Forward fill and backward fill
df_filled = df.fillna(method='ffill')  # Forward fill
df_filled = df.fillna(method='bfill')  # Backward fill

# Interpolation for numeric data
df['Price'].interpolate(method='linear', inplace=True)
```

### Handling Duplicates

```python
# Check for duplicates
print(f"Duplicate rows: {df.duplicated().sum()}")

# View duplicate rows
duplicates = df[df.duplicated()]
print(duplicates)

# Remove duplicates
df_no_duplicates = df.drop_duplicates()

# Remove duplicates based on specific columns
df_no_duplicates = df.drop_duplicates(subset=['Product', 'Category'])

# Keep last occurrence instead of first
df_no_duplicates = df.drop_duplicates(keep='last')
```

### Data Type Conversion

```python
# Check current data types
print(df.dtypes)

# Convert to appropriate types
df['Price'] = pd.to_numeric(df['Price'], errors='coerce')
df['Quantity'] = df['Quantity'].astype(int)
df['Category'] = df['Category'].astype('category')

# Convert strings to datetime
df['Date'] = pd.to_datetime(df['Date'])

# Convert to categorical for memory efficiency
df['Category'] = df['Category'].astype('category')

# Handle mixed types
df['Mixed_Column'] = pd.to_numeric(df['Mixed_Column'], errors='coerce')
```

### String Cleaning

```python
# Remove whitespace
df['Product'] = df['Product'].str.strip()

# Convert to lowercase/uppercase
df['Category'] = df['Category'].str.lower()
df['Product'] = df['Product'].str.title()

# Replace specific values
df['Product'] = df['Product'].str.replace('_', ' ')

# Remove special characters
df['Product'] = df['Product'].str.replace(r'[^\w\s]', '', regex=True)

# Extract numbers from strings
df['Numbers'] = df['Text_Column'].str.extract(r'(\d+)')

# Split strings
df[['First_Name', 'Last_Name']] = df['Full_Name'].str.split(' ', expand=True)
```

### Outlier Detection and Treatment

```python
# Statistical outlier detection (IQR method)
def detect_outliers_iqr(df, column):
    Q1 = df[column].quantile(0.25)
    Q3 = df[column].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    
    outliers = df[(df[column] < lower_bound) | (df[column] > upper_bound)]
    return outliers, lower_bound, upper_bound

# Detect outliers in Price column
outliers, lower, upper = detect_outliers_iqr(df, 'Price')
print(f"Outliers found: {len(outliers)}")
print(f"Valid range: ${lower:.2f} - ${upper:.2f}")

# Remove outliers
df_no_outliers = df[(df['Price'] >= lower) & (df['Price'] <= upper)]

# Cap outliers instead of removing
df['Price_Capped'] = df['Price'].clip(lower=lower, upper=upper)

# Z-score method
from scipy import stats
z_scores = np.abs(stats.zscore(df['Price']))
df_no_outliers = df[z_scores < 3]  # Remove values with z-score > 3
```

---

## Data Manipulation & Transformation

Transform your data to extract insights and prepare for analysis.

### Creating New Columns

```python
# Simple calculations
df['Total_Value'] = df['Price'] * df['Quantity']
df['Price_Category'] = df['Price'].apply(lambda x: 'Expensive' if x > 100 else 'Affordable')

# Conditional logic with np.where
df['Stock_Status'] = np.where(df['Quantity'] > 50, 'High Stock', 
                     np.where(df['Quantity'] > 20, 'Medium Stock', 'Low Stock'))

# Multiple conditions with numpy.select
conditions = [
    df['Price'] < 50,
    (df['Price'] >= 50) & (df['Price'] < 200),
    df['Price'] >= 200
]
choices = ['Budget', 'Mid-range', 'Premium']
df['Price_Tier'] = np.select(conditions, choices, default='Unknown')

# Using apply with custom functions
def categorize_product(row):
    if row['Category'] == 'Electronics' and row['Price'] > 500:
        return 'High-end Electronics'
    elif row['Category'] == 'Accessories':
        return 'Accessory Item'
    else:
        return 'Standard Product'

df['Product_Type'] = df.apply(categorize_product, axis=1)
```

### Column Operations

```python
# Rename columns
df_renamed = df.rename(columns={
    'Product': 'Product_Name',
    'Price': 'Unit_Price'
})

# Rename with a function
df.columns = df.columns.str.lower().str.replace(' ', '_')

# Reorder columns
column_order = ['Product', 'Category', 'Price', 'Quantity', 'Total_Value']
df_reordered = df[column_order]

# Drop columns
df_reduced = df.drop(['Unnecessary_Column'], axis=1)
df_reduced = df.drop(columns=['Col1', 'Col2'])
```

### Row Operations

```python
# Add new rows
new_row = pd.DataFrame({
    'Product': ['Webcam'],
    'Price': [89.99],
    'Quantity': [25],
    'Category': ['Electronics']
})
df_extended = pd.concat([df, new_row], ignore_index=True)

# Remove rows by condition
df_filtered = df[df['Price'] <= 1000]  # Remove very expensive items

# Reset index after operations
df_clean = df.reset_index(drop=True)
```

### Data Transformation Methods

#### Apply Functions

```python
# Apply to single column
df['Price_Squared'] = df['Price'].apply(lambda x: x**2)

# Apply to multiple columns
df['Price_Per_Unit'] = df.apply(lambda row: row['Price'] / row['Quantity'], axis=1)

# Apply with additional arguments
def price_with_tax(price, tax_rate=0.1):
    return price * (1 + tax_rate)

df['Price_With_Tax'] = df['Price'].apply(price_with_tax, tax_rate=0.08)

# Apply to groups
df['Price_Rank_by_Category'] = df.groupby('Category')['Price'].rank(ascending=False)
```

#### Map and Replace

```python
# Map values using dictionary
category_map = {
    'Electronics': 'ELEC',
    'Accessories': 'ACC'
}
df['Category_Code'] = df['Category'].map(category_map)

# Replace specific values
df['Category'] = df['Category'].replace('Electronics', 'Electronic Items')

# Replace multiple values
df['Category'] = df['Category'].replace({
    'Electronics': 'Tech',
    'Accessories': 'Add-ons'
})

# Replace with regex
df['Product'] = df['Product'].str.replace(r'\d+', 'NUM', regex=True)
```

#### Sorting

```python
# Sort by single column
df_sorted = df.sort_values('Price')

# Sort by multiple columns
df_sorted = df.sort_values(['Category', 'Price'], ascending=[True, False])

# Sort by index
df_sorted = df.sort_index()

# Custom sorting
category_order = ['Electronics', 'Accessories', 'Other']
df['Category'] = pd.Categorical(df['Category'], categories=category_order, ordered=True)
df_sorted = df.sort_values('Category')
```

### Binning and Discretization

```python
# Equal-width binning
df['Price_Bins'] = pd.cut(df['Price'], bins=5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# Custom bins
price_bins = [0, 50, 100, 200, float('inf')]
bin_labels = ['Budget', 'Economy', 'Standard', 'Premium']
df['Price_Range'] = pd.cut(df['Price'], bins=price_bins, labels=bin_labels)

# Quantile-based binning
df['Price_Quartiles'] = pd.qcut(df['Price'], q=4, labels=['Q1', 'Q2', 'Q3', 'Q4'])
```

---

## Grouping & Aggregation

Group data and perform calculations on groups for powerful analysis.

### Basic Grouping

```python
# Group by single column
category_groups = df.groupby('Category')

# Basic aggregations
print("Average price by category:")
print(category_groups['Price'].mean())

print("\nTotal quantity by category:")
print(category_groups['Quantity'].sum())

print("\nCount of products by category:")
print(category_groups.size())

# Multiple aggregations
print("\nMultiple statistics by category:")
print(category_groups['Price'].agg(['mean', 'median', 'std', 'min', 'max']))
```

### Advanced Grouping

```python
# Group by multiple columns
multi_group = df.groupby(['Category', 'Price_Range'])
print(multi_group['Quantity'].sum())

# Custom aggregation functions
def price_range(series):
    return series.max() - series.min()

custom_stats = category_groups['Price'].agg([
    'mean',
    'median',
    ('range', price_range),
    ('count', 'size')
])
print(custom_stats)

# Different aggregations for different columns
agg_dict = {
    'Price': ['mean', 'min', 'max'],
    'Quantity': ['sum', 'mean'],
    'Product': 'count'
}
result = df.groupby('Category').agg(agg_dict)
print(result)
```

### Transform and Filter

```python
# Transform - returns same shape as original
df['Price_Normalized'] = df.groupby('Category')['Price'].transform(
    lambda x: (x - x.mean()) / x.std()
)

# Calculate percentage of category total
df['Quantity_Pct'] = df.groupby('Category')['Quantity'].transform(
    lambda x: x / x.sum() * 100
)

# Filter groups based on conditions
large_categories = df.groupby('Category').filter(lambda x: len(x) > 2)

# Groups with high average price
expensive_categories = df.groupby('Category').filter(
    lambda x: x['Price'].mean() > 100
)
```

### Rolling Operations (Time Series)

```python
# Assuming we have a date column
df['Date'] = pd.date_range('2024-01-01', periods=len(df), freq='D')
df = df.set_index('Date')

# Rolling averages
df['Price_7day_avg'] = df['Price'].rolling(window=7).mean()
df['Price_30day_avg'] = df['Price'].rolling(window=30).mean()

# Rolling sum
df['Quantity_7day_sum'] = df['Quantity'].rolling(window=7).sum()

# Expanding operations (cumulative)
df['Cumulative_Revenue'] = df['Total_Value'].expanding().sum()
df['Running_Average_Price'] = df['Price'].expanding().mean()
```

### Window Functions

```python
# Ranking within groups
df['Price_Rank'] = df.groupby('Category')['Price'].rank(ascending=False)
df['Quantity_Rank'] = df.groupby('Category')['Quantity'].rank()

# Percentage rank
df['Price_Percentile'] = df.groupby('Category')['Price'].rank(pct=True)

# Shift operations for comparisons
df['Previous_Price'] = df.groupby('Category')['Price'].shift(1)
df['Price_Change'] = df['Price'] - df['Previous_Price']

# Cumulative operations within groups
df['Cumulative_Quantity'] = df.groupby('Category')['Quantity'].cumsum()
df['Running_Max_Price'] = df.groupby('Category')['Price'].cummax()
```

### Pivot Tables

```python
# Basic pivot table
pivot = df.pivot_table(
    values='Price',
    index='Category',
    columns='Price_Range',
    aggfunc='mean'
)
print(pivot)

# Multiple value columns
pivot_multi = df.pivot_table(
    values=['Price', 'Quantity'],
    index='Category',
    columns='Price_Range',
    aggfunc={'Price': 'mean', 'Quantity': 'sum'},
    fill_value=0
)
print(pivot_multi)

# With margins (totals)
pivot_with_totals = df.pivot_table(
    values='Total_Value',
    index='Category',
    columns='Price_Range',
    aggfunc='sum',
    margins=True,
    margins_name='Total'
)
print(pivot_with_totals)
```

### Cross-tabulation

```python
# Simple cross-tab
crosstab = pd.crosstab(df['Category'], df['Price_Range'])
print(crosstab)

# With percentages
crosstab_pct = pd.crosstab(df['Category'], df['Price_Range'], normalize='index') * 100
print(crosstab_pct)

# With margins
crosstab_margins = pd.crosstab(
    df['Category'], 
    df['Price_Range'], 
    margins=True
)
print(crosstab_margins)
```

---

## Merging & Joining Data

Combine multiple datasets effectively.

### Sample DataFrames for Examples

```python
# Create sample dataframes
customers = pd.DataFrame({
    'customer_id': [1, 2, 3, 4, 5],
    'name': ['Alice', 'Bob', 'Charlie', 'Diana', 'Eve'],
    'city': ['New York', 'London', 'Tokyo', 'Paris', 'Sydney']
})

orders = pd.DataFrame({
    'order_id': [101, 102, 103, 104, 105, 106],
    'customer_id': [1, 2, 2, 3, 4, 7],  # Note: customer 7 doesn't exist in customers
    'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones', 'Tablet'],
    'amount': [999, 25, 80, 300, 150, 500]
})

products = pd.DataFrame({
    'product': ['Laptop', 'Mouse', 'Keyboard', 'Monitor', 'Headphones'],
    'category': ['Electronics', 'Accessories', 'Accessories', 'Electronics', 'Electronics'],
    'supplier': ['TechCorp', 'AccessCorp', 'AccessCorp', 'TechCorp', 'AudioCorp']
})
```

### Merge Operations

```python
# Inner join (default) - only matching records
inner_merge = pd.merge(customers, orders, on='customer_id')
print("Inner merge:")
print(inner_merge)

# Left join - all records from left dataframe
left_merge = pd.merge(customers, orders, on='customer_id', how='left')
print("\nLeft merge:")
print(left_merge)

# Right join - all records from right dataframe
right_merge = pd.merge(customers, orders, on='customer_id', how='right')
print("\nRight merge:")
print(right_merge)

# Outer join - all records from both dataframes
outer_merge = pd.merge(customers, orders, on='customer_id', how='outer')
print("\nOuter merge:")
print(outer_merge)
```

### Advanced Merging

```python
# Merge on different column names
df1 = pd.DataFrame({'id': [1, 2, 3], 'value': ['A', 'B', 'C']})
df2 = pd.DataFrame({'user_id': [1, 2, 4], 'score': [10, 20, 30]})

merged = pd.merge(df1, df2, left_on='id', right_on='user_id', how='outer')
print(merged)

# Merge on multiple columns
sales = pd.DataFrame({
    'year': [2023, 2023, 2024, 2024],
    'quarter': [1, 2, 1, 2],
    'revenue': [1000, 1200, 1100, 1300]
})

targets = pd.DataFrame({
    'year': [2023, 2023, 2024, 2024],
    'quarter': [1, 2, 1, 2],
    'target': [900, 1100, 1000, 1250]
})

performance = pd.merge(sales, targets, on=['year', 'quarter'])
performance['vs_target'] = performance['revenue'] - performance['target']
print(performance)

# Merge with suffixes for conflicting column names
df1 = pd.DataFrame({'id': [1, 2], 'value': [10, 20]})
df2 = pd.DataFrame({'id': [1, 2], 'value': [100, 200]})

merged = pd.merge(df1, df2, on='id', suffixes=('_left', '_right'))
print(merged)
```

### Join Operations

```python
# Join using index
df1_indexed = customers.set_index('customer_id')
df2_indexed = orders.set_index('customer_id')

joined = df1_indexed.join(df2_indexed, how='left')
print("Join result:")
print(joined)

# Multiple dataframe joins
customer_orders = pd.merge(customers, orders, on='customer_id')
full_data = pd.merge(customer_orders, products, on='product')
print("Three-way merge:")
print(full_data)
```

### Concatenation

```python
# Vertical concatenation (stacking dataframes)
df1 = pd.DataFrame({'A': [1, 2], 'B': [3, 4]})
df2 = pd.DataFrame({'A': [5, 6], 'B': [7, 8]})

vertical_concat = pd.concat([df1, df2], ignore_index=True)
print("Vertical concatenation:")
print(vertical_concat)

# Horizontal concatenation
horizontal_concat = pd.concat([df1, df2], axis=1)
print("Horizontal concatenation:")
print(horizontal_concat)

# Concatenation with keys
keyed_concat = pd.concat([df1, df2], keys=['first', 'second'])
print("Concatenation with keys:")
print(keyed_concat)

# Handling different columns
df3 = pd.DataFrame({'A': [9, 10], 'C': [11, 12]})
mixed_concat = pd.concat([df1, df3], sort=False)
print("Mixed columns concatenation:")
print(mixed_concat)
```

### Combining DataFrames

```python
# Combine_first - fill missing values from another dataframe
df1 = pd.DataFrame({'A': [1, None, 3], 'B': [4, 5, None]})
df2 = pd.DataFrame({'A': [None, 2, None], 'B': [None, None, 6]})

combined = df1.combine_first(df2)
print("Combine first:")
print(combined)

# Update - modify dataframe with values from another
df1.update(df2)
print("After update:")
print(df1)
```

---

## Time Series Analysis

Pandas excels at handling time-based data.

### Creating Time Series Data

```python
# Date range creation
dates = pd.date_range('2024-01-01', '2024-12-31', freq='D')
business_days = pd.date_range('2024-01-01', '2024-12-31', freq='B')
monthly_dates = pd.date_range('2024-01-01', '2024-12-31', freq='M')

# Create time series dataframe
np.random.seed(42)
ts_data = pd.DataFrame({
    'date': pd.date_range('2024-01-01', periods=365, freq='D'),
    'sales': np.random.randint(1000, 5000, 365),
    'temperature': np.random.normal(20, 10, 365),
    'day_of_week': pd.date_range('2024-01-01', periods=365, freq='D').day_name()
})

# Set date as index
ts_data.set_index('date', inplace=True)
print(ts_data.head())
```

### DateTime Operations

```python
# Parse string dates
date_strings = ['2024-01-15', '2024-02-20', '2024-03-25']
parsed_dates = pd.to_datetime(date_strings)

# Handle different date formats
mixed_dates = ['2024-01-15', '15/02/2024', 'March 25, 2024']
parsed_mixed = pd.to_datetime(mixed_dates, infer_datetime_format=True)

# Extract date components
ts_data['year'] = ts_data.index.year
ts_data['month'] = ts_data.index.month
ts_data['day'] = ts_data.index.day
ts_data['weekday'] = ts_data.index.dayofweek
ts_data['quarter'] = ts_data.index.quarter

# Day name and month name
ts_data['day_name'] = ts_data.index.day_name()
ts_data['month_name'] = ts_data.index.month_name()

print("Date components:")
print(ts_data[['year', 'month', 'day', 'weekday', 'day_name']].head())
```

### Time-based Indexing and Selection

```python
# Select specific date
single_day = ts_data.loc['2024-01-15']

# Select date range
january_data = ts_data.loc['2024-01-01':'2024-01-31']
q1_data = ts_data.loc['2024-01':'2024-03']

# Select by year
year_2024 = ts_data.loc['2024']

# Boolean indexing with dates
summer_data = ts_data[ts_data.index.month.isin([6, 7, 8])]
weekends = ts_data[ts_data.index.dayofweek >= 5]

# Recent data
last_30_days = ts_data.last('30D')
first_quarter = ts_data.first('3M')
```

### Resampling and Frequency Conversion

```python
# Resample to different frequencies
monthly_sales = ts_data['sales'].resample('M').sum()
weekly_avg_temp = ts_data['temperature'].resample('W').mean()
quarterly_stats = ts_data['sales'].resample('Q').agg(['sum', 'mean', 'std'])

print("Monthly sales:")
print(monthly_sales.head())

# Custom resampling
def sales_range(series):
    return series.max() - series.min()

weekly_stats = ts_data['sales'].resample('W').agg({
    'total': 'sum',
    'average': 'mean',
    'range': sales_range
})

# Upsampling and downsampling
daily_to_hourly = ts_data.resample('H').ffill()  # Forward fill
hourly_to_daily = daily_to_hourly.resample('D').mean()  # Average
```

### Time Shifts and Lags

```python
# Shift data
ts_data['sales_lag1'] = ts_data['sales'].shift(1)
ts_data['sales_lag7'] = ts_data['sales'].shift(7)  # One week lag
ts_data['sales_lead1'] = ts_data['sales'].shift(-1)  # Lead

# Calculate changes
ts_data['sales_change'] = ts_data['sales'] - ts_data['sales_lag1']
ts_data['sales_pct_change'] = ts_data['sales'].pct_change()

# Weekly comparison
ts_data['sales_wow_change'] = ts_data['sales'] - ts_data['sales_lag7']
ts_data['sales_wow_pct'] = ((ts_data['sales'] - ts_data['sales_lag7']) / ts_data['sales_lag7']) * 100

print("Sales with lags and changes:")
print(ts_data[['sales', 'sales_lag1', 'sales_change', 'sales_pct_change']].head(10))
```

### Rolling Operations

```python
# Moving averages
ts_data['sales_ma7'] = ts_data['sales'].rolling(window=7).mean()
ts_data['sales_ma30'] = ts_data['sales'].rolling(window=30).mean()

# Weighted moving average
weights = np.arange(1, 8)  # 1, 2, 3, 4, 5, 6, 7
ts_data['sales_wma7'] = ts_data['sales'].rolling(window=7).apply(
    lambda x: np.average(x, weights=weights)
)

# Rolling statistics
ts_data['sales_rolling_std'] = ts_data['sales'].rolling(window=30).std()
ts_data['sales_rolling_min'] = ts_data['sales'].rolling(window=7).min()
ts_data['sales_rolling_max'] = ts_data['sales'].rolling(window=7).max()

# Exponential weighted moving average
ts_data['sales_ewma'] = ts_data['sales'].ewm(span=7).mean()

print("Rolling statistics:")
print(ts_data[['sales', 'sales_ma7', 'sales_ma30', 'sales_ewma']].head(10))
```

### Seasonal Analysis

```python
# Group by time periods
monthly_avg = ts_data.groupby(ts_data.index.month)['sales'].mean()
daily_pattern = ts_data.groupby(ts_data.index.dayofweek)['sales'].mean()
hourly_pattern = ts_data.groupby(ts_data.index.hour)['sales'].mean()

print("Average sales by month:")
print(monthly_avg)

print("\nAverage sales by day of week:")
day_names = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
daily_pattern.index = day_names
print(daily_pattern)

# Seasonal decomposition (requires statsmodels)
from statsmodels.tsa.seasonal import seasonal_decompose

# Create more complex seasonal data
seasonal_ts = ts_data['sales'].rolling(window=7).mean().dropna()
decomposition = seasonal_decompose(seasonal_ts, model='additive', period=30)

# Plot components
fig, axes = plt.subplots(4, 1, figsize=(12, 10))
decomposition.observed.plot(ax=axes[0], title='Original')
decomposition.trend.plot(ax=axes[1], title='Trend')
decomposition.seasonal.plot(ax=axes[2], title='Seasonal')
decomposition.resid.plot(ax=axes[3], title='Residual')
plt.tight_layout()
plt.show()
```

---

## Advanced Operations

Unlock the full power of pandas with advanced techniques.

### Multi-Index (Hierarchical Indexing)

```python
# Create multi-index from columns
sales_data = pd.DataFrame({
    'Region': ['North', 'North', 'South', 'South', 'East', 'East'],
    'City': ['NYC', 'Boston', 'Miami', 'Atlanta', 'Tokyo', 'Seoul'],
    'Product': ['Laptop', 'Mouse', 'Laptop', 'Mouse', 'Laptop', 'Mouse'],
    'Sales': [1000, 50, 800, 40, 1200, 60],
    'Quantity': [5, 10, 4, 8, 6, 12]
})

# Set multi-index
multi_df = sales_data.set_index(['Region', 'City', 'Product'])
print("Multi-index DataFrame:")
print(multi_df)

# Access multi-index data
print("\nNorth region data:")
print(multi_df.loc['North'])

print("\nNYC data:")
print(multi_df.loc[('North', 'NYC')])

print("\nSpecific product in specific city:")
print(multi_df.loc[('North', 'NYC', 'Laptop')])

# Cross-section
print("\nAll Laptop sales:")
print(multi_df.xs('Laptop', level='Product'))

# Swap levels
swapped = multi_df.swaplevel('Region', 'Product')
print("\nSwapped levels:")
print(swapped.sort_index())
```

### Categorical Data

```python
# Create categorical data
categories = ['Low', 'Medium', 'High']
cat_data = pd.Categorical(['Medium', 'High', 'Low', 'Medium', 'High'], 
                         categories=categories, 
                         ordered=True)

df_cat = pd.DataFrame({
    'product': ['A', 'B', 'C', 'D', 'E'],
    'priority': cat_data,
    'score': [85, 92, 78, 88, 95]
})

print("Categorical data:")
print(df_cat)
print(f"Data types: {df_cat.dtypes}")

# Categorical operations
print("\nValue counts:")
print(df_cat['priority'].value_counts())

# Sort by categorical order
sorted_df = df_cat.sort_values('priority')
print("\nSorted by priority:")
print(sorted_df)

# Add categories
df_cat['priority'] = df_cat['priority'].cat.add_categories(['Critical'])

# Remove unused categories
df_cat['priority'] = df_cat['priority'].cat.remove_unused_categories()

# Memory benefit
regular_series = pd.Series(['A', 'B', 'C'] * 1000)
categorical_series = pd.Series(['A', 'B', 'C'] * 1000, dtype='category')

print(f"\nRegular series memory usage: {regular_series.memory_usage(deep=True)} bytes")
print(f"Categorical series memory usage: {categorical_series.memory_usage(deep=True)} bytes")
```

### Working with Text Data

```python
# Sample text data
text_data = pd.DataFrame({
    'name': ['John Doe', 'jane smith', 'Bob JOHNSON', 'Alice Brown'],
    'email': ['john.doe@email.com', 'JANE@COMPANY.COM', 'bob@work.org', 'alice.b@test.net'],
    'phone': ['(555) 123-4567', '555.987.6543', '5551112222', '555-444-3333'],
    'address': ['123 Main St, NYC, NY', '456 Oak Ave, LA, CA', '789 Pine Rd, Chicago, IL', '321 Elm St, Miami, FL']
})

# String operations
# Case conversion
text_data['name_proper'] = text_data['name'].str.title()
text_data['email_lower'] = text_data['email'].str.lower()

# String length
text_data['name_length'] = text_data['name'].str.len()

# Extract information
text_data['first_name'] = text_data['name'].str.split().str[0]
text_data['last_name'] = text_data['name'].str.split().str[-1]

# Extract domain from email
text_data['email_domain'] = text_data['email'].str.extract(r'@(.+)')

# Clean phone numbers
text_data['phone_clean'] = text_data['phone'].str.replace(r'[^\d]', '', regex=True)

# Extract state from address
text_data['state'] = text_data['address'].str.extract(r', ([A-Z]{2})$')

print("Text data operations:")
print(text_data[['name', 'name_proper', 'first_name', 'last_name', 'email_domain', 'state']])
```

### Custom Functions and Apply

```python
# Complex custom function
def analyze_product(row):
    """Analyze product performance"""
    score = 0
    
    # Price factor
    if row['Price'] > 500:
        score += 3
    elif row['Price'] > 100:
        score += 2
    else:
        score += 1
    
    # Quantity factor
    if row['Quantity'] > 100:
        score += 2
    elif row['Quantity'] > 50:
        score += 1
    
    # Category factor
    if row['Category'] == 'Electronics':
        score += 1
    
    # Determine performance level
    if score >= 5:
        return 'Excellent'
    elif score >= 3:
        return 'Good'
    else:
        return 'Fair'

# Apply custom function
df['Performance'] = df.apply(analyze_product, axis=1)

# Vectorized operations (faster)
def vectorized_analysis(price, quantity, category):
    score = np.where(price > 500, 3, np.where(price > 100, 2, 1))
    score += np.where(quantity > 100, 2, np.where(quantity > 50, 1, 0))
    score += np.where(category == 'Electronics', 1, 0)
    
    return np.where(score >= 5, 'Excellent', 
           np.where(score >= 3, 'Good', 'Fair'))

df['Performance_Vectorized'] = vectorized_analysis(df['Price'], df['Quantity'], df['Category'])

print("Performance analysis:")
print(df[['Product', 'Price', 'Quantity', 'Category', 'Performance']])
```

### Memory Optimization

```python
# Check memory usage
print(f"DataFrame memory usage: {df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")

# Optimize data types
def optimize_dtypes(df):
    """Optimize DataFrame data types for memory efficiency"""
    optimized_df = df.copy()
    
    for col in optimized_df.columns:
        col_type = optimized_df[col].dtype
        
        if col_type != 'object':
            c_min = optimized_df[col].min()
            c_max = optimized_df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    optimized_df[col] = optimized_df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    optimized_df[col] = optimized_df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    optimized_df[col] = optimized_df[col].astype(np.int32)
            
            elif str(col_type)[:5] == 'float':
                if c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    optimized_df[col] = optimized_df[col].astype(np.float32)
        
        else:
            # Convert to category if low cardinality
            if optimized_df[col].nunique() / len(optimized_df) < 0.5:
                optimized_df[col] = optimized_df[col].astype('category')
    
    return optimized_df

optimized_df = optimize_dtypes(df)
print(f"Optimized memory usage: {optimized_df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
```

---

## Performance Optimization

Make your pandas code faster and more efficient.

### Vectorization vs Loops

```python
import time

# Create sample data
large_df = pd.DataFrame({
    'values': np.random.randint(1, 100, 1000000)
})

# Bad: Using loops
start_time = time.time()
result_loop = []
for value in large_df['values']:
    if value > 50:
        result_loop.append(value * 2)
    else:
        result_loop.append(value)
loop_time = time.time() - start_time

# Good: Using vectorization
start_time = time.time()
result_vectorized = np.where(large_df['values'] > 50, 
                           large_df['values'] * 2, 
                           large_df['values'])
vectorized_time = time.time() - start_time

print(f"Loop time: {loop_time:.4f} seconds")
print(f"Vectorized time: {vectorized_time:.4f} seconds")
print(f"Speedup: {loop_time/vectorized_time:.1f}x faster")
```

### Efficient Data Loading

```python
# Use appropriate data types when reading
dtypes = {
    'category_col': 'category',
    'int_col': 'int32',
    'float_col': 'float32'
}

# Read only needed columns
df = pd.read_csv('large_file.csv', 
                usecols=['col1', 'col2', 'col3'],
                dtype=dtypes)

# Use chunking for very large files
chunk_size = 10000
processed_chunks = []

for chunk in pd.read_csv('huge_file.csv', chunksize=chunk_size):
    # Process each chunk
    processed_chunk = chunk[chunk['value'] > 100]
    processed_chunks.append(processed_chunk)

final_df = pd.concat(processed_chunks, ignore_index=True)
```

### Query Optimization

```python
# Use query() for complex filtering (can be faster)
# Traditional method
filtered1 = df[(df['Price'] > 100) & (df['Category'] == 'Electronics')]

# Query method
filtered2 = df.query('Price > 100 and Category == "Electronics"')

# Use categorical data for groupby operations
df['Category'] = df['Category'].astype('category')
grouped = df.groupby('Category')['Price'].mean()  # Faster with categorical
```

### Caching and Memoization

```python
from functools import lru_cache

@lru_cache(maxsize=128)
def expensive_calculation(value):
    """Simulate expensive calculation"""
    time.sleep(0.1)  # Simulate processing time
    return value ** 2 + value ** 0.5

# Use cached function
df['calculated'] = df['Price'].apply(expensive_calculation)
```

---

## Best Practices

Follow these guidelines for clean, efficient pandas code.

### Code Organization

```python
# Good: Clear, readable code with proper variable names
def analyze_sales_data(sales_df):
    """
    Analyze sales data and return summary statistics
    
    Parameters:
    sales_df (DataFrame): Sales data with columns: product, price, quantity, date
    
    Returns:
    dict: Summary statistics
    """
    # Calculate derived metrics
    sales_df['revenue'] = sales_df['price'] * sales_df['quantity']
    sales_df['date'] = pd.to_datetime(sales_df['date'])
    
    # Summary statistics
    summary = {
        'total_revenue': sales_df['revenue'].sum(),
        'avg_order_value': sales_df['revenue'].mean(),
        'total_orders': len(sales_df),
        'date_range': (sales_df['date'].min(), sales_df['date'].max())
    }
    
    return summary

# Bad: Unclear, hard to maintain
def process(d):
    d['x'] = d['a'] * d['b']
    return d['x'].sum()
```

### Error Handling

```python
def safe_data_processing(df, column_name):
    """Process data with proper error handling"""
    try:
        # Check if column exists
        if column_name not in df.columns:
            raise ValueError(f"Column '{column_name}' not found in DataFrame")
        
        # Check for empty DataFrame
        if df.empty:
            raise ValueError("DataFrame is empty")
        
        # Check data type
        if not pd.api.types.is_numeric_dtype(df[column_name]):
            raise TypeError(f"Column '{column_name}' must be numeric")
        
        # Perform processing
        result = df[column_name].mean()
        return result
        
    except Exception as e:
        print(f"Error processing data: {e}")
        return None

# Usage
result = safe_data_processing(df, 'Price')
if result is not None:
    print(f"Average price: ${result:.2f}")
```

### Data Validation

```python
def validate_sales_data(df):
    """Validate sales data integrity"""
    issues = []
    
    # Check required columns
    required_columns = ['product', 'price', 'quantity', 'date']
    missing_columns = [col for col in required_columns if col not in df.columns]
    if missing_columns:
        issues.append(f"Missing columns: {missing_columns}")
    
    # Check for negative values
    if (df['price'] < 0).any():
        issues.append("Negative prices found")
    
    if (df['quantity'] < 0).any():
        issues.append("Negative quantities found")
    
    # Check for missing values
    missing_data = df.isnull().sum()
    if missing_data.any():
        issues.append(f"Missing values: {missing_data[missing_data > 0].to_dict()}")
    
    # Check date format
    try:
        pd.to_datetime(df['date'])
    except:
        issues.append("Invalid date format")
    
    return issues

# Validate data
validation_issues = validate_sales_data(df)
if validation_issues:
    print("Data validation issues:")
    for issue in validation_issues:
        print(f"- {issue}")
else:
    print("Data validation passed ✓")
```

### Configuration and Constants

```python
# Configuration settings
CONFIG = {
    'date_format': '%Y-%m-%d',
    'currency_symbol': '$',
    'decimal_places': 2,
    'chunk_size': 10000,
    'max_memory_usage': 1024,  # MB
}

# Constants
PRODUCT_CATEGORIES = ['Electronics', 'Accessories', 'Software']
VALID_CURRENCIES = ['USD', 'EUR', 'GBP', 'JPY']
DATE_COLUMNS = ['order_date', 'ship_date', 'delivery_date']

def format_currency(value, currency='USD'):
    """Format currency consistently"""
    symbol = CONFIG['currency_symbol'] if currency == 'USD' else currency
    return f"{symbol}{value:.{CONFIG['decimal_places']}f}"
```

---

## Real-World Projects

Apply your pandas skills to realistic scenarios.

### Project 1: E-commerce Sales Analysis

```python
def ecommerce_analysis():
    """Complete e-commerce sales analysis"""
    
    # Load and prepare data
    orders = pd.read_csv('orders.csv')
    customers = pd.read_csv('customers.csv')
    products = pd.read_csv('products.csv')
    
    # Data cleaning
    orders['order_date'] = pd.to_datetime(orders['order_date'])
    orders['revenue'] = orders['price'] * orders['quantity']
    
    # Merge datasets
    sales_data = (orders
                 .merge(customers, on='customer_id')
                 .merge(products, on='product_id'))
    
    # Time-based analysis
    monthly_sales = (sales_data
                    .set_index('order_date')
                    .resample('M')['revenue']
                    .agg(['sum', 'count', 'mean']))
    
    # Customer analysis
    customer_metrics = (sales_data
                       .groupby('customer_id')
                       .agg({
                           'revenue': ['sum', 'count', 'mean'],
                           'order_date': ['min', 'max']
                       }))
    
    # Product performance
    product_analysis = (sales_data
                       .groupby(['category', 'product_name'])
                       .agg({
                           'revenue': 'sum',
                           'quantity': 'sum',
                           'customer_id': 'nunique'
                       })
                       .sort_values('revenue', ascending=False))
    
    # Cohort analysis
    def cohort_analysis(df):
        df['order_period'] = df['order_date'].dt.to_period('M')
        df['cohort_group'] = df.groupby('customer_id')['order_date'].transform('min').dt.to_period('M')
        
        period_number = (df['order_period'] - df['cohort_group']).apply(attrgetter('n'))
        df['period_number'] = period_number
        
        cohort_data = df.groupby(['cohort_group', 'period_number'])['customer_id'].nunique().reset_index()
        cohort_counts = cohort_data.pivot(index='cohort_group', columns='period_number', values='customer_id')
        
        cohort_sizes = df.groupby('cohort_group')['customer_id'].nunique()
        cohort_table = cohort_counts.divide(cohort_sizes, axis=0)
        
        return cohort_table
    
    cohort_retention = cohort_analysis(sales_data)
    
    return {
        'monthly_sales': monthly_sales,
        'customer_metrics': customer_metrics,
        'product_analysis': product_analysis,
        'cohort_retention': cohort_retention
    }
```

### Project 2: Financial Data Analysis

```python
def financial_analysis():
    """Comprehensive financial data analysis"""
    
    # Load stock price data
    stock_data = pd.read_csv('stock_prices.csv')
    stock_data['date'] = pd.to_datetime(stock_data['date'])
    stock_data.set_index('date', inplace=True)
    
    # Calculate technical indicators
    def calculate_indicators(df):
        # Moving averages
        df['MA_20'] = df['close'].rolling(window=20).mean()
        df['MA_50'] = df['close'].rolling(window=50).mean()
        
        # Volatility
        df['volatility'] = df['close'].rolling(window=20).std()
        
        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        df['RSI'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        df['BB_upper'] = df['MA_20'] + (df['volatility'] * 2)
        df['BB_lower'] = df['MA_20'] - (df['volatility'] * 2)
        
        return df
    
    # Apply to each stock
    stocks = stock_data['symbol'].unique()
    processed_stocks = []
    
    for stock in stocks:
        stock_subset = stock_data[stock_data['symbol'] == stock].copy()
        stock_subset = calculate_indicators(stock_subset)
        processed_stocks.append(stock_subset)
    
    final_data = pd.concat(processed_stocks)
    
    # Portfolio analysis
    def portfolio_metrics(returns):
        annual_return = returns.mean() * 252
        annual_volatility = returns.std() * np.sqrt(252)
        sharpe_ratio = annual_return / annual_volatility
        max_drawdown = (returns.cumsum() - returns.cumsum().expanding().max()).min()
        
        return {
            'annual_return': annual_return,
            'annual_volatility': annual_volatility,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown': max_drawdown
        }
    
    # Calculate daily returns
    returns_data = (final_data
                   .groupby('symbol')['close']
                   .pct_change()
                   .unstack(level='symbol'))
    
    portfolio_stats = {}
    for stock in stocks:
        if stock in returns_data.columns:
            portfolio_stats[stock] = portfolio_metrics(returns_data[stock].dropna())
    
    return final_data, portfolio_stats
```

### Project 3: Data Quality Report

```python
def data_quality_report(df, report_name="Data Quality Report"):
    """Generate comprehensive data quality report"""
    
    report = {
        'report_name': report_name,
        'generated_at': pd.Timestamp.now(),
        'dataset_info': {},
        'column_analysis': {},
        'data_issues': []
    }
    
    # Basic dataset information
    report['dataset_info'] = {
        'shape': df.shape,
        'memory_usage_mb': df.memory_usage(deep=True).sum() / (1024**2),
        'duplicate_rows': df.duplicated().sum(),
        'total_missing_values': df.isnull().sum().sum()
    }
    
    # Column-by-column analysis
    for column in df.columns:
        col_analysis = {
            'dtype': str(df[column].dtype),
            'non_null_count': df[column].count(),
            'null_count': df[column].isnull().sum(),
            'null_percentage': (df[column].isnull().sum() / len(df)) * 100,
            'unique_count': df[column].nunique(),
            'unique_percentage': (df[column].nunique() / len(df)) * 100
        }
        
        if pd.api.types.is_numeric_dtype(df[column]):
            col_analysis.update({
                'mean': df[column].mean(),
                'median': df[column].median(),
                'std': df[column].std(),
                'min': df[column].min(),
                'max': df[column].max(),
                'zeros_count': (df[column] == 0).sum(),
                'negative_count': (df[column] < 0).sum()
            })
        
        elif pd.api.types.is_string_dtype(df[column]):
            col_analysis.update({
                'avg_length': df[column].str.len().mean(),
                'max_length': df[column].str.len().max(),
                'min_length': df[column].str.len().min(),
                'empty_strings': (df[column] == '').sum()
            })
        
        report['column_analysis'][column] = col_analysis
    
    # Identify data issues
    for column, analysis in report['column_analysis'].items():
        if analysis['null_percentage'] > 50:
            report['data_issues'].append(f"Column '{column}' has {analysis['null_percentage']:.1f}% missing values")
        
        if analysis['unique_count'] == 1:
            report['data_issues'].append(f"Column '{column}' has only one unique value")
        
        if pd.api.types.is_numeric_dtype(df[column]):
            if analysis.get('negative_count', 0) > 0 and column.lower() in ['price', 'quantity', 'amount']:
                report['data_issues'].append(f"Column '{column}' has {analysis['negative_count']} negative values")
    
    return report

# Generate and display report
quality_report = data_quality_report(df, "Sales Data Quality Report")

print(f"=== {quality_report['report_name']} ===")
print(f"Generated: {quality_report['generated_at']}")
print(f"\nDataset Shape: {quality_report['dataset_info']['shape']}")
print(f"Memory Usage: {quality_report['dataset_info']['memory_usage_mb']:.2f} MB")
print(f"Duplicate Rows: {quality_report['dataset_info']['duplicate_rows']}")

if quality_report['data_issues']:
    print(f"\n⚠️  Data Issues Found:")
    for issue in quality_report['data_issues']:
        print(f"  • {issue}")
else:
    print(f"\n✅ No major data issues found!")
```

---

## 🎯 Practice Exercises

Test your pandas skills with these hands-on exercises:

### Beginner Level

1. **Data Loading**: Load a CSV file and display basic information
    
2. **Data Cleaning**: Remove duplicates and handle missing values
    
3. **Basic Analysis**: Calculate mean, median, and mode for numeric columns
    
4. **Filtering**: Select rows based on conditions
    
5. **Grouping**: Group by category and calculate aggregates
    

### Intermediate Level

1. **Time Series**: Analyze sales trends over time
    
2. **Merging**: Combine multiple datasets
    
3. **Pivot Tables**: Create summary tables
    
4. **Text Processing**: Clean and extract information from text columns
    
5. **Visualization**: Create charts using pandas plotting
    

### Advanced Level

1. **Performance Optimization**: Optimize memory usage and speed
    
2. **Custom Functions**: Create complex analysis functions
    
3. **Multi-Index**: Work with hierarchical data
    
4. **Rolling Operations**: Calculate moving averages and statistics
    
5. **Real Project**: Complete end-to-end data analysis
    

---

## 🔗 Additional Resources

### Official Documentation

* [Pandas Documentation](https://pandas.pydata.org/docs/)
    
* [Pandas User Guide](https://pandas.pydata.org/pandas-docs/stable/user_guide/index.html)
    
* [Pandas API Reference](https://pandas.pydata.org/pandas-docs/stable/reference/index.html)
    

### Tutorials and Learning

* [10 Minutes to Pandas](https://pandas.pydata.org/pandas-docs/stable/user_guide/10min.html)
    
* [Pandas Cookbook](https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html)
    
* [Real Python Pandas Tutorials](https://realpython.com/pandas-python-explore-dataset/)
    

### Performance and Optimization

* [Enhancing Performance](https://pandas.pydata.org/pandas-docs/stable/user_guide/enhancingperf.html)
    
* [Pandas Performance Tips](https://pandas.pydata.org/pandas-docs/stable/user_guide/scale.html)
    

### Community and Help

* [Stack Overflow - Pandas](https://stackoverflow.com/questions/tagged/pandas)
    
* [GitHub - Pandas Issues](https://github.com/pandas-dev/pandas/issues)
    
* [Reddit - r/pandas](https://www.reddit.com/r/pandas/)
    

---

## 📊 Quick Reference Cheat Sheet

### Essential Operations

```python
# Data Loading
df = pd.read_csv('file.csv')
df = pd.read_excel('file.xlsx')

# Basic Info
df.info()
df.describe()
df.head()

# Selection
df['column']
df[['col1', 'col2']]
df.loc[0:5, 'column']
df.iloc[0:5, 0:3]

# Filtering
df[df['column'] > 100]
df.query('column > 100')

# Grouping
df.groupby('category').mean()
df.groupby(['cat1', 'cat2']).agg({'col': 'sum'})

# Merging
pd.merge(df1, df2, on='key')
pd.concat([df1, df2])

# Saving
df.to_csv('output.csv', index=False)
df.to_excel('output.xlsx', index=False)
```

---

Congratulations! You now have a comprehensive guide to mastering pandas. Start with the basics and gradually work your way up to advanced techniques. Remember: practice makes perfect! 🐼✨