# 📘 EDA & Feature Engineering — Complete A-to-Z Notes

> **Purpose**: A structured cheat-sheet so you can independently perform Exploratory Data Analysis (EDA)
> and Feature Engineering (FE) on *any* tabular dataset — with the Google Play Store dataset as a running example.

---

## Table of Contents
1. [The Big Picture — What Is EDA & FE?](#1-the-big-picture)
2. [Step 0 — Import Libraries](#2-step-0--import-libraries)
3. [Step 1 — Load & Inspect the Data](#3-step-1--load--inspect-the-data)
4. [Step 2 — Data Cleaning](#4-step-2--data-cleaning)
5. [Step 3 — Feature Engineering](#5-step-3--feature-engineering)
6. [Step 4 — Exploratory Data Analysis (Answering Questions)](#6-step-4--exploratory-data-analysis)
7. [Key Pandas Functions Cheat Sheet](#7-key-pandas-functions-cheat-sheet)
8. [Common Patterns & Recipes](#8-common-patterns--recipes)
9. [Tips & Best Practices](#9-tips--best-practices)

---

## 1. The Big Picture

| Term | Meaning |
|------|---------|
| **EDA** | Exploring your data to understand its shape, distributions, patterns, and problems *before* modeling. |
| **Feature Engineering** | Creating / transforming columns so they become more useful for analysis or ML models. |

### Typical Workflow
```
Load Data → Inspect → Clean → Engineer Features → Analyze / Visualize → Model (optional)
```

---

## 2. Step 0 — Import Libraries

```python
import pandas as pd          # DataFrames & data manipulation
import numpy as np           # Numerical operations
import matplotlib.pyplot as plt  # Basic plotting
import seaborn as sns        # Statistical plotting (built on matplotlib)
import warnings
warnings.filterwarnings("ignore")  # Suppress noisy warnings

%matplotlib inline           # Show plots inside Jupyter
```

**Why each library?**
- `pandas` → 90% of your data work (loading, cleaning, grouping, filtering)
- `numpy` → Math helpers (`np.nan`, `np.mean`, array ops)
- `matplotlib` → Low-level charts (bar, line, scatter, hist)
- `seaborn` → High-level statistical charts (heatmap, countplot, boxplot)

---

## 3. Step 1 — Load & Inspect the Data

### 3.1 Loading
```python
df = pd.read_csv('path/to/file.csv')       # from local CSV
df = pd.read_csv('https://url/file.csv')   # from URL
df = pd.read_excel('file.xlsx')            # from Excel
```

### 3.2 First Look
```python
df.head()          # First 5 rows — see what the data looks like
df.head(10)        # First 10 rows
df.tail()          # Last 5 rows
df.sample(5)       # 5 random rows
```

### 3.3 Shape & Structure
```python
df.shape           # (rows, columns) — e.g. (10841, 13)
df.info()          # Column names, non-null counts, dtypes
df.dtypes          # Just the data types
df.columns         # List of all column names
df.describe()      # Statistical summary (count, mean, std, min, max, quartiles)
```

**🔑 Key Insight from `df.info()`:** If a column you expect to be numeric shows `object` dtype → it contains strings that need cleaning before analysis.

> **Play Store Example:** `Reviews` column was `object` because one row had the value `"3.0M"` instead of a number. `Installs` had `"10,000+"` (commas and plus sign). `Price` had `"$4.99"` (dollar sign).

---

## 4. Step 2 — Data Cleaning

Data cleaning is about fixing problems so your data is ready for analysis. Think of it as 5 sub-tasks:

### 4.1 Finding & Handling Missing Values (NaN / null)

```python
# Count missing values per column
df.isnull().sum()

# Percentage of missing values
(df.isnull().sum() / len(df)) * 100
```

**Options for handling:**
```python
# Option A: Drop rows with ANY null
df.dropna()

# Option B: Drop rows where a SPECIFIC column is null
df.dropna(subset=['Rating'])

# Option C: Fill with a value
df['Rating'].fillna(df['Rating'].median(), inplace=True)   # median
df['Column'].fillna(df['Column'].mode()[0], inplace=True)   # mode (most frequent)
df['Column'].fillna(0, inplace=True)                        # zero
df['Column'].fillna('Unknown', inplace=True)                # string placeholder
```

**Which to use?**
- If < 5% rows have nulls → `dropna()` is usually fine
- If a column has too many nulls (> 50%) → consider dropping the whole column
- If the column is important → fill with median (numeric) or mode (categorical)

### 4.2 Finding & Removing Duplicates

```python
# Count duplicates
df.duplicated().sum()

# See the duplicate rows
df[df.duplicated()]

# Remove duplicates (keep first occurrence)
df.drop_duplicates(inplace=True)

# Remove duplicates based on specific column(s)
df.drop_duplicates(subset=['App'], keep='first', inplace=True)
```

### 4.3 Fixing Data Types

This is **super common** and **super important**. Many columns look numeric but are stored as strings.

```python
# Convert string to integer
df['Reviews'] = df['Reviews'].astype(int)

# Convert string to float
df['Rating'] = df['Rating'].astype(float)

# Convert to datetime
df['Last Updated'] = pd.to_datetime(df['Last Updated'])
```

### 4.4 Cleaning String Columns (Removing unwanted characters)

This is one of the most critical skills. Real-world data has messy strings.

**The `.str.replace()` Pattern:**
```python
# Remove commas and plus signs from "10,000+"
df['Installs'] = df['Installs'].str.replace(',', '')
df['Installs'] = df['Installs'].str.replace('+', '')
df['Installs'] = df['Installs'].astype(int)

# Remove dollar sign from "$4.99"
df['Price'] = df['Price'].str.replace('$', '')
df['Price'] = df['Price'].astype(float)

# Remove "M" or "k" from Size column (e.g., "19M", "201k")
# This is more complex — need conditional logic
```

**The Chain Pattern (cleaner way):**
```python
# Do multiple replacements and type conversion in one go
df['Installs'] = df['Installs'].str.replace('[,+]', '', regex=True).astype(int)
df['Price'] = df['Price'].str.replace('$', '', regex=False).astype(float)
```

### 4.5 Handling Bad / Corrupted Rows

Sometimes a row has shifted columns (data entry error). You need to find and remove it.

```python
# Example: Find rows where 'Reviews' isn't a number
df[~df['Reviews'].str.isnumeric()]

# Drop that specific row
df = df.drop(df.index[10472])
# OR
df = df[df['Reviews'].str.isnumeric()]
```

### 4.6 Always Work on a Copy!
```python
df_copy = df.copy()   # Now work on df_copy, keep original df safe
```
This way if you mess up, you still have the original data.

---

## 5. Step 3 — Feature Engineering

Feature Engineering = Creating new useful columns from existing ones.

### 5.1 Extracting from Dates
```python
df['Last Updated'] = pd.to_datetime(df['Last Updated'])

df['Year']  = df['Last Updated'].dt.year
df['Month'] = df['Last Updated'].dt.month
df['Day']   = df['Last Updated'].dt.day
df['DayOfWeek'] = df['Last Updated'].dt.dayofweek  # 0=Monday, 6=Sunday
```

### 5.2 Binning / Categorizing Continuous Values
```python
# Create rating categories
df['Rating_Category'] = pd.cut(df['Rating'],
                                bins=[0, 2, 3, 4, 5],
                                labels=['Poor', 'Average', 'Good', 'Excellent'])
```

### 5.3 Log Transformation (for skewed data)
```python
df['Log_Installs'] = np.log1p(df['Installs'])  # log1p = log(1 + x), avoids log(0) error
```

### 5.4 Boolean / Flag Features
```python
df['Is_Free'] = (df['Type'] == 'Free').astype(int)      # 1 if Free, 0 if Paid
df['Has_High_Rating'] = (df['Rating'] >= 4.0).astype(int)
```

### 5.5 Extracting from Strings
```python
# Get primary genre from "Art & Design;Creativity" → "Art & Design"
df['Primary_Genre'] = df['Genres'].str.split(';').str[0]
```

---

## 6. Step 4 — Exploratory Data Analysis (Answering Questions)

This is where it all comes together. Here's how to answer the most common types of questions:

---

### ❓ Q1: "Which Category has the largest number of installations?"

**Logic:** Group by Category → Sum installations → Sort → Pick the top one.

```python
df.groupby('Category')['Installs'].sum().sort_values(ascending=False).head(1)
```

**Breaking it down step by step:**
```python
# Step 1: Group by Category and sum Installs
category_installs = df.groupby('Category')['Installs'].sum()

# Step 2: Sort from highest to lowest
category_installs_sorted = category_installs.sort_values(ascending=False)

# Step 3: See the top result
print(category_installs_sorted.head(1))
# Answer: GAME has the most total installations (~35 Billion)
```

**Visualize it:**
```python
top_10 = category_installs_sorted.head(10)
top_10.plot(kind='bar', figsize=(12, 6), color='skyblue')
plt.title('Top 10 Categories by Total Installations')
plt.ylabel('Total Installs')
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()
```

---

### ❓ Q2: "Top 5 most installed apps in each popular category?"

**Logic:** For each category → Sort by Installs → Pick top 5.

```python
# Method 1: Using groupby + nlargest
top5_per_category = df.groupby('Category').apply(
    lambda x: x.nlargest(5, 'Installs')[['App', 'Installs']]
).reset_index(drop=True)
```

**Or for specific categories:**
```python
popular_categories = ['GAME', 'COMMUNICATION', 'SOCIAL', 'TOOLS', 'PRODUCTIVITY']

for cat in popular_categories:
    print(f"\n--- {cat} ---")
    subset = df[df['Category'] == cat]
    top5 = subset.nlargest(5, 'Installs')[['App', 'Installs']]
    print(top5.to_string(index=False))
```

---

### ❓ Q3: "How many apps have a 5-star rating?"

**Logic:** Filter where Rating == 5 → Count.

```python
count_5_star = df[df['Rating'] == 5.0].shape[0]
print(f"Number of apps with 5-star rating: {count_5_star}")

# OR equivalently:
count_5_star = (df['Rating'] == 5.0).sum()
```

---

### ❓ Q4: "Distribution of app categories?"

```python
# Count apps per category
df['Category'].value_counts()

# As percentages
df['Category'].value_counts(normalize=True) * 100

# Visualize
df['Category'].value_counts().head(15).plot(kind='barh', figsize=(10, 8))
plt.title('App Count by Category')
plt.xlabel('Number of Apps')
plt.show()
```

---

### ❓ Q5: "Free vs Paid apps?"

```python
df['Type'].value_counts()
df['Type'].value_counts().plot(kind='pie', autopct='%1.1f%%', figsize=(6, 6))
plt.title('Free vs Paid Apps')
plt.show()
```

---

### ❓ Q6: "Average rating by category?"

```python
df.groupby('Category')['Rating'].mean().sort_values(ascending=False)
```

---

### ❓ Q7: "Correlation between numeric columns?"

```python
# Select only numeric columns
numeric_df = df.select_dtypes(include=[np.number])

# Correlation matrix
corr = numeric_df.corr()

# Heatmap
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.show()
```

---

## 7. Key Pandas Functions Cheat Sheet

### Filtering
```python
df[df['Rating'] > 4.0]                           # Single condition
df[(df['Rating'] > 4.0) & (df['Type'] == 'Free')]  # AND (&)
df[(df['Rating'] > 4.0) | (df['Type'] == 'Free')]  # OR (|)
df[df['Category'].isin(['GAME', 'SOCIAL'])]       # In a list
df[~df['Reviews'].str.isnumeric()]                 # NOT (~)
```

### Sorting
```python
df.sort_values('Installs', ascending=False)             # Sort descending
df.sort_values(['Category', 'Rating'], ascending=[True, False])  # Multi-sort
```

### Grouping & Aggregation
```python
df.groupby('Category')['Installs'].sum()         # Sum per group
df.groupby('Category')['Rating'].mean()          # Mean per group
df.groupby('Category')['App'].count()            # Count per group

# Multiple aggregations at once
df.groupby('Category').agg({
    'Installs': 'sum',
    'Rating': 'mean',
    'App': 'count'
})
```

### Value Counts
```python
df['Category'].value_counts()                    # Count frequency of each value
df['Category'].value_counts(normalize=True)      # As proportions (0 to 1)
df['Category'].nunique()                         # Number of unique values
df['Category'].unique()                          # Array of unique values
```

### Top N
```python
df.nlargest(5, 'Installs')                       # Top 5 by Installs
df.nsmallest(5, 'Rating')                        # Bottom 5 by Rating
```

### String Operations
```python
df['Col'].str.lower()                            # Lowercase
df['Col'].str.upper()                            # Uppercase
df['Col'].str.strip()                            # Remove whitespace
df['Col'].str.contains('Game')                   # Boolean: contains substring
df['Col'].str.replace('old', 'new')              # Replace substring
df['Col'].str.split(';')                         # Split into list
df['Col'].str.split(';').str[0]                  # Get first element after split
df['Col'].str.len()                              # Length of each string
df['Col'].str.isnumeric()                        # Boolean: is it all digits?
```

### DateTime Operations
```python
pd.to_datetime(df['Date_Col'])                   # Convert to datetime
df['Date_Col'].dt.year                           # Extract year
df['Date_Col'].dt.month                          # Extract month
df['Date_Col'].dt.day                            # Extract day
df['Date_Col'].dt.dayofweek                      # Day of week (0=Mon)
df['Date_Col'].dt.month_name()                   # "January", "February", etc.
```

---

## 8. Common Patterns & Recipes

### Pattern 1: "Clean a messy column and convert type"
```python
# 1. See what's in it
print(df['Column'].unique())

# 2. Remove unwanted characters
df['Column'] = df['Column'].str.replace('[^0-9.]', '', regex=True)

# 3. Handle special values
df['Column'] = df['Column'].replace('', np.nan)

# 4. Convert type
df['Column'] = df['Column'].astype(float)
```

### Pattern 2: "Find top N of something per group"
```python
df.groupby('Group_Col').apply(
    lambda x: x.nlargest(N, 'Value_Col')
).reset_index(drop=True)
```

### Pattern 3: "Compare two groups"
```python
free = df[df['Type'] == 'Free']
paid = df[df['Type'] == 'Paid']

print(f"Free avg rating: {free['Rating'].mean():.2f}")
print(f"Paid avg rating: {paid['Rating'].mean():.2f}")
```

### Pattern 4: "Create a summary table"
```python
summary = df.groupby('Category').agg(
    Total_Apps=('App', 'count'),
    Avg_Rating=('Rating', 'mean'),
    Total_Installs=('Installs', 'sum'),
    Max_Reviews=('Reviews', 'max')
).sort_values('Total_Installs', ascending=False)
```

---

## 9. Tips & Best Practices

1. **Always inspect first** — Run `df.head()`, `df.info()`, `df.describe()`, `df.isnull().sum()` before anything else.

2. **Work on a copy** — `df_copy = df.copy()` keeps your original data safe.

3. **Check dtypes** — If a column shows `object` but should be numeric, it has dirty strings.

4. **Clean before analyzing** — You **cannot** do math on strings. Clean → Convert type → Then analyze.

5. **Use `.value_counts()` for categorical** — It's the fastest way to understand categorical distributions.

6. **Use `.groupby()` for "per group" questions** — Any time you hear "by category", "per type", "for each..." → think `groupby`.

7. **Use `.nlargest()` / `.nsmallest()` for "top/bottom N"** — Cleaner than sorting + head.

8. **Visualize distributions** — Use histograms for numeric, bar charts for categorical, scatter for relationships.

9. **Check for outliers** — `df.describe()` reveals min/max. If max Rating is 19.0 → something is wrong!

10. **Name your variables clearly** — `top5_games_by_installs` is better than `x`.

---

## Quick Reference: The EDA Checklist ✅

```
□ Load data (pd.read_csv)
□ df.head(), df.shape, df.info(), df.describe()
□ Check nulls: df.isnull().sum()
□ Check duplicates: df.duplicated().sum()
□ Check dtypes — fix any object columns that should be numeric
□ Clean string columns (remove $, +, commas, M, k etc.)
□ Convert cleaned columns to proper types (int, float, datetime)
□ Remove duplicates
□ Handle missing values (drop or fill)
□ Engineer new features (from dates, bins, flags, splits)
□ Analyze: groupby, value_counts, filtering, nlargest
□ Visualize: bar, hist, pie, heatmap, scatter
□ Document insights
```

---

> 💡 **Remember:** EDA is not about memorizing code — it's about knowing *what question to ask*
> and then knowing *which Pandas function answers it*. The patterns above cover 90% of real-world EDA tasks.
