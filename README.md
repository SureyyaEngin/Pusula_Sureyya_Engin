# Physical Medicine & Rehabilitation Dataset - EDA & Preprocessing

## **Project Overview**
 This project focuses on **cleaning and preprocessing** a medical dataset containing information about patients undergoing physical therapy treatments.  
 The goal was to prepare the dataset for **predictive modeling** with the target variable being **`TedaviSuresi`** (treatment duration).

The project includes:
- **Exploratory Data Analysis (EDA)**
- **Data Cleaning & Preprocessing**
- **Feature Engineering**
- **Encoding and Normalization**
- Final preparation of **train-test splits** for machine learning.

---

## **Dataset Description**

- **Original Size:** `2235 rows × 13 columns`
- **Target Variable:** `TedaviSuresi`  
  Represents the treatment duration in **number of sessions**.

| **Column Name**     | **Description**                            |
|----------------------|--------------------------------------------|
| HastaNo             | Unique anonymized patient ID                |
| Yas                  | Age of the patient                         |
| Cinsiyet             | Gender                                    |
| KanGrubu             | Blood type                                |
| Uyruk                | Nationality                               |
| KronikHastalik       | Chronic diseases (comma-separated list)   |
| Bolum                | Department or clinic                      |
| Alerji               | Allergies (may be multiple)               |
| Tanilar              | Diagnoses (comma-separated list)          |
| TedaviAdi            | Treatment name                            |
| TedaviSuresi         | Target variable - treatment sessions      |
| UygulamaYerleri      | Application body locations (comma-separated list) |
| UygulamaSuresi       | Duration of each application in minutes   |

---

## **1. Exploratory Data Analysis (EDA)**

### **Initial Dataset Insights**
- **Missing Values Identified:**
  | Column            | Missing Count | % Missing |
  |-------------------|--------------|-----------|
  | Alerji           | 944          | ~42%      |
  | KanGrubu         | 675          | ~30%      |
  | KronikHastalik   | 611          | ~27%      |
  | UygulamaYerleri  | 221          | ~10%      |
  | Cinsiyet         | 169          | ~8%       |
  | Tanilar          | 75           | ~3%       |
  | Bolum            | 11           | <1%       |

- **Numeric Variables:**
  - `TedaviSuresi` ranged between **1 and 37 sessions**, median **15 sessions**.
  - `UygulamaSuresi` ranged between **3 and 45 minutes**, median **20 minutes**.

- **Duplications:**
  - `HastaNo` appeared multiple times because patients can have multiple treatments.

---

## **2. Data Preprocessing Steps**

### **Step-by-Step Process**

#### **Step 1 - Numeric Conversion**
Converted `TedaviSuresi` and `UygulamaSuresi` from text (e.g., `"15 Seans"`, `"20 Dakika"`) to **numeric values**.

```python
df['TedaviSuresi'] = df['TedaviSuresi'].str.extract('(\\d+)').astype(float)
df['UygulamaSuresi'] = df['UygulamaSuresi'].str.extract('(\\d+)').astype(float)
```

#### **Step 2 - Handling Missing Values**
- Categorical Columns:
    - KanGrubu → filled with "Unknown".
    - Cinsiyet & Bolum → filled with mode (most frequent value).
- List Columns:
    - KronikHastalik, Alerji → missing values replaced with "None".
    - Tanilar, UygulamaYerleri → missing values replaced with "Unknown".

#### **Step 3 - Splitting Multi-Value Columns**
Columns like KronikHastalik, Alerji, Tanilar, and UygulamaYerleri contain comma-separated lists.
These were split into Python lists to prepare for multi-hot encoding.
```python
multi_value_cols = ['KronikHastalik', 'Alerji', 'Tanilar', 'UygulamaYerleri']
for col in multi_value_cols:
    df[col] = df[col].apply(lambda x: [i.strip() for i in x.split(',')] if isinstance(x, str) else [])
```

#### **Step 4 - Multi-Hot Encoding**
Each unique value across the list columns was transformed into a binary feature.

Example:
```python
KronikHastalik = ['Diabetes', 'Hypertension']
→ KronikHastalik_Diabetes = 1
→ KronikHastalik_Hypertension = 1
```
This significantly increased the number of features.

#### **Step 5 - One-Hot Encoding of Remaining Categoricals**
Standard one-hot encoding was applied to:
- Cinsiyet, KanGrubu, Uyruk, Bolum

```python
df = pd.get_dummies(df, columns=['Cinsiyet', 'KanGrubu', 'Uyruk', 'Bolum'], drop_first=True)
```

#### **Step 6 - Normalization**
Numerical features such as Yas and UygulamaSuresi were standardized using StandardScaler.
```python
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
df[['Yas', 'UygulamaSuresi']] = scaler.fit_transform(df[['Yas', 'UygulamaSuresi']])
```
#### **Step 7 - Preparing Final Dataset**
- Dropped HastaNo since it's just an identifier.
- Defined target variable y = TedaviSuresi.
- Final feature matrix X was created for modeling.
```python
X = df.drop(columns=['TedaviSuresi', 'HastaNo'])
y = df['TedaviSuresi']
```
#### **Step 8 - Train-Test Split**
The data was split into training (80%) and testing (20%) sets.
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Final Dataset Stats**
  | Dataset           | Rows | Columns |
  |------------------ |------|---------|
  | Final Features (X)| 2235 | 375     |
  | Training Set      | 1788 | 375     |
  | Testing Set       | 447  | 375     |

## **4. Key Results**
- Dataset is now fully clean, numeric, and model-ready.
- All missing values handled.
- Categorical variables encoded properly.
- Numerical variables standardized.
- Target distribution:
```bash
count     2235.000000
mean       14.57
std         3.72
min         1.00
max        37.00
```
## **5. Libraries Used**
- pandas – data manipulation
- numpy – numerical operations
- matplotlib & seaborn – visualization
- scikit-learn – preprocessing and splitting

## **5. How to Run**
Clone the repository:
```bash
git clone https://github.com/yourusername/physical-medicine-eda.git
cd physical-medicine-eda
```

Install required libraries:
```bash
pip install -r requirements.txt
```

Run the notebook:
```bash
jupyter notebook notebooks/Pusula_Süreyya_Engin.ipynb
```