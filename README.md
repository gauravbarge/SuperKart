# SuperKart Sales Forecasting — Model Deployment Project

> **GreatLearning — Model Deployment Capstone Project**
> Predict quarterly product-level sales revenue across SuperKart's store network and deploy the solution as a live REST API + interactive web application.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Business Problem](#2-business-problem)
3. [Dataset](#3-dataset)
4. [Project Structure](#4-project-structure)
5. [Solution Architecture](#5-solution-architecture)
6. [Methodology](#6-methodology)
7. [Feature Engineering](#7-feature-engineering)
8. [Model Building & Evaluation](#8-model-building--evaluation)
9. [Hyperparameter Tuning](#9-hyperparameter-tuning)
10. [Final Model Selection](#10-final-model-selection)
11. [Deployment — Backend (Flask API)](#11-deployment--backend-flask-api)
12. [Deployment — Frontend (Streamlit UI)](#12-deployment--frontend-streamlit-ui)
13. [Live Hugging Face Spaces](#13-live-hugging-face-spaces)
14. [How to Run Locally](#14-how-to-run-locally)
15. [API Reference](#15-api-reference)
16. [Key Insights & Business Recommendations](#16-key-insights--business-recommendations)
17. [Submission Files](#17-submission-files)
18. [Tech Stack](#18-tech-stack)

---

## 1. Project Overview

SuperKart is a retail chain operating supermarkets and food marts across Tier 1, Tier 2, and Tier 3 cities. This project builds an end-to-end machine learning pipeline that:

- Performs **exploratory data analysis** to surface revenue patterns across products and stores.
- Engineers **meaningful features** from raw product and store attributes.
- Trains and tunes **two ensemble ML models** (Random Forest and XGBoost) for regression.
- **Serializes** the best model pipeline and deploys it as a production-grade REST API using **Flask + Gunicorn** on a **Hugging Face Docker Space**.
- Provides an **interactive Streamlit web app** that lets users enter product/store attributes and instantly get a sales forecast.

| Metric | Value |
|---|---|
| Dataset size | 8,763 rows × 12 columns |
| Target variable | `Product_Store_Sales_Total` ($) |
| Problem type | Regression |
| Best model | Random Forest (Tuned) |
| Best Test R² | **0.6672** |
| Backend URL | https://gauravbarge-superkart-backend.hf.space |
| Frontend URL | https://gauravbarge-superkart-frontend.hf.space |

---

## 2. Business Problem

SuperKart wants to accurately forecast the **total sales revenue per product per store** for the upcoming quarter to:

- Optimise **inventory management** — stock the right products at the right stores.
- Drive **regional sales strategy** — allocate resources to highest-revenue regions.
- Enable **data-driven decision making** across 4 stores and 16 product categories.
- Reduce **forecast risk** — minimise over/under-stocking costs.

The solution is not just a notebook — it is a **deployed forecasting system** that can be queried in real-time via a REST API and a no-code web interface.

---

## 3. Dataset

**File:** `SuperKart.csv`  
**Rows:** 8,763 | **Columns:** 12 | **Missing values:** `Product_Weight` (~28% rows)

### Data Dictionary

| Column | Type | Description |
|---|---|---|
| `Product_Id` | string | Unique product identifier. Prefix: `FD` = Food, `DR` = Drinks, `NC` = Non-consumable |
| `Product_Weight` | float | Weight of the product in kg. Range: 4.0 – 21.4 |
| `Product_Sugar_Content` | string | `Low Sugar`, `Regular`, `No Sugar` (also `reg` — standardised to `Regular`) |
| `Product_Allocated_Area` | float | Ratio of product's display area to total store display area. Range: 0.0 – 0.33 |
| `Product_Type` | string | One of 16 categories: Fruits and Vegetables, Snack Foods, Household, Frozen Foods, Dairy, Canned, Baking Goods, Health and Hygiene, Soft Drinks, Meat, Breads, Hard Drinks, Others, Starchy Foods, Breakfast, Seafood |
| `Product_MRP` | float | Maximum Retail Price in $. Range: 31 – 267 |
| `Store_Id` | string | `OUT001`, `OUT002`, `OUT003`, `OUT004` |
| `Store_Establishment_Year` | int | Year the store was established. Range: 1987 – 2009 |
| `Store_Size` | string | `Small`, `Medium`, `High` |
| `Store_Location_City_Type` | string | `Tier 1`, `Tier 2`, `Tier 3` |
| `Store_Type` | string | `Food Mart`, `Supermarket Type1`, `Supermarket Type2`, `Departmental Store` |
| `Product_Store_Sales_Total` | float | **TARGET** — Total revenue generated ($). Range: $33 – $8,000. Mean: $3,464. Median: $3,452 |

### Store Summary

| Store ID | Type | City Tier | Size | Est. Year | Age (2025) | Products | Total Revenue |
|---|---|---|---|---|---|---|---|
| OUT004 | Supermarket Type2 | Tier 2 | Medium | 2009 | 16 yrs | 4,676 | **$15,427,583** |
| OUT003 | Departmental Store | Tier 1 | Medium | 1999 | 26 yrs | 1,349 | $6,673,458 |
| OUT001 | Supermarket Type1 | Tier 2 | High | 1987 | 38 yrs | 1,586 | $6,223,113 |
| OUT002 | Food Mart | Tier 3 | Small | 1998 | 27 yrs | 1,152 | $2,030,910 |
| **Total** | | | | | | **8,763** | **$30,355,064** |

### Target Variable Statistics

| Stat | Value |
|---|---|
| Min | $33.00 |
| 25th Percentile | $2,761.72 |
| Median | $3,452.34 |
| Mean | $3,464.00 |
| 75th Percentile | $4,145.17 |
| Max | $8,000.00 |
| Std Dev | $1,065.63 |

---

## 4. Project Structure

```
SuperKart/
│
├── SuperKart.csv                                          # Raw dataset
├── ProblemStatmentAndRubrics.pdf                          # Assignment rubric
│
├── Full_Code_SuperKart_Model_Deployment_Notebook_[Updated].ipynb   # Source notebook (no outputs)
├── Full_Code_Executed.ipynb                               # Notebook with all cell outputs
├── Full_Code_SuperKart_Solution.html                      # HTML export for submission (2.2 MB)
│
├── Low_Code_SuperKart_Model_Deployment_Notebook_[Updated]_version_2.ipynb  # Reference template
│
├── backend_files/                                         # Flask API (deployed to HF Docker Space)
│   ├── app.py                                             # Flask application
│   ├── requirements.txt                                   # Python dependencies
│   ├── Dockerfile                                         # Container definition
│   └── superkart_model.joblib                             # Serialized best model (~254 KB)
│
├── frontend_files/                                        # Streamlit UI (deployed to HF Docker Space)
│   ├── app.py                                             # Streamlit application
│   ├── requirements.txt                                   # Python dependencies
│   └── Dockerfile                                         # Container definition
│
├── .gitignore
└── README.md
```

---

## 5. Solution Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                        USER / BUSINESS                          │
└─────────────────────────┬───────────────────────────────────────┘
                          │  Input: product + store attributes
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              FRONTEND  (Hugging Face Streamlit Space)           │
│  • Streamlit web app                                            │
│  • 10 input fields (product weight, MRP, store type, etc.)      │
│  • Calls backend API via HTTP POST                              │
│  • Displays predicted $ sales revenue                           │
│  URL: https://gauravbarge-superkart-frontend.hf.space           │
└─────────────────────────┬───────────────────────────────────────┘
                          │  POST /v1/predict  (JSON payload)
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              BACKEND   (Hugging Face Docker Space)              │
│  • Flask REST API served by Gunicorn (4 workers)                │
│  • Loads serialized Random Forest pipeline (joblib)             │
│  • Preprocesses input → OneHotEncodes categoricals              │
│  • Returns predicted sales as JSON {"Sales": <float>}           │
│  URL: https://gauravbarge-superkart-backend.hf.space            │
└─────────────────────────┬───────────────────────────────────────┘
                          │  sklearn Pipeline.predict()
                          ▼
┌─────────────────────────────────────────────────────────────────┐
│              ML PIPELINE  (inside joblib model)                 │
│  ColumnTransformer (OneHotEncoder on categorical cols)          │
│       ↓                                                         │
│  RandomForestRegressor (max_depth=15, n_estimators=100)         │
└─────────────────────────────────────────────────────────────────┘
```

---

## 6. Methodology

### Step 1 — Data Overview
- Checked shape (8763 × 12), dtypes, missing values, duplicates.
- Identified `Product_Weight` with missing values (~28% of rows).
- No duplicate rows found.

### Step 2 — Exploratory Data Analysis (EDA)

**Univariate Analysis:**
- `Product_Weight`: Roughly uniform/bimodal distribution (4–22 kg); a few outliers above 20 kg.
- `Product_Allocated_Area`: Strongly right-skewed; most products have small display area ratios.
- `Product_MRP`: Multimodal distribution with peaks at ~$70, ~$130, and ~$200 — three clear price tiers.
- `Product_Store_Sales_Total` (target): Right-skewed; most products between $500–$4,000; high-value outliers above $6,000.
- `Product_Sugar_Content`: Regular (60%), Low Sugar (29%), No Sugar (11%).
- `Product_Type`: Fruits & Vegetables, Snack Foods, and Household are the top 3 by count.
- `Store_Id`: OUT004 accounts for ~53% of all product records.
- `Store_Size`: Medium (53%), High (32%), Small (15%).
- `Store_Location_City_Type`: Tier 3 (53%), Tier 1 (24%), Tier 2 (24%).

**Bivariate Analysis:**
- `Product_MRP` has the strongest positive correlation with the target (r ≈ 0.57).
- `Store_Establishment_Year` has a weak negative correlation (newer stores slightly outperform).
- OUT004 generates $15.4M — more than the other three stores combined.
- Fruits & Vegetables and Snack Foods lead revenue in every store.
- OUT003 (Departmental Store, Tier 1) achieves the highest per-product revenue range ($3,070–$8,000).

### Step 3 — Data Preprocessing

| Issue | Treatment | Rationale |
|---|---|---|
| `Product_Sugar_Content` has `reg` entries | Replace `reg` → `Regular` | Standardise category labels |
| `Product_Weight` missing (~28%) | Impute with **median** | Median is robust to skewed distribution and outliers |
| Outliers in `Product_Allocated_Area` and target | **No treatment** | Using tree-based models (RF, XGBoost) that are inherently robust to outliers; removing them would discard genuine high-revenue edge cases |
| Categorical encoding | **OneHotEncoder** in sklearn Pipeline | Handles unseen categories gracefully (`handle_unknown='ignore'`) |

---

## 7. Feature Engineering

Three new features were created from existing columns:

| New Feature | Derived From | Logic | Rationale |
|---|---|---|---|
| `Product_Id_char` | `Product_Id` | First 2 characters of the ID | Encodes broad product category: `FD`=Food, `DR`=Drinks, `NC`=Non-consumable. More informative than full Product_Id for modeling. |
| `Store_Age_Years` | `Store_Establishment_Year` | `2025 − Store_Establishment_Year` | Captures store maturity; established stores have loyal customer bases and proven demand patterns. |
| `Product_Type_Category` | `Product_Type` | Map 16 types → 2 groups | `Perishables` (Dairy, Meat, Fruits & Veg, Breakfast, Breads, Seafood) vs `Non Perishables`. Reduces dimensionality while capturing key supply-chain characteristics. |

**Dropped columns:**

| Column | Reason |
|---|---|
| `Product_Id` | Replaced by `Product_Id_char`; full ID is too high-cardinality |
| `Product_Type` | Replaced by `Product_Type_Category`; 16-level feature collapsed to 2 |
| `Store_Id` | Too few stores (4) to generalise; store characteristics captured by `Store_Type`, `Store_Size`, `Store_Location_City_Type`, `Store_Age_Years` |
| `Store_Establishment_Year` | Replaced by `Store_Age_Years` |

**Final feature set (10 features):**

| Feature | Type |
|---|---|
| `Product_Weight` | Numeric |
| `Product_Sugar_Content` | Categorical |
| `Product_Allocated_Area` | Numeric |
| `Product_MRP` | Numeric |
| `Store_Size` | Categorical |
| `Store_Location_City_Type` | Categorical |
| `Store_Type` | Categorical |
| `Product_Id_char` | Categorical |
| `Store_Age_Years` | Numeric |
| `Product_Type_Category` | Categorical |

**Train/Test split:** 70% train (6,134 rows) / 30% test (2,629 rows), `random_state=1`, shuffled.

---

## 8. Model Building & Evaluation

### Metric of Choice: R² (R-squared)

**Primary metric: R²** — chosen because:
- Intuitive interpretation: R²=0.67 means the model explains 67% of the variance in sales revenue.
- Directly meaningful to business stakeholders: "our model captures 67% of what drives sales."
- Standard metric for regression; directly optimised in GridSearchCV.

Additional metrics reported: **RMSE**, **MAE**, **MAPE**, **Adjusted R²**.

### Preprocessing Pipeline

```python
preprocessor = make_column_transformer(
    (Pipeline([('encoder', OneHotEncoder(handle_unknown='ignore'))]),
     categorical_features)
)
```

Each model is wrapped in a `make_pipeline(preprocessor, model)` so preprocessing is applied consistently at train and inference time — no data leakage.

### Model 1: Random Forest Regressor

**Rationale:** Random Forest builds many independent decision trees via bootstrap sampling (bagging) and averages their predictions. It handles non-linear relationships, is robust to outliers, and provides implicit feature importance — a good baseline ensemble.

```python
rf_estimator = RandomForestRegressor(random_state=1, n_jobs=-1)
```

| Dataset | RMSE | MAE | R² | Adj. R² | MAPE |
|---|---|---|---|---|---|
| Train | — | — | **0.6855** | — | — |
| Test | — | — | **0.6672** | — | — |

### Model 2: XGBoost Regressor

**Rationale:** XGBoost builds trees sequentially where each tree corrects the residual errors of the previous one (boosting). It has built-in L1/L2 regularisation, handles sparse one-hot features efficiently, and is generally state-of-the-art for structured tabular data.

```python
xgb_estimator = XGBRegressor(random_state=1, n_jobs=-1)
```

| Dataset | R² |
|---|---|
| Train | **0.6855** |
| Test | **0.6669** |

---

## 9. Hyperparameter Tuning

**Method:** `GridSearchCV` with 3-fold cross-validation, scoring=`r2`, `n_jobs=-1`.

### Random Forest Tuning Grid

```python
parameters = {
    'randomforestregressor__max_depth': [10, 15, 20],
    'randomforestregressor__max_features': ['sqrt', 'log2'],
    'randomforestregressor__n_estimators': [100, 200],
}
```

**Best parameters:** `max_depth=15`, `n_estimators=100`, `max_features='sqrt'`

| Dataset | R² (Base) | R² (Tuned) | Change |
|---|---|---|---|
| Train | 0.6855 | 0.6855 | — |
| Test | 0.6672 | **0.6672** | Stable |

### XGBoost Tuning Grid

```python
parameters = {
    'xgbregressor__n_estimators': [100, 200],
    'xgbregressor__subsample': [0.7, 1.0],
    'xgbregressor__gamma': [0, 1],
    'xgbregressor__colsample_bytree': [0.7, 1.0],
    'xgbregressor__colsample_bylevel': [0.7, 1.0],
}
```

**Best parameters:** `gamma=0`, `n_estimators=100`, `subsample=0.7`

| Dataset | R² (Base) | R² (Tuned) | Change |
|---|---|---|---|
| Train | 0.6855 | 0.6855 | — |
| Test | 0.6669 | 0.6665 | Marginal |

---

## 10. Final Model Selection

### Model Comparison (Test Set)

| Model | Test R² | Rank |
|---|---|---|
| **Random Forest (Base)** | **0.6672** | **🥇 1st** |
| **Random Forest (Tuned)** | **0.6672** | **🥇 1st** |
| XGBoost (Base) | 0.6669 | 3rd |
| XGBoost (Tuned) | 0.6665 | 4th |

### Selected Model: **Random Forest (Best Parameters)**

**Rationale:**
1. Highest test R² (0.6672) — best generalization to unseen product-store combinations.
2. Robust to outliers inherently — no preprocessing of extreme sales values required.
3. Training and test R² gap is minimal (0.6855 vs 0.6672 = 0.018) — no significant overfitting.
4. No regularisation tuning required — simpler hyperparameter space than XGBoost.
5. Directly interpretable feature importances for future business analysis.

**Serialized model:** `backend_files/superkart_model.joblib` (≈ 254 KB)

```python
# Save
joblib.dump(rf_tuned, 'backend_files/superkart_model.joblib')

# Load and predict
model = joblib.load('backend_files/superkart_model.joblib')
predictions = model.predict(X_test)
```

---

## 11. Deployment — Backend (Flask API)

### Framework: Flask + Gunicorn

**Location:** `backend_files/`

### `app.py`

```python
superkart_api = Flask("SuperKart Sales Forecast API")
model = joblib.load("superkart_model.joblib")

@superkart_api.get('/')
def home():
    return "Welcome to the SuperKart Sales Forecasting API!"

@superkart_api.post('/v1/predict')
def predict_sales():
    data = request.get_json()
    input_df = pd.DataFrame([{
        'Product_Weight': data['Product_Weight'],
        'Product_Sugar_Content': data['Product_Sugar_Content'],
        ...
    }])
    prediction = model.predict(input_df).tolist()[0]
    return jsonify({'Sales': prediction})
```

### `Dockerfile`

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip install --no-cache-dir --upgrade -r requirements.txt
CMD ["gunicorn", "-w", "4", "-b", "0.0.0.0:7860", "app:superkart_api"]
```

### `requirements.txt`

```
pandas==2.2.2
numpy==2.0.2
scikit-learn==1.6.1
joblib==1.4.2
xgboost==2.1.4
Werkzeug==2.2.2
flask==2.2.2
gunicorn==20.1.0
```

**Hosted on:** Hugging Face Docker Space — `gauravbarge/superkart-backend`

---

## 12. Deployment — Frontend (Streamlit UI)

### Framework: Streamlit

**Location:** `frontend_files/`

### Input fields

| Field | UI Element | Values |
|---|---|---|
| Product Weight | Number input | 0.0 – 50.0 kg |
| Product Sugar Content | Selectbox | Low Sugar, Regular, No Sugar |
| Product Allocated Area | Number input | 0.0 – 1.0 |
| Product MRP | Number input | $0 – $500 |
| Store Size | Selectbox | Small, Medium, High |
| Store Location City Type | Selectbox | Tier 1, Tier 2, Tier 3 |
| Store Type | Selectbox | Food Mart, Supermarket Type1, Supermarket Type2, Departmental Store |
| Product ID Prefix | Selectbox | FD, DR, NC |
| Store Age | Number input | 1 – 100 years |
| Product Type Category | Selectbox | Perishables, Non Perishables |

On "Predict Sales" click → sends POST to backend → displays `${prediction:.2f}`.

### `Dockerfile`

```dockerfile
FROM python:3.9-slim
WORKDIR /app
COPY . .
RUN pip3 install -r requirements.txt
CMD ["streamlit", "run", "app.py", "--server.port=7860",
     "--server.address=0.0.0.0", "--server.enableXsrfProtection=false"]
```

**Hosted on:** Hugging Face Docker Space — `gauravbarge/superkart-frontend`

---

## 13. Live Hugging Face Spaces

| Component | URL | SDK |
|---|---|---|
| 🔧 **Backend (Flask API)** | https://huggingface.co/spaces/gauravbarge/superkart-backend | Docker |
| 🖥️ **Frontend (Streamlit UI)** | https://huggingface.co/spaces/gauravbarge/superkart-frontend | Docker |

> **Note:** Spaces on the free tier may take 30–60 seconds to wake up on first request after a period of inactivity.

---

## 14. How to Run Locally

### Prerequisites

```bash
python >= 3.9
```

### 1. Clone the repo

```bash
git clone git@github.com:gauravbarge/SuperKart.git
cd SuperKart
```

### 2. Install dependencies

```bash
pip install numpy==2.0.2 pandas==2.2.2 scikit-learn==1.6.1 matplotlib==3.10.0 \
            seaborn==0.13.2 joblib==1.4.2 xgboost==2.1.4 requests==2.32.4 \
            huggingface_hub==0.34.0 flask==2.2.2 gunicorn==20.1.0 streamlit==1.45.0
```

> **macOS users:** XGBoost requires OpenMP. Run `brew install libomp` if you see a `libxgboost.dylib` error.

### 3. Run the notebook

Open `Full_Code_SuperKart_Model_Deployment_Notebook_[Updated].ipynb` in Jupyter and run all cells. This will:
- Train the models and serialize the best one to `backend_files/superkart_model.joblib`
- Write `backend_files/app.py`, `backend_files/Dockerfile`, etc.
- Write `frontend_files/app.py`, `frontend_files/Dockerfile`, etc.

### 4. Run the backend locally

```bash
cd backend_files
flask --app app run --port 5000
# or with gunicorn:
gunicorn -w 4 -b 0.0.0.0:5000 app:superkart_api
```

### 5. Run the frontend locally

Update the API URL in `frontend_files/app.py` from the HF Space URL to `http://localhost:5000/v1/predict`, then:

```bash
cd frontend_files
streamlit run app.py
```

### 6. Docker (optional)

```bash
# Backend
cd backend_files
docker build -t superkart-backend .
docker run -p 7860:7860 superkart-backend

# Frontend
cd frontend_files
docker build -t superkart-frontend .
docker run -p 7860:7860 superkart-frontend
```

---

## 15. API Reference

### Base URL

```
https://gauravbarge-superkart-backend.hf.space
```

### `GET /`

Health check / welcome message.

**Response:**
```
Welcome to the SuperKart Sales Forecasting API! Use POST /v1/predict to get sales predictions.
```

### `POST /v1/predict`

Predict total sales revenue for a product-store combination.

**Request body (JSON):**

```json
{
  "Product_Weight": 12.66,
  "Product_Sugar_Content": "Regular",
  "Product_Allocated_Area": 0.06,
  "Product_MRP": 150.0,
  "Store_Size": "Medium",
  "Store_Location_City_Type": "Tier 2",
  "Store_Type": "Supermarket Type2",
  "Product_Id_char": "FD",
  "Store_Age_Years": 16,
  "Product_Type_Category": "Non Perishables"
}
```

**Response (JSON):**

```json
{
  "Sales": 3512.47
}
```

**Example with curl:**

```bash
curl -X POST https://gauravbarge-superkart-backend.hf.space/v1/predict \
  -H "Content-Type: application/json" \
  -d '{
    "Product_Weight": 12.66,
    "Product_Sugar_Content": "Regular",
    "Product_Allocated_Area": 0.06,
    "Product_MRP": 150.0,
    "Store_Size": "Medium",
    "Store_Location_City_Type": "Tier 2",
    "Store_Type": "Supermarket Type2",
    "Product_Id_char": "FD",
    "Store_Age_Years": 16,
    "Product_Type_Category": "Non Perishables"
  }'
```

**Example with Python:**

```python
import requests

payload = {
    "Product_Weight": 12.66,
    "Product_Sugar_Content": "Regular",
    "Product_Allocated_Area": 0.06,
    "Product_MRP": 150.0,
    "Store_Size": "Medium",
    "Store_Location_City_Type": "Tier 2",
    "Store_Type": "Supermarket Type2",
    "Product_Id_char": "FD",
    "Store_Age_Years": 16,
    "Product_Type_Category": "Non Perishables"
}

response = requests.post(
    "https://gauravbarge-superkart-backend.hf.space/v1/predict",
    json=payload
)
print(response.json())  # {'Sales': 3512.47}
```

### Valid Input Values

| Field | Valid Values |
|---|---|
| `Product_Sugar_Content` | `"Low Sugar"`, `"Regular"`, `"No Sugar"` |
| `Store_Size` | `"Small"`, `"Medium"`, `"High"` |
| `Store_Location_City_Type` | `"Tier 1"`, `"Tier 2"`, `"Tier 3"` |
| `Store_Type` | `"Food Mart"`, `"Supermarket Type1"`, `"Supermarket Type2"`, `"Departmental Store"` |
| `Product_Id_char` | `"FD"` (Food), `"DR"` (Drinks), `"NC"` (Non-consumable) |
| `Product_Type_Category` | `"Perishables"`, `"Non Perishables"` |

---

## 16. Key Insights & Business Recommendations

### Insight 1 — Product MRP is the Strongest Sales Lever (r = 0.57)
Higher-priced products consistently generate more revenue. The multimodal MRP distribution reveals three price tiers ($31–90, $91–180, $181–267).

**Recommendation:** Actively shift product mix toward mid-range and premium items in stores with high-purchasing-power customers (Tier 1, Departmental Store). Premium pricing strategy can significantly boost revenue without needing more inventory count.

---

### Insight 2 — OUT004 Generates 51% of All Revenue
OUT004 (Supermarket Type2, Tier 2, Medium, 16 years old) generates $15.4M — more than OUT001 + OUT002 + OUT003 combined. It has the most products (4,676 vs 1,152–1,586 for others).

**Recommendation:** Conduct a detailed audit of OUT004's product assortment, store layout, and pricing strategy. Replicate its most successful practices — particularly the Fruits & Vegetables and Snack Foods mix — in OUT001 and OUT003.

---

### Insight 3 — Fruits & Vegetables and Snack Foods Drive the Most Revenue
These two categories are top revenue contributors in every single store regardless of size, type, or city tier.

**Recommendation:** Increase shelf space allocation for these categories, negotiate better procurement deals to protect margins, and run targeted promotions (bundle offers, loyalty discounts) to grow basket size.

---

### Insight 4 — Tier 3 Has Volume, Tier 1 Has Value
- Tier 3 cities: 53% of all transactions but lower per-product revenue (Food Mart model, $33–$2,300).
- Tier 1 cities: Only 24% of transactions but highest per-product revenue ($3,070–$8,000 in Departmental Stores).

**Recommendation:** Maintain separate inventory and pricing strategies by tier. Tier 3: focus on affordable staples in bulk. Tier 1: focus on premium, high-MRP products and experiential retail. Do not apply a one-size-fits-all pricing policy across tiers.

---

### Insight 5 — Food Mart (OUT002) Has Growth Potential in Tier 3
OUT002 generates only $2.03M due to its small size, but it serves a Tier 3 market that has the highest transaction volume citywise.

**Recommendation:** Expand OUT002's footprint or open additional Food Mart outlets in other Tier 3 cities. Focus on fast-moving consumer goods (Fruits & Vegetables, Snack Foods) which are already its top sellers. A 2× expansion could capture significant untapped Tier 3 demand.

---

### Insight 6 — Low Sugar Segment is the Fastest-Growing Opportunity
Low Sugar products: 29% of products, growing health-conscious consumer base. No Sugar: 11%. Regular: 60%.

**Recommendation:** Gradually expand Low Sugar and No Sugar product ranges, especially in Tier 1 Departmental Stores where customers have higher disposable income and greater health awareness. Partner with health-focused brands to create a premium health aisle.

---

### Insight 7 — Older Stores Need Infrastructure Investment
OUT001 (est. 1987, age 38 years) and OUT002 (est. 1998, age 27 years) are the oldest stores. Established stores have loyal customers but may suffer from outdated infrastructure.

**Recommendation:** Prioritise renovation of stores older than 25 years — modern shelving, digital payment systems, improved cold-chain for perishables. Historical data shows Departmental Stores (modern, Tier 1) achieve the highest per-product revenue, suggesting infrastructure investment has strong ROI.

---

### Insight 8 — The Deployed Model Enables Real-Time Forecasting
The REST API can be integrated directly into SuperKart's ERP or inventory management system to:
1. **Quarterly inventory planning** — predict demand before placing procurement orders.
2. **New store setup** — estimate expected revenue for a proposed store given its planned attributes.
3. **What-if analysis** — simulate the revenue impact of changing a product's MRP or store display area before committing to changes.
4. **Regional budget allocation** — rank stores/regions by predicted revenue to prioritise investment.

---

## 17. Submission Files

| File | Purpose |
|---|---|
| `Full_Code_SuperKart_Solution.html` | **Primary submission** — full notebook executed with all outputs, exported to HTML (2.2 MB) |
| `Full_Code_Executed.ipynb` | Executed notebook with outputs (alternative reference) |
| Hugging Face Backend Space | https://huggingface.co/spaces/gauravbarge/superkart-backend |
| Hugging Face Frontend Space | https://huggingface.co/spaces/gauravbarge/superkart-frontend |

---

## 18. Tech Stack

| Layer | Technology | Version |
|---|---|---|
| Language | Python | 3.9+ |
| Data manipulation | pandas | 2.2.2 |
| Numerical computing | numpy | 2.0.2 |
| ML framework | scikit-learn | 1.6.1 |
| Gradient boosting | xgboost | 2.1.4 |
| Model serialization | joblib | 1.4.2 |
| Visualization | matplotlib | 3.10.0 |
| Visualization | seaborn | 0.13.2 |
| Web framework (API) | Flask | 2.2.2 |
| WSGI server | Gunicorn | 20.1.0 |
| Frontend UI | Streamlit | 1.45.0 |
| Containerization | Docker | (python:3.9-slim base) |
| Model hosting | Hugging Face Spaces | Docker SDK |
| Version control | Git + GitHub | — |

---

## Author

**Gaurav Barge**
- GitHub: [@gauravbarge](https://github.com/gauravbarge)
- Hugging Face: [@gauravbarge](https://huggingface.co/gauravbarge)
- Course: GreatLearning — Model Deployment (Module 7)

---

*README generated as part of the SuperKart Model Deployment capstone project.*
