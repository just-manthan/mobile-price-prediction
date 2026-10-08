# 📱 Feature Extraction and Price Prediction for Mobile Phones

Predicting mobile phone prices from their specifications and identifying the features that drive price, to support a mobile retailer's pricing and marketing strategy.

## 🔑 Key results
| | |
|---|---|
| **Best model** | Gradient Boosting Regressor |
| **Average error (MAE)** | **Rs 1,730** per phone |
| **R² (test set)** | **0.94** |
| **5-fold CV R²** | 0.95 |
| **MAPE** | 9.4 % |

**Top price drivers:** front camera, rear camera, processor tier, storage, RAM and brand.
**Non-drivers:** colour (all colour variants sell at the same price) and AI lens.

| Model | MAE (Rs) | RMSE (Rs) | R² | CV R² |
|---|---|---|---|---|
| **Gradient Boosting** | **1,730** | **3,239** | **0.94** | **0.95** |
| Random Forest | 2,479 | 5,026 | 0.85 | 0.92 |
| Decision Tree | 2,960 | 4,737 | 0.86 | 0.87 |
| Linear Regression | 3,019 | 6,923 | 0.71 | 0.91 |

![Feature importance](figures/11_feature_importance_grouped.png)

📄 **Full report with all charts, insights and recommendations: [reports/REPORT.md](reports/REPORT.md)**

## Project workflow
| Step | What was done |
|---|---|
| **1. Data exploration** | 541 listings × 12 columns; data types, ranges, duplicates (`reports/eda_summary.txt`) |
| **2. Preprocessing** | Parsed text to numbers (`50MP` → 50); removed 10 exact duplicates and 216 colour-variant duplicates (avoids train/test leakage); fixed an out-of-range value; capped price outliers |
| **3. Feature engineering** | `brand` from model name; `processor_family` and `processor_tier` (Entry / Mid / Upper-mid / Flagship) parsed from 100+ messy processor strings; `colour_group`; one-hot encoding |
| **4. Feature extraction** | Pearson & Spearman correlation, mutual information, SelectKBest (F-test), PCA |
| **5. Model building** | Linear Regression, Decision Tree, Random Forest, Gradient Boosting on log(price); 80/20 split |
| **6. Evaluation** | MAE, RMSE, R², MAPE + 5-fold cross-validation |
| **7. Feature importance** | Impurity + permutation importance, grouped back to original features |
| **8. Report** | Insights and 7 business recommendations (`reports/REPORT.md`) |

## Dataset
Flipkart mobile listings: Model, Colour, Memory, RAM, Battery, Rear Camera, Front Camera, AI Lens, Mobile Height, Processor, and the target **Price** (`Prize`).

## Repository structure
```
├── data/
│   ├── Processed_Flipdata.xlsx       # raw dataset
│   └── cleaned_data.csv              # cleaned dataset (generated)
├── figures/                          # 12 charts (EDA, feature extraction, models, importance)
├── models/best_model.joblib          # trained Gradient Boosting model
├── reports/
│   ├── REPORT.md                     # full project report
│   ├── eda_summary.txt               # data exploration output
│   ├── model_results.csv             # model metrics
│   └── feature_importance.csv        # feature importances
├── mobile_price_prediction.py        # complete pipeline (one command)
├── Mobile_Price_Prediction.ipynb     # notebook version with outputs
└── requirements.txt
```

## How to run
```bash
pip install -r requirements.txt
python mobile_price_prediction.py
```
All figures, the report and the model are regenerated. Alternatively, open `Mobile_Price_Prediction.ipynb` in Jupyter or Google Colab and run all cells.

## Tech stack
Python · pandas · NumPy · scikit-learn · matplotlib · seaborn · Jupyter

## Author
Manthan Desai
