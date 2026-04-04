# ⚡ Sales Intelligence Dashboard

A full-stack **Streamlit** analytics dashboard covering **Descriptive → Diagnostic → Prescriptive → Predictive** analysis of regional sales performance data.

![Python](https://img.shields.io/badge/Python-3.9%2B-blue?style=flat-square&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-1.32%2B-FF4B4B?style=flat-square&logo=streamlit)
![Plotly](https://img.shields.io/badge/Plotly-5.20%2B-3F4F75?style=flat-square&logo=plotly)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.4%2B-F7931E?style=flat-square&logo=scikit-learn)

---

## 📸 Dashboard Tabs

| Tab | Analysis Type | What it shows |
|-----|--------------|---------------|
| 📊 Descriptive | What happened? | Revenue trends, regional share, heatmaps, stats table |
| 🔍 Diagnostic | Why did it happen? | Funnel radar, efficiency scatter, root cause insights |
| 💡 Prescriptive | What should we do? | Rep quadrant, action plan, resource allocation |
| 🤖 Predictive | What will happen? | ML forecast, feature importance, what-if simulator |

---

## 🚀 Quick Start

### 1. Clone the repo
```bash
git clone https://github.com/YOUR_USERNAME/sales-dashboard.git
cd sales-dashboard
```

### 2. Create a virtual environment (recommended)
```bash
python -m venv venv

# Windows
venv\Scripts\activate

# macOS / Linux
source venv/bin/activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Add your data file
Place your `sales_performance_data.xlsx` file in the root of the project folder.

> The Excel file must contain a single sheet with comma-separated values in one column,
> with this header structure:
> `Region, AUH_Name, Senior_Manager_Name, Sales_Manager_Name, Sales_Rep_Name,
> Calls_Dialed, Call_Time_Mins, New_Leads, Disqualified, No_Answer, Qualified,
> Converted, Deals_Closed, Followup_Leads, Total_Revenue, Avg_Unit_Value`

### 5. Run the dashboard
```bash
streamlit run dashboard.py
```

The app opens automatically at `http://localhost:8501`

---

## 📁 Project Structure

```
sales-dashboard/
│
├── dashboard.py                  # Main Streamlit app
├── requirements.txt              # Python dependencies
├── README.md                     # This file
├── .gitignore                    # Git ignore rules
│
├── .streamlit/
│   └── config.toml               # Streamlit theme & server config
│
└── sales_performance_data.xlsx   # Your data file (add manually)
```

---

## 🤖 Predictive Analytics — How It Works

The predictive tab trains an ML model on your sales data to forecast revenue and simulate outcomes.

### Features used for prediction:
- `Calls_Dialed` — activity volume
- `Call_Time_Mins` — engagement depth
- `New_Leads`, `Qualified` — pipeline quality
- `Converted`, `Deals_Closed` — closing ability
- `Avg_Unit_Value` — deal size
- `Region` (encoded) — geographic effect
- `Quarter` (1–4) — seasonality/time trend

### Models available (selectable from sidebar):
| Model | Best for | Speed |
|-------|----------|-------|
| **Random Forest** | Robust, handles non-linear patterns | Fast |
| **Gradient Boosting** | Highest accuracy, sequential error correction | Medium |
| **Linear Regression** | Interpretable, baseline comparison | Very Fast |

### Evaluation metrics shown:
- **R² Score** — % of revenue variance explained by the model
- **MAE** — average dollar error on unseen test data

### What-If Simulator:
Adjust any sales activity metric using sliders and instantly see the ML model's predicted revenue — with comparison against the regional average and a gauge chart.

---

## 🛠️ Tech Stack

| Library | Purpose |
|---------|---------|
| `streamlit` | Web app framework |
| `plotly` | Interactive charts |
| `pandas` | Data manipulation |
| `numpy` | Numerical computing |
| `scikit-learn` | ML models (RF, GBM, LR) |
| `openpyxl` | Excel file reading |

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Click **New app** → select your repo → set `dashboard.py` as the main file
4. Upload `sales_performance_data.xlsx` via **Secrets** or add it to the repo
5. Click **Deploy**

> ⚠️ If uploading the data file to GitHub, make sure it doesn't contain sensitive information.

---

## 📊 Regions in the Dataset

| Region | Performance Profile |
|--------|-------------------|
| 🔵 North | High performer — avg ~$130K/rep, growing each quarter |
| 🔴 South | Declining — avg ~$25K/rep, needs intervention |
| 🟢 East | Mid-tier, consistent growth trajectory |
| 🟡 West | Mid-high, strong upward trend |

---

## 📝 License

MIT License — free to use and modify.
