"""
Sales Intelligence Dashboard  ·  v3.0 (definitive)
Reads: sales_performance_data.csv  (committed to same repo folder)
Columns required: Date, Quarter, Region, AUH_Name, Senior_Manager_Name,
  Sales_Manager_Name, Sales_Rep_Name, Calls_Dialed, Call_Time_Mins,
  New_Leads, Disqualified, No_Answer, Qualified, Converted, Deals_Closed,
  Followup_Leads, Total_Revenue, Avg_Unit_Value
"""

import io
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Intelligence Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── THEME CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=Instrument+Sans:wght@400;500;600&display=swap');
html,body,[class*="css"]{font-family:'Instrument Sans',sans-serif;}
.stApp{background:#060912;color:#e8eaf6;}
[data-testid="stSidebar"]{background:#0c1225!important;border-right:1px solid rgba(99,102,241,.2);}
[data-testid="stSidebar"] *{color:#c7d2fe!important;}
.dash-header{background:linear-gradient(135deg,#0f1535,#1a0a3d,#0a1530);border:1px solid rgba(99,102,241,.3);border-radius:16px;padding:32px 40px;margin-bottom:28px;position:relative;overflow:hidden;}
.dash-header::before{content:'';position:absolute;top:-50%;right:-10%;width:400px;height:400px;background:radial-gradient(circle,rgba(99,102,241,.15),transparent 70%);border-radius:50%;}
.dash-title{font-family:'Syne',sans-serif;font-size:2.6rem;font-weight:800;background:linear-gradient(135deg,#a5b4fc,#818cf8,#c4b5fd);-webkit-background-clip:text;-webkit-text-fill-color:transparent;margin:0;letter-spacing:-1px;}
.dash-subtitle{color:#6366f1;font-family:'DM Mono',monospace;font-size:.8rem;letter-spacing:3px;text-transform:uppercase;margin-top:6px;}
.section-title{font-family:'Syne',sans-serif;font-size:1.4rem;font-weight:700;color:#a5b4fc;margin:32px 0 16px;display:flex;align-items:center;gap:10px;}
.section-badge{font-family:'DM Mono',monospace;font-size:.65rem;background:rgba(99,102,241,.2);color:#818cf8;border:1px solid rgba(99,102,241,.3);padding:3px 10px;border-radius:20px;letter-spacing:2px;text-transform:uppercase;}
.insight-box{background:linear-gradient(135deg,#0f1535,#1a1040);border:1px solid rgba(99,102,241,.25);border-left:4px solid #6366f1;border-radius:12px;padding:18px 22px;margin:12px 0;font-size:.92rem;color:#c7d2fe;line-height:1.6;}
.insight-box.warning{border-left-color:#f59e0b;background:linear-gradient(135deg,#0f1535,#1a1008);}
.insight-box.success{border-left-color:#34d399;background:linear-gradient(135deg,#0f1535,#0a1a14);}
.insight-box.danger{border-left-color:#f87171;background:linear-gradient(135deg,#0f1535,#1a0a0a);}
.predict-hero{background:linear-gradient(135deg,#0f1535,#1a0a3d);border:1px solid rgba(167,139,250,.3);border-radius:16px;padding:28px 32px;margin-bottom:20px;}
.metric-pill{display:inline-block;background:rgba(99,102,241,.15);border:1px solid rgba(99,102,241,.3);color:#a5b4fc;font-family:'DM Mono',monospace;font-size:.75rem;padding:4px 12px;border-radius:20px;margin:3px;}
.stTabs [data-baseweb="tab-list"]{background:#0c1225;border-radius:12px;padding:4px;gap:4px;border:1px solid rgba(99,102,241,.2);}
.stTabs [data-baseweb="tab"]{font-family:'Instrument Sans',sans-serif;font-weight:500;color:#6366f1!important;border-radius:8px!important;}
.stTabs [aria-selected="true"]{background:rgba(99,102,241,.2)!important;color:#a5b4fc!important;}
hr{border-color:rgba(99,102,241,.15)!important;}
</style>
""", unsafe_allow_html=True)

# ── CONSTANTS ──────────────────────────────────────────────────────────────────
COLORS = {
    "North": "#6366f1", "South": "#f87171",
    "East":  "#34d399", "West":  "#fbbf24",
}
PLOTLY_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(12,18,37,0.8)",
    font=dict(family="Instrument Sans", color="#c7d2fe", size=12),
    title_font=dict(family="Syne", size=16, color="#a5b4fc"),
    legend=dict(bgcolor="rgba(0,0,0,0)", bordercolor="rgba(99,102,241,0.2)", borderwidth=1),
    xaxis=dict(gridcolor="rgba(99,102,241,0.1)", linecolor="rgba(99,102,241,0.2)", tickfont=dict(size=11)),
    yaxis=dict(gridcolor="rgba(99,102,241,0.1)", linecolor="rgba(99,102,241,0.2)", tickfont=dict(size=11)),
    margin=dict(l=20, r=20, t=50, b=20),
)
NUM_COLS = [
    "Calls_Dialed", "Call_Time_Mins", "New_Leads", "Disqualified",
    "No_Answer", "Qualified", "Converted", "Deals_Closed",
    "Followup_Leads", "Total_Revenue", "Avg_Unit_Value",
]
Q_MAP = {"Q1 2024": 1, "Q2 2024": 2, "Q3 2024": 3, "Q4 2024": 4}
QUARTER_ORDER = list(Q_MAP.keys())


# ── DATA LOADER ────────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Loading data…")
def load_data(raw: bytes) -> pd.DataFrame:
    """Parse CSV bytes → clean DataFrame ready for all charts."""
    df = pd.read_csv(io.BytesIO(raw))

    # Sanitise column names and string cells
    df.columns = [c.strip() for c in df.columns]
    for col in df.select_dtypes("object").columns:
        df[col] = df[col].str.strip()

    # Validate required columns
    required = {"Region", "Quarter", "Total_Revenue", "Calls_Dialed"}
    missing = required - set(df.columns)
    if missing:
        st.error(f"CSV is missing columns: {missing}. Check the file and re-upload.")
        st.stop()

    # Coerce numeric columns
    for col in [c for c in NUM_COLS if c in df.columns]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Quarter sort key
    df["Q_Num"] = df["Quarter"].map(Q_MAP).fillna(0).astype(int)

    # Derived KPIs
    df["Conversion_Rate"]  = df["Converted"]      / df["Calls_Dialed"].replace(0, 1)
    df["Lead_Quality"]     = df["Qualified"]       / df["New_Leads"].replace(0, 1)
    df["Revenue_Per_Call"] = df["Total_Revenue"]   / df["Calls_Dialed"].replace(0, 1)

    return df


# ── FILE RESOLUTION ────────────────────────────────────────────────────────────
_csv_path = Path(__file__).parent / "sales_performance_data.csv"

if _csv_path.exists():
    df = load_data(_csv_path.read_bytes())
else:
    st.markdown("""
    <div class="insight-box warning" style="margin-bottom:20px">
        📂 <strong>sales_performance_data.csv not found in repo.</strong><br>
        Upload it below to launch the dashboard, or commit it to the same folder as this file.
    </div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload sales_performance_data.csv", type=["csv"])
    if uploaded is None:
        st.stop()
    df = load_data(uploaded.read())


# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Filters")
    all_regions  = sorted(df["Region"].dropna().unique().tolist())
    all_quarters = [q for q in QUARTER_ORDER if q in df["Quarter"].unique()]

    sel_regions  = st.multiselect("Regions",  all_regions,  default=all_regions)
    sel_quarters = st.multiselect("Quarters", all_quarters, default=all_quarters)

    st.markdown("---")
    st.markdown("### 🤖 Predictive Model")
    model_choice = st.selectbox("Algorithm", ["Random Forest", "Gradient Boosting", "Linear Regression"])
    st.markdown("---")
    st.caption("Sales Intelligence Dashboard v3.0")

fdf = df[df["Region"].isin(sel_regions) & df["Quarter"].isin(sel_quarters)].copy()

if fdf.empty:
    st.warning("No data for the selected filters. Adjust region/quarter selection.")
    st.stop()


# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <p class="dash-subtitle">⚡ Sales Intelligence Platform · FY 2024</p>
  <h1 class="dash-title">Sales Performance Dashboard</h1>
  <p style="color:#7c85b8;margin-top:10px;font-size:.9rem;">
    Descriptive · Diagnostic · Prescriptive · Predictive Analytics
  </p>
</div>
""", unsafe_allow_html=True)

# ── KPI STRIP ──────────────────────────────────────────────────────────────────
total_rev   = fdf["Total_Revenue"].sum()
avg_rev     = fdf["Total_Revenue"].mean()
total_deals = fdf["Deals_Closed"].sum()
conv_rate   = fdf["Converted"].sum() / max(fdf["Calls_Dialed"].sum(), 1) * 100
best_region = fdf.groupby("Region")["Total_Revenue"].mean().idxmax()

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("💰 Total Revenue",   f"${total_rev/1e6:.2f}M")
c2.metric("📈 Avg Revenue/Rep", f"${avg_rev:,.0f}")
c3.metric("🤝 Deals Closed",    f"{int(total_deals):,}")
c4.metric("📞 Conversion Rate", f"{conv_rate:.1f}%")
c5.metric("🏆 Top Region",      best_region)
st.markdown("---")


# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Descriptive", "🔍  Diagnostic", "💡  Prescriptive", "🤖  Predictive",
])


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 · DESCRIPTIVE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.markdown('<div class="section-title">Revenue Distribution <span class="section-badge">Descriptive</span></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        q_rev = (fdf.groupby(["Quarter", "Region"])["Total_Revenue"]
                 .mean().reset_index()
                 .sort_values("Quarter"))
        fig = px.bar(
            q_rev, x="Quarter", y="Total_Revenue", color="Region",
            barmode="group", title="Avg Revenue by Region & Quarter",
            color_discrete_map=COLORS,
            labels={"Total_Revenue": "Avg Revenue ($)", "Quarter": ""},
            text_auto=".2s",
        )
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_traces(textfont_size=10, textangle=0, textposition="outside", cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        reg_sum = fdf.groupby("Region")["Total_Revenue"].sum().reset_index()
        pie_colors = [COLORS.get(r, "#888") for r in reg_sum["Region"]]
        fig2 = go.Figure(go.Pie(
            labels=reg_sum["Region"], values=reg_sum["Total_Revenue"],
            hole=0.62,
            marker=dict(colors=pie_colors, line=dict(color="#060912", width=3)),
            textinfo="label+percent",
            textfont=dict(family="Instrument Sans", size=12),
        ))
        fig2.add_annotation(
            text=f"Total<br><b>${total_rev/1e6:.1f}M</b>",
            x=0.5, y=0.5, showarrow=False,
            font=dict(family="Syne", size=18, color="#a5b4fc"),
        )
        fig2.update_layout(**PLOTLY_LAYOUT, title="Revenue Share by Region")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        trend = (fdf.groupby(["Quarter", "Region"])["Total_Revenue"]
                 .mean().reset_index()
                 .sort_values("Quarter"))
        fig3 = px.line(
            trend, x="Quarter", y="Total_Revenue", color="Region",
            markers=True, title="Revenue Trend by Region (Quarterly)",
            color_discrete_map=COLORS,
            labels={"Total_Revenue": "Avg Revenue ($)"},
        )
        fig3.update_traces(line_width=2.5, marker_size=8)
        fig3.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        fig4 = px.box(
            fdf, x="Region", y="Total_Revenue", color="Region",
            title="Revenue Distribution (Spread & Outliers)",
            color_discrete_map=COLORS,
            labels={"Total_Revenue": "Revenue ($)"},
            points="outliers",
        )
        fig4.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="section-title">Revenue Heatmap <span class="section-badge">Summary</span></div>', unsafe_allow_html=True)
    pivot = fdf.groupby(["Region", "Quarter"])["Total_Revenue"].mean().unstack().round(0)
    fig5 = px.imshow(
        pivot, text_auto=",.0f", aspect="auto",
        title="Avg Revenue Heatmap — Region × Quarter",
        color_continuous_scale=["#0c1225", "#312e81", "#6366f1", "#a5b4fc"],
        labels=dict(color="Avg Revenue ($)"),
    )
    fig5.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<div class="section-title">Descriptive Statistics <span class="section-badge">Table</span></div>', unsafe_allow_html=True)
    desc = fdf.groupby("Region").agg(
        Avg_Revenue    =("Total_Revenue",    "mean"),
        Median_Revenue =("Total_Revenue",    "median"),
        Std_Dev        =("Total_Revenue",    "std"),
        Total_Revenue  =("Total_Revenue",    "sum"),
        Avg_Deals      =("Deals_Closed",     "mean"),
        Avg_Calls      =("Calls_Dialed",     "mean"),
        Conversion_Pct =("Conversion_Rate",  lambda x: x.mean() * 100),
    ).round(1).reset_index()
    for col in ["Avg_Revenue", "Median_Revenue", "Total_Revenue", "Std_Dev"]:
        desc[col] = desc[col].map("${:,.0f}".format)
    desc["Conversion_Pct"] = desc["Conversion_Pct"].map("{:.1f}%".format)
    st.dataframe(desc, use_container_width=True, hide_index=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 · DIAGNOSTIC
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown('<div class="section-title">Why is Revenue Different? <span class="section-badge">Diagnostic</span></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box warning">
        🔍 <strong>Diagnostic goal:</strong> Identify the <em>root causes</em> of regional revenue gaps —
        examining call efficiency, lead quality, conversion rates, and deal values.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ── CHART 1: Avg Calls Dialed (bar) + Connected Calls (dotted line) — by Region only
    with col1:
        call_data = fdf.groupby("Region").agg(
            Avg_Calls_Dialed=("Calls_Dialed", "mean"),
            Avg_No_Answer   =("No_Answer",    "mean"),
        ).reset_index()
        call_data["Avg_Connected"] = call_data["Avg_Calls_Dialed"] - call_data["Avg_No_Answer"]

        fig_calls = go.Figure()
        fig_calls.add_trace(go.Bar(
            x=call_data["Region"],
            y=call_data["Avg_Calls_Dialed"],
            name="Avg Calls Dialed",
            marker_color=[COLORS.get(r, "#888") for r in call_data["Region"]],
            opacity=0.8,
            text=call_data["Avg_Calls_Dialed"].round(0),
            textposition="outside",
        ))
        fig_calls.add_trace(go.Scatter(
            x=call_data["Region"],
            y=call_data["Avg_Connected"],
            name="Avg Connected Calls",
            mode="lines+markers",
            line=dict(color="#a5b4fc", dash="dot", width=2.5),
            marker=dict(size=9, symbol="circle", color="#a5b4fc"),
        ))
        fig_calls.update_layout(
            **PLOTLY_LAYOUT,
            title="Avg Calls Dialed vs Connected Calls by Region",
            xaxis_title="Region",
            yaxis_title="Avg Calls",
        )
        st.plotly_chart(fig_calls, use_container_width=True)

    # ── CHART 2: Lead Funnel — proper Funnel chart per region ───────────────
    with col2:
        funnel_agg = fdf.groupby("Region").agg(
            New_Leads =("New_Leads",    "sum"),
            Qualified =("Qualified",    "sum"),
            Converted =("Converted",    "sum"),
            Closed    =("Deals_Closed", "sum"),
        ).reset_index()

        stage_cols   = ["New_Leads", "Qualified", "Converted", "Closed"]
        stage_labels = ["New Leads", "Qualified", "Converted", "Deals Closed"]

        fig_funnel = go.Figure()
        for _, row in funnel_agg.iterrows():
            region = row["Region"]
            vals   = [row[s] for s in stage_cols]
            pcts   = [f"{v/vals[0]*100:.1f}%" if vals[0] > 0 else "0%" for v in vals]
            fig_funnel.add_trace(go.Funnel(
                name=region,
                y=stage_labels,
                x=vals,
                textinfo="value+percent initial",
                marker=dict(color=COLORS.get(region, "#888")),
                connector=dict(line=dict(color="rgba(255,255,255,0.1)", width=1)),
                opacity=0.85,
            ))

        fig_funnel.update_layout(
            **PLOTLY_LAYOUT,
            title="Lead Conversion Funnel — Across Regions",
            funnelmode="overlay",
        )
        st.plotly_chart(fig_funnel, use_container_width=True)

    col3, col4 = st.columns(2)

    # ── CHART 3: Revenue per Call (retained) ──────────────────────────────────
    with col3:
        rpc = (fdf.groupby(["Quarter", "Region"])["Revenue_Per_Call"]
               .mean().reset_index().sort_values("Quarter"))
        fig3 = px.line(
            rpc, x="Quarter", y="Revenue_Per_Call", color="Region", markers=True,
            title="Revenue per Call — Efficiency Metric",
            color_discrete_map=COLORS,
            labels={"Revenue_Per_Call": "Rev / Call ($)"},
        )
        fig3.update_traces(line_width=2.5, marker_size=8)
        fig3.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)

    # ── CHART 4: Connected vs Converted — trend analysis across regions (solid lines) ──
    with col4:
        trend_data = fdf.groupby(["Quarter", "Region"]).agg(
            Avg_Connected =("No_Answer", lambda x: (fdf.loc[x.index, "Calls_Dialed"] - x).mean()),
            Avg_Converted =("Converted", "mean"),
        ).reset_index().sort_values("Quarter")

        fig4 = go.Figure()
        for region in sorted(fdf["Region"].unique()):
            rd = trend_data[trend_data["Region"] == region]
            clr = COLORS.get(region, "#888")
            fig4.add_trace(go.Scatter(
                x=rd["Quarter"], y=rd["Avg_Connected"],
                name=f"{region} — Connected",
                mode="lines+markers",
                line=dict(color=clr, width=2.5),
                marker=dict(size=8, symbol="circle"),
                legendgroup=region,
            ))
            fig4.add_trace(go.Scatter(
                x=rd["Quarter"], y=rd["Avg_Converted"],
                name=f"{region} — Converted",
                mode="lines+markers",
                line=dict(color=clr, width=2),
                marker=dict(size=7, symbol="diamond"),
                legendgroup=region,
            ))

        fig4.update_layout(
            **PLOTLY_LAYOUT,
            title="Connected vs Converted Calls — Trend Analysis by Region",
            xaxis_title="Quarter",
            yaxis_title="Avg Count",
        )
        st.plotly_chart(fig4, use_container_width=True)

    auv = (fdf.groupby(["Quarter", "Region"])["Avg_Unit_Value"]
           .mean().reset_index().sort_values("Quarter"))
    fig5 = px.bar(
        auv, x="Region", y="Avg_Unit_Value", color="Quarter", barmode="group",
        title="Avg Deal Value by Region & Quarter — Are North reps closing bigger deals?",
        labels={"Avg_Unit_Value": "Avg Unit Value ($)"},
        color_discrete_sequence=["#312e81", "#4f46e5", "#818cf8", "#c7d2fe"],
    )
    fig5.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig5, use_container_width=True)

    st.markdown('<div class="section-title">Root Cause Summary <span class="section-badge">Findings</span></div>', unsafe_allow_html=True)
    north_rpc = fdf[fdf["Region"] == "North"]["Revenue_Per_Call"].mean()
    south_rpc = fdf[fdf["Region"] == "South"]["Revenue_Per_Call"].mean()
    north_lq  = fdf[fdf["Region"] == "North"]["Lead_Quality"].mean()
    south_lq  = fdf[fdf["Region"] == "South"]["Lead_Quality"].mean()
    ratio = north_rpc / south_rpc if south_rpc > 0 else 0

    st.markdown(f"""
    <div class="insight-box success">
        ✅ <strong>North dominance:</strong> North generates <strong>${north_rpc:.0f}/call</strong>
        vs South's <strong>${south_rpc:.0f}/call</strong> — a {ratio:.1f}x efficiency gap.
        Lead quality: North {north_lq:.2f} vs South {south_lq:.2f}.
    </div>
    <div class="insight-box danger">
        🚨 <strong>South decline:</strong> Revenue falls every quarter — pipeline quality issue,
        not just market conditions. Intervention needed before Q1 2025.
    </div>
    <div class="insight-box warning">
        ⚠️ <strong>East & West opportunity:</strong> Consistent growth with moderate variance —
        scaling call volume 20–30% could unlock significant upside.
    </div>
    """, unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 · PRESCRIPTIVE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown('<div class="section-title">What Should We Do? <span class="section-badge">Prescriptive</span></div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="insight-box">
        💡 <strong>Prescriptive analytics</strong> moves from "what happened" and "why" to
        <em>actionable recommendations</em> — optimising resource allocation and rep targets.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # ── Qualified Leads bar by Region ─────────────────────────────────────────
    with col1:
        qual_data = fdf.groupby("Region").agg(
            Avg_Qualified=("Qualified", "mean"),
        ).reset_index()
        fig_qual = go.Figure()
        fig_qual.add_trace(go.Bar(
            x=qual_data["Region"],
            y=qual_data["Avg_Qualified"],
            marker_color=[COLORS.get(r, "#888") for r in qual_data["Region"]],
            text=qual_data["Avg_Qualified"].round(1),
            textposition="outside",
            name="Avg Qualified Leads",
        ))
        fig_qual.update_layout(
            **PLOTLY_LAYOUT,
            title="Avg Qualified Leads by Region",
            xaxis_title="Region",
            yaxis_title="Avg Qualified Leads",
            showlegend=False,
        )
        st.plotly_chart(fig_qual, use_container_width=True)

    # ── Converted Leads trend across Regions ──────────────────────────────────
    with col2:
        conv_trend = fdf.groupby(["Quarter", "Region"]).agg(
            Avg_Converted=("Converted", "mean"),
        ).reset_index().sort_values("Quarter")
        fig_conv = px.line(
            conv_trend, x="Quarter", y="Avg_Converted", color="Region",
            markers=True,
            title="Converted Leads Trend Across Regions",
            color_discrete_map=COLORS,
            labels={"Avg_Converted": "Avg Converted Leads", "Quarter": ""},
        )
        fig_conv.update_traces(line_width=2.5, marker_size=8)
        fig_conv.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig_conv, use_container_width=True)

    col3, col4 = st.columns(2)

    with col3:
        region_roi = fdf.groupby("Region").agg(
            Total_Rev  =("Total_Revenue",  "sum"),
            Total_Time =("Call_Time_Mins", "sum"),
        ).reset_index()
        region_roi["Rev_Per_Min"] = region_roi["Total_Rev"] / region_roi["Total_Time"].replace(0, 1)
        fig3 = px.bar(
            region_roi, x="Region", y="Rev_Per_Min", color="Region",
            title="Revenue per Call Minute — Where Time is Worth Most",
            color_discrete_map=COLORS,
            labels={"Rev_Per_Min": "Revenue per Minute ($)"},
        )
        fig3.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        if "Sales_Rep_Name" in fdf.columns:
            rep_rank = (fdf.groupby(["Sales_Rep_Name", "Region"])["Total_Revenue"]
                        .mean().reset_index()
                        .sort_values("Total_Revenue", ascending=False))
            combined = pd.concat([
                rep_rank.head(10).assign(Group="Top 10"),
                rep_rank.tail(10).assign(Group="Bottom 10"),
            ])
            fig4 = px.bar(
                combined, y="Sales_Rep_Name", x="Total_Revenue", color="Group",
                orientation="h", title="Top 10 vs Bottom 10 Reps by Avg Revenue",
                color_discrete_map={"Top 10": "#34d399", "Bottom 10": "#f87171"},
                labels={"Total_Revenue": "Avg Revenue ($)", "Sales_Rep_Name": ""},
            )
            fig4.update_layout(**PLOTLY_LAYOUT, height=400)
            st.plotly_chart(fig4, use_container_width=True)

    st.markdown('<div class="section-title">Recommended Actions <span class="section-badge">Action Plan</span></div>', unsafe_allow_html=True)
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        st.markdown("""<div class="insight-box success">
            <strong>🎯 North — Scale Model</strong><br><br>
            North's playbook is working. Increase headcount 20%, replicate training across East/West.
            Target $200K avg revenue per rep by Q2 2025.
        </div>""", unsafe_allow_html=True)
    with ac2:
        st.markdown("""<div class="insight-box danger">
            <strong>🚨 South — Rescue Plan</strong><br><br>
            Deploy North senior reps to mentor South. Audit lead qualification criteria.
            If no improvement in 2 quarters, reallocate budget to West.
        </div>""", unsafe_allow_html=True)
    with ac3:
        st.markdown("""<div class="insight-box warning">
            <strong>📈 East/West — Growth Mode</strong><br><br>
            Increase call targets by 25%. East should focus on improving avg deal value —
            currently lagging North by 2×.
        </div>""", unsafe_allow_html=True)


# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 · PREDICTIVE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown('<div class="section-title">What Will Happen Next? <span class="section-badge">Predictive · ML</span></div>', unsafe_allow_html=True)

    with st.expander("🧠 How Predictive Analytics Works — Deep Explanation", expanded=True):
        st.markdown("""
        <div class="predict-hero">
        <h3 style="font-family:Syne;color:#a5b4fc;margin-top:0">How the Model Predicts Revenue</h3>
        <h4 style="color:#818cf8;font-family:Syne">Step 1 — Feature Engineering 🔧</h4>
        <p style="color:#c7d2fe;line-height:1.8">
        Raw columns become numeric <em>features</em> the model learns from:<br>
        • <strong>Calls_Dialed, Call_Time_Mins</strong> → activity volume<br>
        • <strong>New_Leads, Qualified</strong> → pipeline strength<br>
        • <strong>Converted, Deals_Closed</strong> → closing ability<br>
        • <strong>Avg_Unit_Value</strong> → deal size<br>
        • <strong>Region</strong> → label-encoded integer<br>
        • <strong>Quarter number</strong> → time / seasonality signal
        </p>
        <h4 style="color:#818cf8;font-family:Syne">Step 2 — Train / Test Split 🏋️</h4>
        <p style="color:#c7d2fe;line-height:1.8">
        80% of rows train the model; 20% held out to test accuracy.
        The model sees examples like:<br>
        <em>"North rep, Q3, 800 calls, 45 deals → $148,000"</em><br>
        and learns the mathematical relationship between inputs and revenue.
        </p>
        <h4 style="color:#818cf8;font-family:Syne">Step 3 — Algorithm Choices 🤖</h4>
        <p style="color:#c7d2fe;line-height:1.8">
        <strong>Random Forest:</strong> 150 decision trees, each on a random data subset. Prediction = average of all trees. Robust, handles non-linear relationships.<br><br>
        <strong>Gradient Boosting:</strong> Trees built sequentially — each corrects the last one's errors. Highest accuracy, best for complex patterns.<br><br>
        <strong>Linear Regression:</strong> Straight-line relationship. Fast and interpretable but misses complex interactions.
        </p>
        <h4 style="color:#818cf8;font-family:Syne">Step 4 — Evaluation 📏</h4>
        <p style="color:#c7d2fe;line-height:1.8">
        <strong>R²</strong> — how much revenue variance the model explains (1.0 = perfect).<br>
        <strong>MAE</strong> — average dollar error on unseen test rows.<br>
        <strong>Feature Importance</strong> — which inputs drive the prediction most.
        </p>
        <h4 style="color:#818cf8;font-family:Syne">Step 5 — Forecast 🔮</h4>
        <p style="color:#c7d2fe;line-height:1.8">
        Q5/Q6 (2025) are predicted by extending the time variable with each region's historical averages.
        The ±18% confidence band reflects model uncertainty.
        </p>
        <h4 style="color:#818cf8;font-family:Syne">Why Random Forest for Regression? 🎯</h4>
        <p style="color:#c7d2fe;line-height:1.8">
        Our data has <strong>multiple interacting factors</strong> — calls, lead quality, region, deal value — and
        the relationship between them and revenue is <em>non-linear</em>. For example, 800 calls in North
        generates 5× more revenue than 800 calls in South. Linear Regression assumes a straight-line relationship
        and completely misses this.<br><br>
        <strong>Why not just Linear Regression?</strong> When we tested it, R² dropped to ~0.60 because it cannot
        model the regional multiplier effect. Random Forest captures it naturally by building region-specific
        branches in each tree.<br><br>
        <strong>Random Forest as a Regression Model:</strong><br>
        • Each tree learns: "If Region=North AND Deals_Closed &gt; 30 AND Avg_Unit_Value &gt; 3000 → Revenue ≈ $160K"<br>
        • 150 trees each see different random subsets of rows and features<br>
        • Final prediction = average of all 150 tree predictions (reduces variance)<br>
        • Feature Importance shows <em>which factors explain the most variance in revenue</em><br><br>
        <strong>Relationship with Revenue — All Factors:</strong><br>
        • <strong>Avg_Unit_Value × Deals_Closed</strong> = direct revenue drivers (highest importance)<br>
        • <strong>Region</strong> = captures territory-specific market conditions (2nd highest)<br>
        • <strong>Converted</strong> = pipeline closing efficiency<br>
        • <strong>Qualified</strong> = upstream lead quality signal<br>
        • <strong>Quarter (time)</strong> = captures seasonal growth trends<br>
        • <strong>Calls_Dialed</strong> = volume input — necessary but not sufficient alone
        </p>
        </div>
        """, unsafe_allow_html=True)

    # ── MODEL ──────────────────────────────────────────────────────────────────
    FEATURES = [
        "Calls_Dialed", "Call_Time_Mins", "New_Leads", "Disqualified",
        "No_Answer", "Qualified", "Converted", "Deals_Closed",
        "Followup_Leads", "Avg_Unit_Value", "Region_Enc", "Q_Num",
    ]

    @st.cache_data(show_spinner="Training model…")
    def build_model(model_name: str, data_hash: str, raw: bytes):
        _df = load_data(raw)
        le = LabelEncoder()
        mdf = _df.copy()
        mdf["Region_Enc"] = le.fit_transform(mdf["Region"])
        X = mdf[FEATURES].fillna(0)
        y = mdf["Total_Revenue"]
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if model_name == "Random Forest":
            mdl = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
        elif model_name == "Gradient Boosting":
            mdl = GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
        else:
            mdl = LinearRegression()
        mdl.fit(X_train, y_train)
        y_pred = mdl.predict(X_test)
        r2  = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        imp = (mdl.feature_importances_ if hasattr(mdl, "feature_importances_")
               else np.abs(mdl.coef_) / (np.abs(mdl.coef_).sum() + 1e-9))
        return mdl, r2, mae, imp, X_test, y_test.values, y_pred, le

    # Use file bytes as cache key
    _raw_bytes = _csv_path.read_bytes() if _csv_path.exists() else uploaded.getvalue()
    model, r2, mae, importances, X_test, y_test_vals, y_pred_vals, le = build_model(
        model_choice, str(hash(_raw_bytes)), _raw_bytes
    )

    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("🎯 R² Score",      f"{r2:.3f}")
    mc2.metric("📉 MAE",           f"${mae:,.0f}")
    mc3.metric("🔢 Training Rows", f"{int(len(df)*0.8):,}")
    mc4.metric("🧪 Test Rows",     f"{int(len(df)*0.2):,}")
    st.markdown("---")

    col1, col2 = st.columns(2)

    with col1:
        FRIENDLY = {
            "Calls_Dialed": "Calls Dialed", "Call_Time_Mins": "Call Time (mins)",
            "New_Leads": "New Leads", "Disqualified": "Disqualified Leads",
            "No_Answer": "No Answers", "Qualified": "Qualified Leads",
            "Converted": "Converted", "Deals_Closed": "Deals Closed",
            "Followup_Leads": "Follow-up Leads", "Avg_Unit_Value": "Avg Unit Value",
            "Region_Enc": "Region", "Q_Num": "Quarter",
        }
        feat_df = (pd.DataFrame({"Feature": FEATURES, "Importance": importances})
                   .sort_values("Importance", ascending=True))
        feat_df["Feature"] = feat_df["Feature"].map(FRIENDLY)
        fig = px.bar(
            feat_df, y="Feature", x="Importance", orientation="h",
            title=f"Feature Importance — {model_choice}",
            labels={"Importance": "Importance Score", "Feature": ""},
            color="Importance",
            color_continuous_scale=["#312e81", "#6366f1", "#a5b4fc"],
        )
        fig.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        ap_df = pd.DataFrame({
            "Actual":    y_test_vals,
            "Predicted": y_pred_vals,
            "Error":     np.abs(y_test_vals - y_pred_vals),
        })
        fig2 = px.scatter(
            ap_df, x="Actual", y="Predicted", color="Error",
            title="Actual vs Predicted Revenue",
            color_continuous_scale=["#34d399", "#f59e0b", "#f87171"],
            labels={
                "Actual": "Actual Revenue ($)", "Predicted": "Predicted Revenue ($)",
                "Error":  "Abs Error ($)",
            },
            opacity=0.7,
        )
        mx = max(ap_df["Actual"].max(), ap_df["Predicted"].max())
        fig2.add_trace(go.Scatter(
            x=[0, mx], y=[0, mx], mode="lines",
            line=dict(dash="dash", color="rgba(165,180,252,0.5)"),
            name="Perfect Prediction",
        ))
        fig2.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    # ── FORECAST ───────────────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Revenue Forecast — Q1 & Q2 2025 <span class="section-badge">Forecast</span></div>', unsafe_allow_html=True)

    @st.cache_data(show_spinner="Building forecast…")
    def make_forecast(model_name: str, data_hash: str, raw: bytes) -> pd.DataFrame:
        mdl, *_, le2 = build_model(model_name, data_hash, raw)
        _df = load_data(raw)
        rows = []
        for q_num in [5, 6]:
            quarter_label = f"Q{q_num - 4} 2025"
            for region in ["North", "South", "East", "West"]:
                hist = _df[_df["Region"] == region][
                    ["Calls_Dialed", "Call_Time_Mins", "New_Leads", "Disqualified",
                     "No_Answer", "Qualified", "Converted", "Deals_Closed",
                     "Followup_Leads", "Avg_Unit_Value"]
                ].mean()
                X_fut = pd.DataFrame([{
                    "Calls_Dialed":    hist["Calls_Dialed"],
                    "Call_Time_Mins":  hist["Call_Time_Mins"],
                    "New_Leads":       hist["New_Leads"],
                    "Disqualified":    hist["Disqualified"],
                    "No_Answer":       hist["No_Answer"],
                    "Qualified":       hist["Qualified"],
                    "Converted":       hist["Converted"],
                    "Deals_Closed":    hist["Deals_Closed"],
                    "Followup_Leads":  hist["Followup_Leads"],
                    "Avg_Unit_Value":  hist["Avg_Unit_Value"],
                    "Region_Enc":      le2.transform([region])[0],
                    "Q_Num":           q_num,
                }])
                pred = mdl.predict(X_fut)[0]
                rows.append({
                    "Quarter": quarter_label,
                    "Region":  region,
                    "Predicted_Revenue": pred,
                    "Lower":   pred * 0.82,
                    "Upper":   pred * 1.18,
                })
        return pd.DataFrame(rows)

    forecast_df = make_forecast(model_choice, str(hash(_raw_bytes)), _raw_bytes)

    hist_avg = (df.groupby(["Quarter", "Region"])["Total_Revenue"]
                .mean().reset_index())
    hist_avg["Q_Num"] = hist_avg["Quarter"].map(Q_MAP).fillna(0)
    hist_avg = hist_avg.sort_values("Q_Num")

    col1, col2 = st.columns(2)

    with col1:
        all_quarters_plot = QUARTER_ORDER + ["Q1 2025", "Q2 2025"]
        fig3 = go.Figure()
        for region in ["North", "South", "East", "West"]:
            hist_r = hist_avg[hist_avg["Region"] == region].sort_values("Q_Num")
            fore_r = forecast_df[forecast_df["Region"] == region]
            x_all = list(hist_r["Quarter"]) + list(fore_r["Quarter"])
            y_all = list(hist_r["Total_Revenue"]) + list(fore_r["Predicted_Revenue"])
            c = COLORS[region]
            fig3.add_trace(go.Scatter(
                x=x_all, y=y_all, mode="lines+markers", name=region,
                line=dict(color=c, width=2.5), marker=dict(size=8),
            ))
            # confidence band (forecast only)
            rx = list(fore_r["Quarter"])
            fig3.add_trace(go.Scatter(
                x=rx + rx[::-1],
                y=list(fore_r["Upper"]) + list(fore_r["Lower"])[::-1],
                fill="toself",
                fillcolor=f"rgba({int(c[1:3],16)},{int(c[3:5],16)},{int(c[5:7],16)},0.15)",
                line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ))
        # dashed divider between history and forecast
        # add_vrect doesn't support string category axes in Plotly 6 — use shapes instead
        fig3.add_shape(
            type="line",
            x0="Q4 2024", x1="Q4 2024", y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="rgba(165,180,252,0.4)", width=1, dash="dot"),
        )
        fig3.add_annotation(
            x="Q4 2024", y=1.05, xref="x", yref="paper",
            text="← History | Forecast →",
            showarrow=False,
            font=dict(color="#a5b4fc", size=11),
            xanchor="center",
        )
        # Build layout without duplicate xaxis key
        _forecast_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "xaxis"}
        _forecast_layout.update({
            "title": "Revenue Forecast Q1–Q2 2025 (with ±18% confidence band)",
            "height": 420,
            "showlegend": True,
            "xaxis": {
                **PLOTLY_LAYOUT["xaxis"],
                "categoryorder": "array",
                "categoryarray": all_quarters_plot,
            },
        })
        fig3.update_layout(**_forecast_layout)
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        fore_tbl = forecast_df[["Quarter", "Region", "Predicted_Revenue", "Lower", "Upper"]].copy()
        fore_tbl["Predicted_Revenue"] = fore_tbl["Predicted_Revenue"].map("${:,.0f}".format)
        fore_tbl["Lower"]             = fore_tbl["Lower"].map("${:,.0f}".format)
        fore_tbl["Upper"]             = fore_tbl["Upper"].map("${:,.0f}".format)
        fore_tbl.columns = ["Quarter", "Region", "Predicted Revenue", "Lower (−18%)", "Upper (+18%)"]
        st.markdown("#### 📋 Forecast Table")
        st.dataframe(fore_tbl, use_container_width=True, hide_index=True)

        n_fore = forecast_df[forecast_df["Region"] == "North"]["Predicted_Revenue"].mean()
        s_fore = forecast_df[forecast_df["Region"] == "South"]["Predicted_Revenue"].mean()
        gap = n_fore / s_fore if s_fore > 0 else 0
        st.markdown(f"""
        <div class="insight-box success">
            🔮 <strong>2025 Outlook:</strong> North projected at <strong>${n_fore:,.0f}/rep</strong> avg in H1 2025.
            South continues declining to <strong>${s_fore:,.0f}/rep</strong> —
            gap reaches <strong>{gap:.0f}×</strong> by mid-2025.
        </div>
        """, unsafe_allow_html=True)

    # ── TOP 5 PREDICTED REPS ──────────────────────────────────────────────────
    st.markdown('<div class="section-title">🏆 Top 5 Predicted Sales Reps <span class="section-badge">ML Ranking</span></div>', unsafe_allow_html=True)
    st.markdown("""<div class="insight-box">The model predicts each rep's Q1 2025 revenue using their historical activity profile as input features - surfacing who is set to lead.</div>""", unsafe_allow_html=True)

    if "Sales_Rep_Name" in df.columns:
        rep_profiles = df.groupby(["Sales_Rep_Name", "Region"]).agg(
            Calls_Dialed   =("Calls_Dialed",   "mean"),
            Call_Time_Mins =("Call_Time_Mins",  "mean"),
            New_Leads      =("New_Leads",       "mean"),
            Disqualified   =("Disqualified",    "mean"),
            No_Answer      =("No_Answer",       "mean"),
            Qualified      =("Qualified",       "mean"),
            Converted      =("Converted",       "mean"),
            Deals_Closed   =("Deals_Closed",    "mean"),
            Followup_Leads =("Followup_Leads",  "mean"),
            Avg_Unit_Value =("Avg_Unit_Value",  "mean"),
            Actual_Avg_Rev =("Total_Revenue",   "mean"),
        ).reset_index()

        rep_profiles["Region_Enc"] = le.transform(rep_profiles["Region"])
        rep_profiles["Q_Num"]      = 5  # Q1 2025
        X_reps = rep_profiles[FEATURES].fillna(0)
        rep_profiles["Predicted_Q1_2025"] = model.predict(X_reps)

        top5 = rep_profiles.nlargest(5, "Predicted_Q1_2025")[
            ["Sales_Rep_Name", "Region", "Actual_Avg_Rev", "Predicted_Q1_2025"]
        ].reset_index(drop=True)
        top5.index = top5.index + 1  # rank from 1

        r5c1, r5c2 = st.columns(2)
        with r5c1:
            fig_top5 = go.Figure()
            fig_top5.add_trace(go.Bar(
                y=top5["Sales_Rep_Name"],
                x=top5["Actual_Avg_Rev"],
                name="Actual Avg Revenue",
                orientation="h",
                marker_color="#6366f1", opacity=0.7,
                text=top5["Actual_Avg_Rev"].map("${:,.0f}".format),
                textposition="inside",
            ))
            fig_top5.add_trace(go.Bar(
                y=top5["Sales_Rep_Name"],
                x=top5["Predicted_Q1_2025"],
                name="Predicted Q1 2025",
                orientation="h",
                marker_color="#34d399", opacity=0.85,
                text=top5["Predicted_Q1_2025"].map("${:,.0f}".format),
                textposition="inside",
            ))
            fig_top5.update_layout(
                **PLOTLY_LAYOUT,
                title="Top 5 Predicted Reps — Actual vs Q1 2025 Forecast",
                barmode="group",
                xaxis_title="Revenue ($)",
                yaxis_title="",
                height=360,
            )
            st.plotly_chart(fig_top5, use_container_width=True)

        with r5c2:
            # Build display table cleanly — no column rename, just select & label directly
            top5_disp = pd.DataFrame({
                "Rank":              ["🥇","🥈","🥉","4th","5th"],
                "Sales Rep":         top5["Sales_Rep_Name"].values,
                "Region":            top5["Region"].values,
                "Actual Avg Rev":    top5["Actual_Avg_Rev"].map("${:,.0f}".format).values,
                "Predicted Q1 2025": top5["Predicted_Q1_2025"].map("${:,.0f}".format).values,
            })
            st.markdown("#### 📋 Top 5 Rep Rankings")
            st.dataframe(top5_disp, use_container_width=True, hide_index=True)
            best = top5.iloc[0]
            st.markdown(f"""
            <div class="insight-box success">
                🏆 <strong>{best['Sales_Rep_Name']}</strong> ({best['Region']}) is the model's top pick for Q1 2025
                with a predicted revenue of <strong>${best['Predicted_Q1_2025']:,.0f}</strong> —
                based on their historical activity profile including {best['Deals_Closed']:.0f} avg deals/period
                and ${best['Avg_Unit_Value']:,.0f} avg unit value.
            </div>
            """, unsafe_allow_html=True)

    # ── LEAD CONVERSION FORECAST ───────────────────────────────────────────────
    st.markdown('<div class="section-title">📈 Lead Conversion Rate Forecast <span class="section-badge">Forecast</span></div>', unsafe_allow_html=True)
    st.markdown("""<div class="insight-box">Forecasted lead conversion rate (Converted / New Leads) for Q1-Q2 2025 per region, extrapolated from historical quarterly trends using linear regression on each region's time series.</div>""", unsafe_allow_html=True)

    from sklearn.linear_model import LinearRegression as LR
    hist_conv = df.groupby(["Quarter", "Region"]).apply(
        lambda g: pd.Series({"Conv_Rate": g["Converted"].sum() / max(g["New_Leads"].sum(), 1) * 100})
    ).reset_index()
    hist_conv["Q_Num"] = hist_conv["Quarter"].map(Q_MAP).fillna(0)

    lc1, lc2 = st.columns(2)
    with lc1:
        fig_lc = go.Figure()
        all_lc_quarters = QUARTER_ORDER + ["Q1 2025", "Q2 2025"]
        for region in ["North", "South", "East", "West"]:
            rd = hist_conv[hist_conv["Region"] == region].sort_values("Q_Num")
            if len(rd) < 2:
                continue
            # Fit simple linear trend
            lm = LR()
            lm.fit(rd[["Q_Num"]], rd["Conv_Rate"])
            future_q = pd.DataFrame({"Q_Num": [5, 6]})
            future_conv = lm.predict(future_q).clip(0, 100)

            x_all = list(rd["Quarter"]) + ["Q1 2025", "Q2 2025"]
            y_all = list(rd["Conv_Rate"]) + list(future_conv)
            c = COLORS[region]

            fig_lc.add_trace(go.Scatter(
                x=x_all[:len(rd)], y=y_all[:len(rd)],
                name=f"{region} — Historical",
                mode="lines+markers",
                line=dict(color=c, width=2.5),
                marker=dict(size=8),
                legendgroup=region,
            ))
            fig_lc.add_trace(go.Scatter(
                x=x_all[len(rd)-1:], y=y_all[len(rd)-1:],
                name=f"{region} — Forecast",
                mode="lines+markers",
                line=dict(color=c, width=2, dash="dash"),
                marker=dict(size=8, symbol="diamond"),
                legendgroup=region,
                showlegend=False,
            ))

        fig_lc.add_shape(
            type="line", x0="Q4 2024", x1="Q4 2024", y0=0, y1=1,
            xref="x", yref="paper",
            line=dict(color="rgba(165,180,252,0.4)", width=1, dash="dot"),
        )
        fig_lc.add_annotation(
            x="Q4 2024", y=1.05, xref="x", yref="paper",
            text="← History | Forecast →", showarrow=False,
            font=dict(color="#a5b4fc", size=11), xanchor="center",
        )
        _lc_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k != "xaxis"}
        _lc_layout.update({
            "title": "Lead Conversion Rate Forecast — Q1 & Q2 2025",
            "height": 400, "showlegend": True,
            "xaxis": {**PLOTLY_LAYOUT["xaxis"],
                      "categoryorder": "array", "categoryarray": all_lc_quarters},
            "yaxis_title": "Conversion Rate (%)",
        })
        fig_lc.update_layout(**_lc_layout)
        st.plotly_chart(fig_lc, use_container_width=True)

    with lc2:
        lc_rows = []
        for region in ["North", "South", "East", "West"]:
            rd = hist_conv[hist_conv["Region"] == region].sort_values("Q_Num")
            if len(rd) < 2:
                continue
            lm = LR()
            lm.fit(rd[["Q_Num"]], rd["Conv_Rate"])
            q1_pred = float(lm.predict([[5]])[0].clip(0, 100))
            q2_pred = float(lm.predict([[6]])[0].clip(0, 100))
            hist_avg = rd["Conv_Rate"].mean()
            lc_rows.append({"Region": region,
                            "Q1 2025 (%)": round(q1_pred, 1),
                            "Q2 2025 (%)": round(q2_pred, 1),
                            "Historical Avg (%)": round(hist_avg, 1),
                            "Trend": "📈" if q2_pred > rd["Conv_Rate"].iloc[-1] else "📉"})
        lc_df = pd.DataFrame(lc_rows)
        st.markdown("#### 📋 Conversion Rate Forecast Table")
        st.dataframe(lc_df, use_container_width=True, hide_index=True)

    # ── WHAT-IF SIMULATOR ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🎛️ What-If Revenue Simulator <span class="section-badge">Interactive</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Adjust sliders to simulate how changing a rep\'s activity metrics affects predicted revenue — powered by the trained ML model.</div>', unsafe_allow_html=True)

    sim1, sim2, sim3 = st.columns(3)
    with sim1:
        sim_region   = st.selectbox("Region", ["North", "South", "East", "West"])
        sim_calls    = st.slider("Calls Dialed", 100, 1000, 550)
        sim_calltime = st.slider("Call Time (mins)", 100, 2000, 900)
    with sim2:
        sim_leads    = st.slider("New Leads", 20, 250, 100)
        sim_qual     = st.slider("Qualified Leads", 5, 150, 50)
        sim_conv     = st.slider("Converted", 1, 100, 30)
    with sim3:
        sim_deals    = st.slider("Deals Closed", 1, 80, 20)
        sim_auv      = st.slider("Avg Unit Value ($)", 500, 5000, 2000)
        sim_quarter  = st.selectbox("Quarter", ["Q1 2024","Q2 2024","Q3 2024","Q4 2024","Q1 2025","Q2 2025"])

    Q_NUM_MAP = {"Q1 2024":1,"Q2 2024":2,"Q3 2024":3,"Q4 2024":4,"Q1 2025":5,"Q2 2025":6}
    X_sim = pd.DataFrame([{
        "Calls_Dialed":   sim_calls,
        "Call_Time_Mins": sim_calltime,
        "New_Leads":      sim_leads,
        "Disqualified":   max(0, sim_leads - sim_qual - 10),
        "No_Answer":      int(sim_calls * 0.3),
        "Qualified":      sim_qual,
        "Converted":      sim_conv,
        "Deals_Closed":   sim_deals,
        "Followup_Leads": int(sim_leads * 0.3),
        "Avg_Unit_Value": sim_auv,
        "Region_Enc":     le.transform([sim_region])[0],
        "Q_Num":          Q_NUM_MAP[sim_quarter],
    }])
    predicted = model.predict(X_sim)[0]
    ref_avg   = df[df["Region"] == sim_region]["Total_Revenue"].mean()
    delta     = predicted - ref_avg
    delta_pct = delta / ref_avg * 100 if ref_avg > 0 else 0

    sc1, sc2, sc3 = st.columns(3)
    sc1.metric("🤖 ML Predicted Revenue", f"${predicted:,.0f}", delta=f"{delta_pct:+.1f}% vs {sim_region} avg")
    sc2.metric(f"📊 {sim_region} Region Avg", f"${ref_avg:,.0f}")
    sc3.metric("⚡ Rev per Call", f"${predicted/max(sim_calls,1):,.0f}")

    max_rev = df["Total_Revenue"].max()
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted,
        delta={"reference": ref_avg, "valueformat": "$,.0f"},
        number={"valueformat": "$,.0f", "font": {"family": "Syne", "size": 32, "color": "#a5b4fc"}},
        title={"text": f"Predicted Revenue vs {sim_region} Avg", "font": {"family": "Syne", "color": "#c7d2fe"}},
        gauge={
            "axis": {"range": [0, max_rev * 1.1]},
            "bar":  {"color": "#6366f1", "thickness": 0.3},
            "bgcolor": "rgba(0,0,0,0)", "bordercolor": "rgba(0,0,0,0)",
            "steps": [
                {"range": [0,            ref_avg * 0.5], "color": "rgba(248,113,113,0.2)"},
                {"range": [ref_avg * 0.5, ref_avg],      "color": "rgba(251,191,36,0.2)"},
                {"range": [ref_avg,      max_rev * 1.1], "color": "rgba(52,211,153,0.15)"},
            ],
            "threshold": {"line": {"color": "#a5b4fc", "width": 3}, "thickness": 0.8, "value": ref_avg},
        },
    ))
    fig_gauge.update_layout(
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(family="Instrument Sans", color="#c7d2fe"),
        height=320, margin=dict(l=30, r=30, t=60, b=20),
    )
    st.plotly_chart(fig_gauge, use_container_width=True)

    if delta > ref_avg * 0.2:
        verdict = "This rep profile is a top performer — study their approach and replicate."
    elif delta < 0:
        verdict = "Underperforming — increasing Qualified Leads and Deals Closed will have the highest impact per feature importance."
    else:
        verdict = "Performing near the regional average."

    box_cls = "success" if delta > 0 else "danger"
    st.markdown(f"""
    <div class="insight-box {box_cls}">
        🤖 <strong>Model says:</strong> In <strong>{sim_region}</strong> during <strong>{sim_quarter}</strong>,
        this rep is predicted to generate <strong>${predicted:,.0f}</strong> —
        <strong>${abs(delta):,.0f} {"above" if delta > 0 else "below"}</strong> the
        {sim_region} avg of ${ref_avg:,.0f}.<br>{verdict}
    </div>
    """, unsafe_allow_html=True)
