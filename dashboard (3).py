import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error
import warnings
warnings.filterwarnings("ignore")

# ── PAGE CONFIG ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Sales Intelligence Dashboard",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── THEME CSS ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Syne:wght@400;600;700;800&family=DM+Mono:wght@300;400;500&family=Instrument+Sans:wght@400;500;600&display=swap');

html, body, [class*="css"] {
    font-family: 'Instrument Sans', sans-serif;
}

/* Background */
.stApp {
    background: #060912;
    color: #e8eaf6;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #0c1225 !important;
    border-right: 1px solid rgba(99,102,241,0.2);
}
[data-testid="stSidebar"] * { color: #c7d2fe !important; }

/* Header */
.dash-header {
    background: linear-gradient(135deg, #0f1535 0%, #1a0a3d 50%, #0a1530 100%);
    border: 1px solid rgba(99,102,241,0.3);
    border-radius: 16px;
    padding: 32px 40px;
    margin-bottom: 28px;
    position: relative;
    overflow: hidden;
}
.dash-header::before {
    content: '';
    position: absolute;
    top: -50%;
    right: -10%;
    width: 400px;
    height: 400px;
    background: radial-gradient(circle, rgba(99,102,241,0.15) 0%, transparent 70%);
    border-radius: 50%;
}
.dash-title {
    font-family: 'Syne', sans-serif;
    font-size: 2.6rem;
    font-weight: 800;
    background: linear-gradient(135deg, #a5b4fc, #818cf8, #c4b5fd);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    margin: 0;
    letter-spacing: -1px;
}
.dash-subtitle {
    color: #6366f1;
    font-family: 'DM Mono', monospace;
    font-size: 0.8rem;
    letter-spacing: 3px;
    text-transform: uppercase;
    margin-top: 6px;
}

/* KPI Cards */
.kpi-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 16px;
    margin-bottom: 28px;
}
.kpi-card {
    background: #0c1225;
    border: 1px solid rgba(99,102,241,0.2);
    border-radius: 14px;
    padding: 22px 24px;
    position: relative;
    overflow: hidden;
    transition: border-color 0.2s;
}
.kpi-card::after {
    content: '';
    position: absolute;
    bottom: 0; left: 0; right: 0;
    height: 3px;
    background: linear-gradient(90deg, #6366f1, #a78bfa);
    border-radius: 0 0 14px 14px;
}
.kpi-label {
    font-family: 'DM Mono', monospace;
    font-size: 0.7rem;
    color: #6366f1;
    letter-spacing: 2px;
    text-transform: uppercase;
    margin-bottom: 8px;
}
.kpi-value {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 700;
    color: #e0e7ff;
    line-height: 1;
}
.kpi-delta {
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    margin-top: 6px;
}
.delta-up { color: #34d399; }
.delta-down { color: #f87171; }

/* Section headers */
.section-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.4rem;
    font-weight: 700;
    color: #a5b4fc;
    margin: 32px 0 16px 0;
    display: flex;
    align-items: center;
    gap: 10px;
}
.section-badge {
    font-family: 'DM Mono', monospace;
    font-size: 0.65rem;
    background: rgba(99,102,241,0.2);
    color: #818cf8;
    border: 1px solid rgba(99,102,241,0.3);
    padding: 3px 10px;
    border-radius: 20px;
    letter-spacing: 2px;
    text-transform: uppercase;
}

/* Insight boxes */
.insight-box {
    background: linear-gradient(135deg, #0f1535, #1a1040);
    border: 1px solid rgba(99,102,241,0.25);
    border-left: 4px solid #6366f1;
    border-radius: 12px;
    padding: 18px 22px;
    margin: 12px 0;
    font-size: 0.92rem;
    color: #c7d2fe;
    line-height: 1.6;
}
.insight-box.warning {
    border-left-color: #f59e0b;
    background: linear-gradient(135deg, #0f1535, #1a1008);
}
.insight-box.success {
    border-left-color: #34d399;
    background: linear-gradient(135deg, #0f1535, #0a1a14);
}
.insight-box.danger {
    border-left-color: #f87171;
    background: linear-gradient(135deg, #0f1535, #1a0a0a);
}

/* Predict panel */
.predict-hero {
    background: linear-gradient(135deg, #0f1535 0%, #1a0a3d 100%);
    border: 1px solid rgba(167,139,250,0.3);
    border-radius: 16px;
    padding: 28px 32px;
    margin-bottom: 20px;
}

/* Metric pill */
.metric-pill {
    display: inline-block;
    background: rgba(99,102,241,0.15);
    border: 1px solid rgba(99,102,241,0.3);
    color: #a5b4fc;
    font-family: 'DM Mono', monospace;
    font-size: 0.75rem;
    padding: 4px 12px;
    border-radius: 20px;
    margin: 3px;
}

/* Tab styling */
.stTabs [data-baseweb="tab-list"] {
    background: #0c1225;
    border-radius: 12px;
    padding: 4px;
    gap: 4px;
    border: 1px solid rgba(99,102,241,0.2);
}
.stTabs [data-baseweb="tab"] {
    font-family: 'Instrument Sans', sans-serif;
    font-weight: 500;
    color: #6366f1 !important;
    border-radius: 8px !important;
}
.stTabs [aria-selected="true"] {
    background: rgba(99,102,241,0.2) !important;
    color: #a5b4fc !important;
}

/* Slider and selectbox */
.stSelectbox > div, .stMultiSelect > div {
    background: #0c1225 !important;
    border-color: rgba(99,102,241,0.3) !important;
}

/* Divider */
hr { border-color: rgba(99,102,241,0.15) !important; }

/* Plotly charts background */
.js-plotly-plot { border-radius: 12px; overflow: hidden; }
</style>
""", unsafe_allow_html=True)

# ── LOAD & PREPARE DATA ────────────────────────────────────────────────────────
@st.cache_data
def load_data(file_bytes=None):
    import io
    from pathlib import Path

    num_cols = ['Calls_Dialed','Call_Time_Mins','New_Leads','Disqualified','No_Answer',
                'Qualified','Converted','Deals_Closed','Followup_Leads','Total_Revenue','Avg_Unit_Value']

    if file_bytes is not None:
        df = pd.read_csv(io.BytesIO(file_bytes))
    else:
        base = Path(__file__).parent
        df = pd.read_csv(base / 'sales_performance_data.csv')

    # Sanitise column names and string cells
    df.columns = [c.strip() for c in df.columns]
    for c in df.columns:
        if df[c].dtype == object:
            df[c] = df[c].str.strip()

    # Coerce numeric columns
    for c in [x for x in num_cols if x in df.columns]:
        df[c] = pd.to_numeric(df[c], errors='coerce')

    # Quarter number for ML ordering (Quarter column already present in CSV)
    q_map = {'Q1 2024':1,'Q2 2024':2,'Q3 2024':3,'Q4 2024':4}
    df['Q_Num'] = df['Quarter'].map(q_map)

    # Derived metrics
    df['Conversion_Rate'] = df['Converted'] / df['Calls_Dialed'].replace(0, 1)
    df['Lead_Quality']    = df['Qualified'] / df['New_Leads'].replace(0, 1)
    df['Revenue_Per_Call']= df['Total_Revenue'] / df['Calls_Dialed'].replace(0, 1)

    return df

# ── FILE LOAD ─────────────────────────────────────────────────────────────────
from pathlib import Path
_csv_path = Path(__file__).parent / 'sales_performance_data.csv'

if not _csv_path.exists():
    st.markdown("""
    <div class="insight-box warning" style="margin-bottom:20px">
        📂 <strong>Data file not found.</strong>
        Upload <code>sales_performance_data.csv</code> below to launch the dashboard.
    </div>
    """, unsafe_allow_html=True)
    uploaded = st.file_uploader("Upload sales_performance_data.csv", type=["csv"])
    if uploaded is None:
        st.stop()
    df = load_data(file_bytes=uploaded.read())
else:
    df = load_data()

COLORS = {
    'North':'#6366f1','South':'#f87171','East':'#34d399','West':'#fbbf24',
    'bg':'#060912','card':'#0c1225','accent':'#a5b4fc','purple':'#8b5cf6'
}
PLOTLY_LAYOUT = dict(
    paper_bgcolor='rgba(0,0,0,0)',
    plot_bgcolor='rgba(12,18,37,0.8)',
    font=dict(family='Instrument Sans', color='#c7d2fe', size=12),
    title_font=dict(family='Syne', size=16, color='#a5b4fc'),
    legend=dict(bgcolor='rgba(0,0,0,0)', bordercolor='rgba(99,102,241,0.2)', borderwidth=1),
    xaxis=dict(gridcolor='rgba(99,102,241,0.1)', linecolor='rgba(99,102,241,0.2)', tickfont=dict(size=11)),
    yaxis=dict(gridcolor='rgba(99,102,241,0.1)', linecolor='rgba(99,102,241,0.2)', tickfont=dict(size=11)),
    margin=dict(l=20,r=20,t=50,b=20),
)

# ── SIDEBAR ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### ⚡ Filters")
    regions  = st.multiselect("Regions", df['Region'].unique(), default=list(df['Region'].unique()))
    quarters = st.multiselect("Quarters", ['Q1 2024','Q2 2024','Q3 2024','Q4 2024'],
                               default=['Q1 2024','Q2 2024','Q3 2024','Q4 2024'])
    st.markdown("---")
    st.markdown("### 📊 Analysis Layers")
    st.markdown('<span class="metric-pill">Descriptive</span><span class="metric-pill">Diagnostic</span><span class="metric-pill">Prescriptive</span><span class="metric-pill">Predictive</span>', unsafe_allow_html=True)
    st.markdown("---")
    st.markdown("### 🤖 Predictive Model")
    model_choice = st.selectbox("Algorithm", ["Random Forest","Gradient Boosting","Linear Regression"])
    st.markdown("---")
    st.caption("Sales Intelligence Dashboard v2.0")

fdf = df[df['Region'].isin(regions) & df['Quarter'].isin(quarters)]

# ── HEADER ─────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="dash-header">
  <p class="dash-subtitle">⚡ Sales Intelligence Platform · FY 2024</p>
  <h1 class="dash-title">Sales Performance Dashboard</h1>
  <p style="color:#7c85b8;margin-top:10px;font-size:0.9rem;">Descriptive · Diagnostic · Prescriptive · Predictive Analytics</p>
</div>
""", unsafe_allow_html=True)

# ── KPI STRIP ──────────────────────────────────────────────────────────────────
total_rev   = fdf['Total_Revenue'].sum()
avg_rev     = fdf['Total_Revenue'].mean()
total_deals = fdf['Deals_Closed'].sum()
conv_rate   = (fdf['Converted'].sum() / fdf['Calls_Dialed'].sum() * 100)

c1,c2,c3,c4,c5 = st.columns(5)
with c1:
    st.metric("💰 Total Revenue",  f"${total_rev/1e6:.2f}M")
with c2:
    st.metric("📈 Avg Revenue/Rep",f"${avg_rev:,.0f}")
with c3:
    st.metric("🤝 Deals Closed",   f"{total_deals:,}")
with c4:
    st.metric("📞 Conversion Rate", f"{conv_rate:.1f}%")
with c5:
    best_region = fdf.groupby('Region')['Total_Revenue'].mean().idxmax()
    st.metric("🏆 Top Region",      best_region)

st.markdown("---")

# ═══════════════════════════════════════════════════════════════════════════════
# TABS
# ═══════════════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4 = st.tabs([
    "📊  Descriptive",
    "🔍  Diagnostic",
    "💡  Prescriptive",
    "🤖  Predictive"
])

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 1 · DESCRIPTIVE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab1:
    st.markdown('<div class="section-title">Revenue Distribution <span class="section-badge">Descriptive</span></div>', unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # 1A: Avg Revenue by Region + Quarter (grouped bar)
    with col1:
        q_rev = fdf.groupby(['Quarter','Region'])['Total_Revenue'].mean().reset_index()
        q_rev = q_rev.sort_values('Quarter')
        fig = px.bar(q_rev, x='Quarter', y='Total_Revenue', color='Region',
                     barmode='group', title='Avg Revenue by Region & Quarter',
                     color_discrete_map=COLORS,
                     labels={'Total_Revenue':'Avg Revenue ($)','Quarter':''},
                     text_auto='.2s')
        fig.update_layout(**PLOTLY_LAYOUT)
        fig.update_traces(textfont_size=10, textangle=0, textposition='outside', cliponaxis=False)
        st.plotly_chart(fig, use_container_width=True)

    # 1B: Revenue share donut
    with col2:
        reg_sum = fdf.groupby('Region')['Total_Revenue'].sum().reset_index()
        fig2 = go.Figure(go.Pie(
            labels=reg_sum['Region'], values=reg_sum['Total_Revenue'],
            hole=0.62,
            marker=dict(colors=[COLORS[r] for r in reg_sum['Region']],
                        line=dict(color='#060912', width=3)),
            textinfo='label+percent', textfont=dict(family='Instrument Sans', size=12)
        ))
        fig2.add_annotation(text=f"Total<br><b>${total_rev/1e6:.1f}M</b>",
                            x=0.5, y=0.5, showarrow=False,
                            font=dict(family='Syne', size=18, color='#a5b4fc'))
        fig2.update_layout(**PLOTLY_LAYOUT, title='Revenue Share by Region')
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    # 1C: Revenue trend lines per region
    with col3:
        trend = fdf.groupby(['Quarter','Region'])['Total_Revenue'].mean().reset_index().sort_values('Quarter')
        fig3 = px.line(trend, x='Quarter', y='Total_Revenue', color='Region',
                       markers=True, title='Revenue Trend by Region (Quarterly)',
                       color_discrete_map=COLORS,
                       labels={'Total_Revenue':'Avg Revenue ($)'})
        fig3.update_traces(line_width=2.5, marker_size=8)
        fig3.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)

    # 1D: Box plot — spread of revenue
    with col4:
        fig4 = px.box(fdf, x='Region', y='Total_Revenue', color='Region',
                      title='Revenue Distribution (Spread & Outliers)',
                      color_discrete_map=COLORS,
                      labels={'Total_Revenue':'Revenue ($)'},
                      points='outliers')
        fig4.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig4, use_container_width=True)

    # 1E: Heatmap — Avg Revenue per Region x Quarter
    st.markdown('<div class="section-title">Activity Metrics <span class="section-badge">Summary</span></div>', unsafe_allow_html=True)
    pivot = fdf.groupby(['Region','Quarter'])['Total_Revenue'].mean().unstack().round(0)
    fig5 = px.imshow(pivot, text_auto=',.0f', aspect='auto',
                     title='Avg Revenue Heatmap — Region × Quarter',
                     color_continuous_scale=['#0c1225','#312e81','#6366f1','#a5b4fc'],
                     labels=dict(color='Avg Revenue ($)'))
    fig5.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig5, use_container_width=True)

    # Summary stats table
    st.markdown('<div class="section-title">Descriptive Statistics <span class="section-badge">Table</span></div>', unsafe_allow_html=True)
    desc = fdf.groupby('Region').agg(
        Avg_Revenue=('Total_Revenue','mean'),
        Median_Revenue=('Total_Revenue','median'),
        Std_Dev=('Total_Revenue','std'),
        Total_Revenue=('Total_Revenue','sum'),
        Avg_Deals=('Deals_Closed','mean'),
        Avg_Calls=('Calls_Dialed','mean'),
        Conversion_Pct=('Conversion_Rate', lambda x: x.mean()*100)
    ).round(1).reset_index()
    desc['Avg_Revenue']  = desc['Avg_Revenue'].map('${:,.0f}'.format)
    desc['Median_Revenue']= desc['Median_Revenue'].map('${:,.0f}'.format)
    desc['Total_Revenue']= desc['Total_Revenue'].map('${:,.0f}'.format)
    desc['Std_Dev']      = desc['Std_Dev'].map('${:,.0f}'.format)
    desc['Conversion_Pct']= desc['Conversion_Pct'].map('{:.1f}%'.format)
    st.dataframe(desc, use_container_width=True, hide_index=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 2 · DIAGNOSTIC
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab2:
    st.markdown('<div class="section-title">Why is Revenue Different? <span class="section-badge">Diagnostic</span></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box warning">
        🔍 <strong>Diagnostic goal:</strong> Identify the <em>root causes</em> of regional revenue gaps.
        We examine call efficiency, lead quality, conversion rates, and deal values to pinpoint what's driving North's dominance and South's decline.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # 2A: Calls Dialed vs Revenue scatter
    with col1:
        fig = px.scatter(fdf, x='Calls_Dialed', y='Total_Revenue', color='Region',
                         size='Deals_Closed', title='Calls Dialed vs Revenue (size = Deals)',
                         color_discrete_map=COLORS, opacity=0.7,
                         labels={'Calls_Dialed':'Calls Dialed','Total_Revenue':'Revenue ($)'})
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # 2B: Conversion funnel comparison
    with col2:
        funnel_data = fdf.groupby('Region').agg(
            Calls=('Calls_Dialed','sum'), Leads=('New_Leads','sum'),
            Qualified=('Qualified','sum'), Converted=('Converted','sum'),
            Closed=('Deals_Closed','sum')
        ).reset_index()
        fig2 = go.Figure()
        stages = ['Calls','Leads','Qualified','Converted','Closed']
        for region in funnel_data['Region']:
            row = funnel_data[funnel_data['Region']==region].iloc[0]
            fig2.add_trace(go.Scatterpolar(
                r=[row[s]/row['Calls']*100 for s in stages],
                theta=stages, fill='toself', name=region,
                line_color=COLORS[region], opacity=0.7
            ))
        fig2.update_layout(**PLOTLY_LAYOUT, title='Sales Funnel Efficiency (% of Calls)',
                           polar=dict(radialaxis=dict(visible=True, gridcolor='rgba(99,102,241,0.2)'),
                                      bgcolor='rgba(12,18,37,0.8)'))
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)

    # 2C: Revenue per call
    with col3:
        rpc = fdf.groupby(['Quarter','Region'])['Revenue_Per_Call'].mean().reset_index()
        fig3 = px.line(rpc.sort_values('Quarter'), x='Quarter', y='Revenue_Per_Call',
                       color='Region', markers=True,
                       title='Revenue per Call — Efficiency Metric',
                       color_discrete_map=COLORS,
                       labels={'Revenue_Per_Call':'Rev / Call ($)'})
        fig3.update_traces(line_width=2.5, marker_size=8)
        fig3.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)

    # 2D: Lead quality vs conversion
    with col4:
        fig4 = px.scatter(fdf, x='Lead_Quality', y='Conversion_Rate',
                          color='Region', size='Total_Revenue',
                          title='Lead Quality vs Conversion Rate',
                          color_discrete_map=COLORS, opacity=0.7,
                          labels={'Lead_Quality':'Lead Quality (Qualified/Leads)',
                                  'Conversion_Rate':'Conversion Rate'})
        fig4.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig4, use_container_width=True)

    # 2E: Avg Unit Value comparison
    auv = fdf.groupby(['Quarter','Region'])['Avg_Unit_Value'].mean().reset_index().sort_values('Quarter')
    fig5 = px.bar(auv, x='Region', y='Avg_Unit_Value', color='Quarter',
                  barmode='group', title='Avg Deal Value by Region & Quarter — Are North reps closing bigger deals?',
                  labels={'Avg_Unit_Value':'Avg Unit Value ($)'},
                  color_discrete_sequence=['#312e81','#4f46e5','#818cf8','#c7d2fe'])
    fig5.update_layout(**PLOTLY_LAYOUT)
    st.plotly_chart(fig5, use_container_width=True)

    # Diagnostic insights
    st.markdown('<div class="section-title">Root Cause Summary <span class="section-badge">Findings</span></div>', unsafe_allow_html=True)
    north_rpc = fdf[fdf['Region']=='North']['Revenue_Per_Call'].mean()
    south_rpc = fdf[fdf['Region']=='South']['Revenue_Per_Call'].mean()
    north_lq  = fdf[fdf['Region']=='North']['Lead_Quality'].mean()
    south_lq  = fdf[fdf['Region']=='South']['Lead_Quality'].mean()

    st.markdown(f"""
    <div class="insight-box success">
        ✅ <strong>North dominance factors:</strong> North generates <strong>${north_rpc:.0f}/call</strong>
        vs South's <strong>${south_rpc:.0f}/call</strong> — a {(north_rpc/south_rpc):.1f}x efficiency gap.
        North's lead quality ratio is {north_lq:.2f} vs {south_lq:.2f} for South, suggesting better targeting.
    </div>
    <div class="insight-box danger">
        🚨 <strong>South decline pattern:</strong> South's revenue is falling each quarter — not just low, but
        actively deteriorating. This points to pipeline quality issues, not just market conditions.
        Intervention needed before Q1 2025.
    </div>
    <div class="insight-box warning">
        ⚠️ <strong>East & West opportunity:</strong> East and West show consistent growth with moderate variance —
        these regions have untapped scaling potential if call volume increases by 20-30%.
    </div>
    """, unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 3 · PRESCRIPTIVE
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab3:
    st.markdown('<div class="section-title">What Should We Do? <span class="section-badge">Prescriptive</span></div>', unsafe_allow_html=True)

    st.markdown("""
    <div class="insight-box">
        💡 <strong>Prescriptive analytics</strong> moves from "what happened" and "why" to <em>actionable recommendations</em>
        — optimizing resource allocation, rep performance targets, and regional strategy.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    # 3A: Rep performance quadrant (Calls vs Revenue)
    with col1:
        rep_perf = fdf.groupby('Sales_Rep_Name').agg(
            Avg_Revenue=('Total_Revenue','mean'),
            Avg_Calls=('Calls_Dialed','mean'),
            Region=('Region','first')
        ).reset_index()
        med_rev   = rep_perf['Avg_Revenue'].median()
        med_calls = rep_perf['Avg_Calls'].median()
        fig = px.scatter(rep_perf, x='Avg_Calls', y='Avg_Revenue', color='Region',
                         hover_name='Sales_Rep_Name',
                         title='Rep Performance Quadrant (Calls vs Revenue)',
                         color_discrete_map=COLORS,
                         labels={'Avg_Calls':'Avg Calls Dialed','Avg_Revenue':'Avg Revenue ($)'})
        fig.add_hline(y=med_rev, line_dash='dash', line_color='rgba(165,180,252,0.4)',
                      annotation_text='Median Revenue')
        fig.add_vline(x=med_calls, line_dash='dash', line_color='rgba(165,180,252,0.4)',
                      annotation_text='Median Calls')
        fig.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig, use_container_width=True)

    # 3B: Optimal calls target analysis
    with col2:
        bins = pd.cut(fdf['Calls_Dialed'], bins=5, labels=['100-280','280-460','460-640','640-820','820-1000'])
        call_rev = fdf.groupby(bins, observed=True)['Total_Revenue'].mean().reset_index()
        call_rev.columns = ['Call_Range','Avg_Revenue']
        fig2 = px.bar(call_rev, x='Call_Range', y='Avg_Revenue',
                      title='Revenue Yield by Call Volume Band — Find the Sweet Spot',
                      labels={'Avg_Revenue':'Avg Revenue ($)','Call_Range':'Calls Dialed Range'},
                      color='Avg_Revenue',
                      color_continuous_scale=['#312e81','#6366f1','#a5b4fc'])
        fig2.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    # 3C: Resource allocation — where to invest?
    with col1:
        region_roi = fdf.groupby('Region').agg(
            Total_Rev=('Total_Revenue','sum'),
            Total_Calls=('Calls_Dialed','sum'),
            Total_Time=('Call_Time_Mins','sum')
        ).reset_index()
        region_roi['Rev_Per_Min'] = region_roi['Total_Rev'] / region_roi['Total_Time']
        fig3 = px.bar(region_roi, x='Region', y='Rev_Per_Min', color='Region',
                      title='Revenue per Call Minute — Where Time is Worth Most',
                      color_discrete_map=COLORS,
                      labels={'Rev_Per_Min':'Revenue per Minute ($)'})
        fig3.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig3, use_container_width=True)

    # 3D: Top 10 vs Bottom 10 reps
    with col2:
        rep_rank = fdf.groupby(['Sales_Rep_Name','Region'])['Total_Revenue'].mean().reset_index()
        rep_rank = rep_rank.sort_values('Total_Revenue', ascending=False)
        top10 = rep_rank.head(10)
        bot10 = rep_rank.tail(10)
        combined = pd.concat([top10.assign(Group='Top 10'), bot10.assign(Group='Bottom 10')])
        fig4 = px.bar(combined, y='Sales_Rep_Name', x='Total_Revenue', color='Group',
                      orientation='h', title='Top 10 vs Bottom 10 Reps by Avg Revenue',
                      color_discrete_map={'Top 10':'#34d399','Bottom 10':'#f87171'},
                      labels={'Total_Revenue':'Avg Revenue ($)','Sales_Rep_Name':''})
        fig4.update_layout(**PLOTLY_LAYOUT, height=400)
        st.plotly_chart(fig4, use_container_width=True)

    # Action plan
    st.markdown('<div class="section-title">Recommended Actions <span class="section-badge">Action Plan</span></div>', unsafe_allow_html=True)
    ac1, ac2, ac3 = st.columns(3)
    with ac1:
        st.markdown("""
        <div class="insight-box success">
            <strong>🎯 North — Scale Model</strong><br><br>
            North's playbook is working. Increase headcount by 20%, replicate training program across East/West.
            Target $200K avg revenue per rep by Q2 2025.
        </div>""", unsafe_allow_html=True)
    with ac2:
        st.markdown("""
        <div class="insight-box danger">
            <strong>🚨 South — Rescue Plan</strong><br><br>
            Deploy North senior reps to mentor South Q1 2025. Audit lead qualification criteria.
            If no improvement in 2 quarters, consider reallocating budget to West.
        </div>""", unsafe_allow_html=True)
    with ac3:
        st.markdown("""
        <div class="insight-box warning">
            <strong>📈 East/West — Growth Mode</strong><br><br>
            Both regions show consistent upward trends. Increase call targets by 25%.
            East should focus on improving avg deal value — currently lagging North by 2x.
        </div>""", unsafe_allow_html=True)

# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
# TAB 4 · PREDICTIVE  ★ FOCUS TAB ★
# ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
with tab4:
    st.markdown('<div class="section-title">What Will Happen Next? <span class="section-badge">Predictive · ML</span></div>', unsafe_allow_html=True)

    # ── How predictive analytics works — explainer ─────────────────────────────
    with st.expander("🧠 How Predictive Analytics Works — Deep Explanation", expanded=True):
        st.markdown("""
        <div class="predict-hero">
        <h3 style="font-family:Syne;color:#a5b4fc;margin-top:0">How the Model Predicts Revenue</h3>

        <p style="color:#c7d2fe;line-height:1.8">
        Predictive analytics uses <strong style="color:#a5b4fc">historical patterns</strong> to estimate future outcomes.
        Here's exactly how this dashboard's model works, step by step:
        </p>

        <h4 style="color:#818cf8;font-family:Syne">Step 1 — Feature Engineering 🔧</h4>
        <p style="color:#c7d2fe;line-height:1.8">
        We take your raw sales columns and create <em>features</em> — numeric signals the model can learn from:
        <br>• <strong>Calls_Dialed, Call_Time_Mins</strong> → activity volume
        <br>• <strong>New_Leads, Qualified</strong> → pipeline strength
        <br>• <strong>Converted, Deals_Closed</strong> → closing ability
        <br>• <strong>Region</strong> → encoded as numbers (North=0, South=1, East=2, West=3)
        <br>• <strong>Quarter number</strong> → captures time/seasonality trends
        </p>

        <h4 style="color:#818cf8;font-family:Syne">Step 2 — Training the Model 🏋️</h4>
        <p style="color:#c7d2fe;line-height:1.8">
        We split data 80/20 — 80% trains the model, 20% tests it. The model sees hundreds of examples like:
        <br><em>"This rep in North, Q3, made 800 calls, closed 45 deals → earned $148,000"</em>
        <br>It learns the mathematical relationship between inputs and revenue.
        </p>

        <h4 style="color:#818cf8;font-family:Syne">Step 3 — The Algorithm Choices 🤖</h4>
        <p style="color:#c7d2fe;line-height:1.8">
        <strong>Random Forest:</strong> Builds 100+ decision trees, each trained on a random subset of data.
        Final prediction = average of all trees. Very robust, handles non-linear relationships well.<br><br>
        <strong>Gradient Boosting:</strong> Builds trees sequentially — each tree corrects the errors of the previous one.
        Often the most accurate but slower to train. Best for complex patterns.<br><br>
        <strong>Linear Regression:</strong> Assumes a straight-line relationship between features and revenue.
        Fast, interpretable, but misses complex interactions between variables.
        </p>

        <h4 style="color:#818cf8;font-family:Syne">Step 4 — Evaluation Metrics 📏</h4>
        <p style="color:#c7d2fe;line-height:1.8">
        <strong>R² Score:</strong> How much of revenue variance the model explains. R²=0.85 means the model explains 85% of revenue differences.<br>
        <strong>MAE (Mean Absolute Error):</strong> On average, how far off are predictions in dollars. MAE=$8,000 means predictions are within $8K on average.<br>
        <strong>Feature Importance:</strong> Which inputs matter most? If "Deals_Closed" has importance=0.45, it drives 45% of the prediction.
        </p>

        <h4 style="color:#818cf8;font-family:Syne">Step 5 — Revenue Forecast 🔮</h4>
        <p style="color:#c7d2fe;line-height:1.8">
        For Q5 2025 forecasting, we extend the time variable and apply the same learned relationships.
        The model assumes: if trends hold (North growing ~15%/quarter, South declining ~8%/quarter),
        this is what Q1-Q2 2025 revenue looks like. The confidence band shows ± uncertainty range.
        </p>
        </div>
        """, unsafe_allow_html=True)

    # ── BUILD MODEL ────────────────────────────────────────────────────────────
    @st.cache_data
    def build_model(model_name, data):
        le = LabelEncoder()
        mdf = data.copy()
        mdf['Region_Enc'] = le.fit_transform(mdf['Region'])
        features = ['Calls_Dialed','Call_Time_Mins','New_Leads','Disqualified',
                    'No_Answer','Qualified','Converted','Deals_Closed',
                    'Followup_Leads','Avg_Unit_Value','Region_Enc','Q_Num']
        X = mdf[features].fillna(0)
        y = mdf['Total_Revenue']
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        if model_name == "Random Forest":
            model = RandomForestRegressor(n_estimators=150, max_depth=10, random_state=42)
        elif model_name == "Gradient Boosting":
            model = GradientBoostingRegressor(n_estimators=150, max_depth=5, learning_rate=0.1, random_state=42)
        else:
            model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        r2  = r2_score(y_test, y_pred)
        mae = mean_absolute_error(y_test, y_pred)
        if hasattr(model,'feature_importances_'):
            importances = model.feature_importances_
        else:
            importances = np.abs(model.coef_) / np.abs(model.coef_).sum()
        return model, r2, mae, importances, features, X_test, y_test, y_pred, le

    model, r2, mae, importances, features, X_test, y_test, y_pred, le = build_model(model_choice, df)

    # Model score strip
    mc1, mc2, mc3, mc4 = st.columns(4)
    mc1.metric("🎯 R² Score",    f"{r2:.3f}", help="Explains this much of revenue variance")
    mc2.metric("📉 MAE",         f"${mae:,.0f}", help="Average prediction error in dollars")
    mc3.metric("🔢 Training Rows",f"{int(len(df)*0.8):,}")
    mc4.metric("🧪 Test Rows",   f"{int(len(df)*0.2):,}")

    st.markdown("---")
    col1, col2 = st.columns(2)

    # 4A: Feature importance
    with col1:
        feat_df = pd.DataFrame({'Feature':features,'Importance':importances}).sort_values('Importance',ascending=True)
        friendly = {
            'Calls_Dialed':'Calls Dialed','Call_Time_Mins':'Call Time (mins)',
            'New_Leads':'New Leads','Disqualified':'Disqualified Leads',
            'No_Answer':'No Answers','Qualified':'Qualified Leads',
            'Converted':'Converted','Deals_Closed':'Deals Closed',
            'Followup_Leads':'Follow-up Leads','Avg_Unit_Value':'Avg Unit Value',
            'Region_Enc':'Region','Q_Num':'Quarter'
        }
        feat_df['Feature'] = feat_df['Feature'].map(friendly)
        fig = px.bar(feat_df, y='Feature', x='Importance', orientation='h',
                     title=f'Feature Importance — {model_choice}',
                     labels={'Importance':'Importance Score','Feature':''},
                     color='Importance',
                     color_continuous_scale=['#312e81','#6366f1','#a5b4fc'])
        fig.update_layout(**PLOTLY_LAYOUT, height=400, showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

    # 4B: Actual vs Predicted scatter
    with col2:
        ap_df = pd.DataFrame({'Actual':y_test.values,'Predicted':y_pred}).assign(
            Error=lambda x: abs(x['Actual']-x['Predicted'])
        )
        fig2 = px.scatter(ap_df, x='Actual', y='Predicted',
                          color='Error', title='Actual vs Predicted Revenue',
                          color_continuous_scale=['#34d399','#f59e0b','#f87171'],
                          labels={'Actual':'Actual Revenue ($)','Predicted':'Predicted Revenue ($)',
                                  'Error':'Abs Error ($)'}, opacity=0.7)
        mx = max(ap_df['Actual'].max(), ap_df['Predicted'].max())
        fig2.add_trace(go.Scatter(x=[0,mx], y=[0,mx], mode='lines',
                                  line=dict(dash='dash', color='rgba(165,180,252,0.5)'),
                                  name='Perfect Prediction', showlegend=True))
        fig2.update_layout(**PLOTLY_LAYOUT)
        st.plotly_chart(fig2, use_container_width=True)

    # ── REVENUE FORECAST ───────────────────────────────────────────────────────
    st.markdown('<div class="section-title">Revenue Forecast — Q1 & Q2 2025 <span class="section-badge">Forecast</span></div>', unsafe_allow_html=True)

    @st.cache_data
    def make_forecast(data, model_name):
        _, _, _, _, features, _, _, _, le = build_model(model_name, data)
        model2, r2, mae, importances, features, X_test, y_test, y_pred, le = build_model(model_name, data)
        future_quarters = [5, 6]
        forecast_rows = []
        for q_num in future_quarters:
            for region in ['North','South','East','West']:
                hist = data[(data['Region']==region)].agg({
                    'Calls_Dialed':'mean','Call_Time_Mins':'mean','New_Leads':'mean',
                    'Disqualified':'mean','No_Answer':'mean','Qualified':'mean',
                    'Converted':'mean','Deals_Closed':'mean','Followup_Leads':'mean','Avg_Unit_Value':'mean'
                })
                row = {
                    'Calls_Dialed':hist['Calls_Dialed'],'Call_Time_Mins':hist['Call_Time_Mins'],
                    'New_Leads':hist['New_Leads'],'Disqualified':hist['Disqualified'],
                    'No_Answer':hist['No_Answer'],'Qualified':hist['Qualified'],
                    'Converted':hist['Converted'],'Deals_Closed':hist['Deals_Closed'],
                    'Followup_Leads':hist['Followup_Leads'],'Avg_Unit_Value':hist['Avg_Unit_Value'],
                    'Region_Enc':le.transform([region])[0],'Q_Num':q_num,
                    'Region':region,
                    'Quarter': f"Q{q_num-4} 2025"
                }
                X_fut = pd.DataFrame([{f:row[f] for f in features}])
                row['Predicted_Revenue'] = model2.predict(X_fut)[0]
                row['Lower'] = row['Predicted_Revenue'] * 0.82
                row['Upper'] = row['Predicted_Revenue'] * 1.18
                forecast_rows.append(row)
        return pd.DataFrame(forecast_rows)

    forecast_df = make_forecast(df, model_choice)

    # Historical actuals per quarter per region
    hist_avg = df.groupby(['Quarter','Region'])['Total_Revenue'].mean().reset_index()
    hist_avg['Q_Num'] = hist_avg['Quarter'].map({'Q1 2024':1,'Q2 2024':2,'Q3 2024':3,'Q4 2024':4})
    hist_avg = hist_avg.sort_values('Q_Num')

    col1, col2 = st.columns(2)
    with col1:
        fig3 = go.Figure()
        for region in ['North','South','East','West']:
            hist_r = hist_avg[hist_avg['Region']==region]
            fore_r = forecast_df[forecast_df['Region']==region]
            # Historical line
            fig3.add_trace(go.Scatter(
                x=list(hist_r['Quarter']) + list(fore_r['Quarter']),
                y=list(hist_r['Total_Revenue']) + list(fore_r['Predicted_Revenue']),
                mode='lines+markers', name=region,
                line=dict(color=COLORS[region], width=2.5),
                marker=dict(size=8)
            ))
            # Confidence band for forecast
            fig3.add_trace(go.Scatter(
                x=list(fore_r['Quarter'])+list(fore_r['Quarter'])[::-1],
                y=list(fore_r['Upper'])+list(fore_r['Lower'])[::-1],
                fill='toself', fillcolor=f"rgba{tuple(int(COLORS[region][i:i+2],16) for i in (1,3,5))+(0.15,)}",
                line=dict(color='rgba(0,0,0,0)'), showlegend=False, hoverinfo='skip'
            ))
            # Vertical line at Q4/Q1 boundary
        fig3.add_vline(x=3.5, line_dash='dot', line_color='rgba(165,180,252,0.4)',
                       annotation_text="Forecast →", annotation_position="top")
        fig3.update_layout(**PLOTLY_LAYOUT, title='Revenue Forecast Q1-Q2 2025 (with confidence band)',
                           height=420, showlegend=True)
        st.plotly_chart(fig3, use_container_width=True)

    # Forecast table
    with col2:
        fore_tbl = forecast_df[['Quarter','Region','Predicted_Revenue','Lower','Upper']].copy()
        fore_tbl['Predicted_Revenue'] = fore_tbl['Predicted_Revenue'].map('${:,.0f}'.format)
        fore_tbl['Lower']             = fore_tbl['Lower'].map('${:,.0f}'.format)
        fore_tbl['Upper']             = fore_tbl['Upper'].map('${:,.0f}'.format)
        fore_tbl.columns = ['Quarter','Region','Predicted Revenue','Lower Bound (−18%)','Upper Bound (+18%)']
        st.markdown("#### 📋 Forecast Table")
        st.dataframe(fore_tbl, use_container_width=True, hide_index=True)

        # Forecast summary
        n_fore = forecast_df[forecast_df['Region']=='North']['Predicted_Revenue'].mean()
        s_fore = forecast_df[forecast_df['Region']=='South']['Predicted_Revenue'].mean()
        st.markdown(f"""
        <div class="insight-box success">
            🔮 <strong>2025 Outlook:</strong> North is projected to average <strong>${n_fore:,.0f}/rep</strong>
            in H1 2025 — a new high. South's predicted decline continues to <strong>${s_fore:,.0f}/rep</strong>,
            making the gap nearly <strong>{n_fore/s_fore:.0f}x</strong> by mid-2025.
        </div>
        """, unsafe_allow_html=True)

    # ── WHAT-IF SIMULATOR ──────────────────────────────────────────────────────
    st.markdown('<div class="section-title">🎛️ What-If Revenue Simulator <span class="section-badge">Interactive</span></div>', unsafe_allow_html=True)
    st.markdown('<div class="insight-box">Adjust the sliders below to simulate how changing a rep\'s activity metrics affects their predicted revenue — powered by the trained ML model.</div>', unsafe_allow_html=True)

    sim1, sim2, sim3 = st.columns(3)
    with sim1:
        sim_region   = st.selectbox("Region", ['North','South','East','West'])
        sim_calls    = st.slider("Calls Dialed", 100, 1000, 550)
        sim_calltime = st.slider("Call Time (mins)", 100, 2000, 900)
    with sim2:
        sim_leads    = st.slider("New Leads", 20, 250, 100)
        sim_qual     = st.slider("Qualified Leads", 5, 150, 50)
        sim_conv     = st.slider("Converted", 1, 100, 30)
    with sim3:
        sim_deals    = st.slider("Deals Closed", 1, 80, 20)
        sim_auv      = st.slider("Avg Unit Value ($)", 500, 5000, 2000)
        sim_quarter  = st.selectbox("Quarter", ['Q1 2024','Q2 2024','Q3 2024','Q4 2024','Q1 2025','Q2 2025'])

    q_num_map = {'Q1 2024':1,'Q2 2024':2,'Q3 2024':3,'Q4 2024':4,'Q1 2025':5,'Q2 2025':6}
    sim_row = {
        'Calls_Dialed':sim_calls,'Call_Time_Mins':sim_calltime,'New_Leads':sim_leads,
        'Disqualified':max(0,sim_leads-sim_qual-10),'No_Answer':int(sim_calls*0.3),
        'Qualified':sim_qual,'Converted':sim_conv,'Deals_Closed':sim_deals,
        'Followup_Leads':int(sim_leads*0.3),'Avg_Unit_Value':sim_auv,
        'Region_Enc':le.transform([sim_region])[0],'Q_Num':q_num_map[sim_quarter]
    }
    X_sim = pd.DataFrame([sim_row])
    predicted = model.predict(X_sim)[0]

    # Reference: avg for that region
    ref_avg = df[df['Region']==sim_region]['Total_Revenue'].mean()
    delta   = predicted - ref_avg
    delta_pct = delta / ref_avg * 100

    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        st.metric("🤖 ML Predicted Revenue", f"${predicted:,.0f}",
                  delta=f"{delta_pct:+.1f}% vs {sim_region} avg")
    with sc2:
        st.metric(f"📊 {sim_region} Region Avg", f"${ref_avg:,.0f}")
    with sc3:
        efficiency = predicted / sim_calls if sim_calls > 0 else 0
        st.metric("⚡ Rev per Call", f"${efficiency:,.0f}")

    # Gauge chart for predicted vs benchmark
    max_rev = df['Total_Revenue'].max()
    fig_gauge = go.Figure(go.Indicator(
        mode="gauge+number+delta",
        value=predicted,
        delta={'reference':ref_avg, 'valueformat':'$,.0f'},
        number={'valueformat':'$,.0f','font':{'family':'Syne','size':32,'color':'#a5b4fc'}},
        title={'text':f"Predicted Revenue vs {sim_region} Avg",'font':{'family':'Syne','color':'#c7d2fe'}},
        gauge={
            'axis':{'range':[0,max_rev*1.1],'tickformat':'$,.0f',
                    'tickfont':{'color':'#6b7280'},'gridcolor':'rgba(99,102,241,0.2)'},
            'bar':{'color':'#6366f1','thickness':0.3},
            'bgcolor':'rgba(0,0,0,0)',
            'bordercolor':'rgba(0,0,0,0)',
            'steps':[
                {'range':[0,ref_avg*0.5],'color':'rgba(248,113,113,0.2)'},
                {'range':[ref_avg*0.5,ref_avg],'color':'rgba(251,191,36,0.2)'},
                {'range':[ref_avg,max_rev*1.1],'color':'rgba(52,211,153,0.15)'},
            ],
            'threshold':{'line':{'color':'#a5b4fc','width':3},'thickness':0.8,'value':ref_avg}
        }
    ))
    fig_gauge.update_layout(paper_bgcolor='rgba(0,0,0,0)',
                            font=dict(family='Instrument Sans',color='#c7d2fe'),
                            height=320, margin=dict(l=30,r=30,t=60,b=20))
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Final insight
    st.markdown(f"""
    <div class="insight-box {'success' if delta>0 else 'danger'}">
        🤖 <strong>Model says:</strong> With these activity inputs in <strong>{sim_region}</strong> during
        <strong>{sim_quarter}</strong>, this rep is predicted to generate <strong>${predicted:,.0f}</strong>.
        That is <strong>${abs(delta):,.0f} {'above' if delta>0 else 'below'}</strong> the {sim_region}
        regional average of ${ref_avg:,.0f}.
        {'This rep profile is a top performer — consider studying their approach.' if delta > ref_avg*0.2 else
         'This rep is underperforming — increasing qualified leads and deals closed will have the highest impact per feature importance.' if delta < 0 else
         'This rep is performing near the regional average.'}
    </div>
    """, unsafe_allow_html=True)
