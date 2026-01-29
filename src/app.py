import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

st.set_page_config(page_title="MMM Dashboard", layout="wide")

BASE_DIR = Path.cwd()
PROC_DIR = BASE_DIR / "data_processed"

st.title("🚀 Marketing Mix Revenue Optimizer")

@st.cache_data
def train_model():
    dim_channel = pd.read_csv(PROC_DIR / "dim_channel.csv")
    fact_marketing = pd.read_csv(PROC_DIR / "fact_marketing.csv")
    fact_revenue = pd.read_csv(PROC_DIR / "fact_revenue.csv")
    
    pivot = fact_marketing.pivot_table(
        index='date_key', columns='channel_id', values='spend', aggfunc='sum'
    ).fillna(0).reset_index()
    
    mmm_data = pivot.merge(fact_revenue, on='date_key', how='inner')
    X = mmm_data.select_dtypes(include=[np.number]).drop(columns=['revenue'])
    y = mmm_data['revenue']
    
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    return model, dim_channel, r2_score(y, model.predict(X))

model, channels, r2 = train_model()

col_metrics = st.columns(2)
col_metrics[0].metric("📊 Channels", len(channels))
col_metrics[1].metric("📈 Model R²", f"{r2:.3f}")

st.markdown("---")

# TOTAL BUDGET SLIDER (0 to 100000)
total_budget = st.slider("💰 Total Budget", 0, 100000, 30000, 1000)

st.markdown("---")

# INDEPENDENT SLIDERS FOR EACH CHANNEL (0 to total_budget)
st.subheader("💵 Budget Allocation by Channel")

col_left, col_right = st.columns([2, 1])

spend_allocation = {}
with col_left:
    for _, row in channels.iterrows():
        ch_name = row['channel_name']
        spend = st.slider(
            ch_name,
            min_value=0,
            max_value=total_budget,
            value=int(total_budget / len(channels)),
            step=100,
            key=f"spend_{ch_name}"
        )
        spend_allocation[ch_name] = spend

# SHOW CURRENT TOTAL
with col_right:
    st.subheader("Summary")
    total_spent = sum(spend_allocation.values())
    st.metric("Total Spent", f"${total_spent:,.0f}")
    st.metric("Total Budget", f"${total_budget:,.0f}")
    
    if total_spent == total_budget:
        st.success("✅ Budget matches!")
    else:
        diff = total_spent - total_budget
        if diff > 0:
            st.warning(f"⚠️ Over by ${diff:,.0f}")
        else:
            st.warning(f"⚠️ Under by ${abs(diff):,.0f}")

st.markdown("---")

# PREDICT BUTTON - VALIDATION ON CLICK
if st.button("🎯 Predict Revenue", type="primary"):
    total_spent = sum(spend_allocation.values())
    unallocated = total_budget - total_spent
    
    # If over budget - ERROR
    if total_spent > total_budget:
        st.error(f"❌ ERROR: Alloted amount exceeds the budget")
        st.stop()
    
    # If under budget - WARNING but still predict
    if unallocated > 0:
        st.warning(f"⚠️ Unallocated budget: ${unallocated:,.0f}")
    
    # PREDICT (works for both exact and under-budget)
    spend_vec = np.array([spend_allocation.get(ch, 0) for ch in channels['channel_name']])
    revenue = model.predict(spend_vec.reshape(1, -1))[0]
    
    col1, col2, col3 = st.columns(3)
    col1.metric("💰 Predicted Revenue", f"${revenue:,.0f}")
    col2.metric("💵 Total Spend", f"${total_spent:,.0f}")
    col3.metric("📈 Net Profit", f"${revenue - total_spent:,.0f}")

st.markdown("---")

# CHANNEL PERFORMANCE CHART
st.subheader("📊 Channel Impact")
spend_vec = np.array([spend_allocation.get(ch, 0) for ch in channels['channel_name']])
contrib = spend_vec * model.coef_

fig, ax = plt.subplots(figsize=(12, 6))
colors = ['green' if c > 0 else 'red' for c in model.coef_]
ax.bar(channels['channel_name'], contrib, color=colors)
ax.set_ylabel('Revenue Contribution')
ax.set_title('Channel Impact at Current Allocation')
plt.xticks(rotation=45)
st.pyplot(fig)

# ROI TABLE
st.subheader("📋 ROI Breakdown")
roi_data = []
for i, (_, row) in enumerate(channels.iterrows()):
    ch = row['channel_name']
    roi_data.append({
        'Channel': ch,
        'Allocated Budget': f"${spend_allocation.get(ch, 0):,.0f}",
        'ROI': f"{model.coef_[i]*100:.0%}",
        'Revenue Contribution': f"${contrib[i]:,.0f}"
    })

st.dataframe(pd.DataFrame(roi_data), use_container_width=True)
