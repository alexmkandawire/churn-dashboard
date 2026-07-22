# -*- coding: utf-8 -*-
"""
Created on Wed Jul  1 10:24:00 2026

@author: amkandawire
"""

import streamlit as st
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import anthropic

# Read API key from Streamlit secrets
api_key = st.secrets["ANTHROPIC_API_KEY"]

# ── Page config — must be first st. call ──────────
st.set_page_config(
    page_title="Churn Analytics",
    page_icon="📡",
    layout="wide"
)

# ── Cached model — runs once, never again ─────────
@st.cache_data
def build_model():
    np.random.seed(42)
    n = 200
    df = pd.DataFrame({
        'customer_id':   range(1, n + 1),
        'tenure_months': np.random.randint(1, 60, n),
        'monthly_spend': np.round(np.random.uniform(5, 80, n), 2),
        'num_complaints':np.random.randint(0, 6, n),
        'plan_type':     np.random.choice(['prepaid', 'postpaid'], n),
        'support_calls': np.random.randint(0, 8, n),
    })
    churn_prob = (
        (1 - df['tenure_months'] / 60) * 0.4 +
        (df['num_complaints']  / 5)  * 0.4 +
        (1 - df['monthly_spend']  / 80) * 0.2
    )
    df['churned'] = (np.random.rand(n) < churn_prob).astype(int)

    # Features
    df['plan_encoded'] = (df['plan_type'] == 'postpaid').astype(int)
    features = ['tenure_months', 'monthly_spend',
                'num_complaints', 'support_calls', 'plan_encoded']
    X = df[features]
    y = df['churned']

    # Train
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    # Score all customers
    df['churn_probability'] = model.predict_proba(X)[:, 1]
    df['risk_tier'] = pd.cut(
        df['churn_probability'],
        bins=[0, 0.4, 0.7, 1],
        labels=['🟢 Low', '🟡 Medium', '🔴 High']
    )

    # Feature importance
    importance = pd.DataFrame({
        'feature': features,
        'coefficient': model.coef_[0]
    }).sort_values('coefficient', ascending=False)

    return df, importance

def build_system_prompt(df, high_risk, med_risk, low_risk, priority, importance):
    top_driver    = importance.iloc[0]['feature']
    top_protector = importance.iloc[-1]['feature']
    spend_high = high_risk['monthly_spend'].mean()
    spend_low  = low_risk['monthly_spend'].mean()
    avg_comp_high = high_risk['num_complaints'].mean()
    avg_comp_low  = low_risk['num_complaints'].mean()
    return f"""You are an AI churn analyst embedded in a telecom analytics dashboard.
Answer questions using ONLY the figures below. Never invent numbers.
If a question cannot be answered from this data, say so clearly.

=== DATA SUMMARY ===
Total customers:        {len(df)}
Overall churn rate:     {df['churned'].mean():.1%}
High risk  (>70%):      {len(high_risk)} customers ({len(high_risk)/len(df):.1%})
Medium risk (40-70%):   {len(med_risk)} customers ({len(med_risk)/len(df):.1%})
Low risk   (<40%):      {len(low_risk)} customers ({len(low_risk)/len(df):.1%})
Revenue at risk:        ${high_risk['monthly_spend'].sum():,.2f}/month
Priority segment:       {len(priority)} customers (high risk + high spend)
Avg spend high-risk:    ${spend_high:.2f}/month
Avg spend low-risk:     ${spend_low:.2f}/month
Avg complaints high-risk: {avg_comp_high:.1f}
Avg complaints low-risk:  {avg_comp_low:.1f}
Top churn driver:       {top_driver}
Top churn protector:    {top_protector}
Prepaid high-risk:      {len(high_risk[high_risk['plan_type']=='prepaid'])} customers
Postpaid high-risk:     {len(high_risk[high_risk['plan_type']=='postpaid'])} customers

=== RULES ===
- Plain English only, no jargon
- Lead with the direct answer, then evidence
- Max 4 sentences unless a table genuinely helps
- Never invent figures not in the summary above
- If data is unavailable say: "I don't have that data in the current summary"
"""

def ask_claude(question, system_prompt, api_key):
    """Send a question to Claude and return the answer as a string."""
    try:
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=500,
            system=system_prompt,
            messages=[{"role": "user", "content": question}]
        )
        return response.content[0].text
    except Exception as e:
        return f"Sorry, I couldn't process that question. Error: {str(e)}"
    
# ── Load model output ─────────────────────────────
df, importance = build_model()

# ── Derived values ────────────────────────────────
high_risk   = df[df['churn_probability'] >= 0.7]
med_risk    = df[df['churn_probability'].between(0.4, 0.7)]
priority    = df[
    (df['churn_probability'] >= 0.7) &
    (df['monthly_spend'] >= df['monthly_spend'].median())
]
rev_at_risk = high_risk['monthly_spend'].sum()
top_driver  = importance.iloc[0]['feature']

# ── Dashboard ─────────────────────────────────────
st.title("📡 Churn Analytics Dashboard")
st.caption("AI-powered customer retention intelligence · synthetic data demo")

st.divider()

# Headline metrics
col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Customers",  len(df))
col2.metric("High Risk",
           len(high_risk),
           delta=f"{len(high_risk)/len(df):.1%} of base",
           delta_color="inverse")
col3.metric("Revenue at Risk / mo",
           f"${rev_at_risk:,.0f}")
col4.metric("Priority Segment",
           f"{len(priority)} customers",
           delta="high risk + high spend")

st.divider()

# Insight callout
avg_complaints_high = high_risk['num_complaints'].mean()
avg_complaints_low  = df[df['churn_probability'] < 0.4]['num_complaints'].mean()

st.info(f"💡 Top churn driver: **{top_driver}** — "
        f"high-risk customers average **{avg_complaints_high:.1f} complaints** "
        f"vs **{avg_complaints_low:.1f}** for low-risk customers. "
        f"Priority retention segment: **{len(priority)} customers** representing "
        f"**${priority['monthly_spend'].sum():,.0f}/mo** in revenue.")

# ── Sidebar filters ───────────────────────────────
with st.sidebar:
    st.header("🔍 Filters")

    risk_filter = st.multiselect(
        "Risk tier",
        options=['🔴 High', '🟡 Medium', '🟢 Low'],
        default=['🔴 High', '🟡 Medium']
    )

    plan_filter = st.multiselect(
        "Plan type",
        options=['prepaid', 'postpaid'],
        default=['prepaid', 'postpaid']
    )

    top_n = st.slider(
        "Show top N customers",
        min_value=10,
        max_value=100,
        value=20,
        step=10
    )

# ── Apply filters ─────────────────────────────────
filtered = df[
    (df['risk_tier'].astype(str).isin(risk_filter)) &
    (df['plan_type'].isin(plan_filter))
].sort_values('churn_probability', ascending=False).head(top_n)

# ── Screen 1: At-risk table ───────────────────────
st.header("🔴 At-Risk Customers")
st.caption(f"Showing top {len(filtered)} customers · sorted by churn probability")

display_cols = [
    'customer_id', 'tenure_months', 'monthly_spend',
    'num_complaints', 'plan_type',
    'churn_probability', 'risk_tier'
]


display_df = filtered[display_cols].copy()
display_df['churn_%'] = (display_df['churn_probability'] * 100).round(1).astype(str) + '%'

display_order = [
    'customer_id', 'tenure_months', 'monthly_spend',
    'num_complaints', 'plan_type', 'churn_%',
    'churn_probability', 'risk_tier'
]

st.dataframe(
    display_df[display_order],
    use_container_width=True,
    hide_index=True,
    column_config={
        "churn_probability": st.column_config.ProgressColumn(
            "Risk Bar",
            min_value=0, max_value=1,
            format=" "
        ),
        "churn_%": st.column_config.TextColumn("Churn Probability"),
        "monthly_spend": st.column_config.NumberColumn(
            "Monthly Spend", format="$%.2f"
        ),
        "risk_tier": st.column_config.TextColumn("Risk Tier"),
        "customer_id": st.column_config.NumberColumn("Customer ID"),
    }
)
# Download button — export filtered list to CSV
st.download_button(
    label="⬇ Download filtered list as CSV",
    data=filtered[display_cols].to_csv(index=False),
    file_name="at_risk_customers.csv",
    mime="text/csv"
)

st.divider()
st.header("📊 Churn Insights")

# ── Row 1: two charts side by side ────────────────
chart_col1, chart_col2 = st.columns(2)

with chart_col1:
    st.subheader("Risk Distribution")
    risk_counts = filtered['risk_tier'].astype(str)\
                .value_counts()\
                .reindex(['🔴 High', '🟡 Medium', '🟢 Low'])\
                .fillna(0)                    
    st.bar_chart(risk_counts)
    st.caption(f"High: {int(risk_counts['🔴 High'])} · "
               f"Medium: {int(risk_counts['🟡 Medium'])} · "
               f"Low: {int(risk_counts['🟢 Low'])}")

with chart_col2:
    st.subheader("Top Churn Drivers")
    # Show absolute coefficient values for clean bar chart
    imp_display = importance.copy()
    imp_display['impact'] = imp_display['coefficient'].abs()
    imp_display['direction'] = imp_display['coefficient']\
                                    .apply(lambda x: '↑ churn' if x > 0 else '↓ churn')
    imp_display = imp_display.sort_values('impact', ascending=True)
    st.bar_chart(imp_display.set_index('feature')['impact'],
                horizontal=True)
    st.caption("Bar length = strength of influence on churn ")
    st.caption("🟢 plan_encoded & tenure_months reduce churn risk· "
               "🔴 num_complaints & support_calls increase churn risk"
               )
        
    

# ── Row 2: spend by risk tier — full width ────────
st.subheader("Average Monthly Spend by Risk Tier")
#spend_by_plan = df.groupby('plan_type')['monthly_spend'].mean().round(2)
#st.bar_chart(spend_by_plan)
spend_by_risk = filtered.groupby(
    filtered['risk_tier'].astype(str)
)['monthly_spend'].mean()\
  .reindex(['🔴 High', '🟡 Medium', '🟢 Low'])\
  .round(2)


st.bar_chart(spend_by_risk)
st.caption(
    f"High risk avg: ${spend_by_risk['🔴 High']:.2f} · "
    f"Medium avg: ${spend_by_risk['🟡 Medium']:.2f} · "
    f"Low risk avg: ${spend_by_risk['🟢 Low']:.2f} · "
    "High-risk customers spend less — prioritise high-risk + high-spend segment."
)

st.divider()
st.header("💬 Ask the Data")
st.caption("Ask any question about your customer churn data in plain English")

# Build system prompt from live model output
low_risk       = df[df['churn_probability'] < 0.4]
system_prompt  = build_system_prompt(
    df, high_risk, med_risk, low_risk, priority, importance)

# Example questions to guide the user
st.markdown("**Try asking:** What is our biggest churn risk? · "
           "Which customers should we call first? · "
           "How does spend differ by risk tier?")

# Query input
question = st.text_input(
    "Your question",
    placeholder="e.g. What is driving churn in our prepaid base?",
    key="query_input"
)

# Session state — persists answer across reruns
if "last_answer" not in st.session_state:
    st.session_state.last_answer = ""
if "last_question" not in st.session_state:
    st.session_state.last_question = ""

# Ask button
if st.button("Ask ↗", type="primary") and question.strip():
    with st.spinner("Analysing your data..."):
        answer = ask_claude(question, system_prompt, api_key)
        st.session_state.last_answer   = answer
        st.session_state.last_question = question

# Display answer if one exists
if st.session_state.last_answer:
    st.markdown(f"**Q: {st.session_state.last_question}**")
    st.markdown(st.session_state.last_answer)
    st.caption("Powered by Claude AI · answers based on your data summary")