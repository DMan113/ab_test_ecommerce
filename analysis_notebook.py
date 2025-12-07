# %% [markdown]
# # 🔬 A/B Test Analysis Dashboard - Enhanced Version
#
# **Experiment:** Recommendation Engine v2
# **Analysis Date:** December 2025
# **Analyst:** Dmytro Rybentsev

# %% [markdown]
# ## 📦 Setup and Imports

# %%
import pandas as pd
import numpy as np
from IPython.display import display, HTML, Markdown, Image
from pathlib import Path
import warnings

warnings.filterwarnings('ignore')

from src.ab_test import ABTestAnalyzer
from src.cohort_builder import CohortAnalyzer
from src.visualizations import ABTestVisualizer

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', 100)
pd.set_option('display.float_format', '{:.2f}'.format)

print("✅ All modules imported successfully!")

# %% [markdown]
# ## 🎨 Enhanced CSS Styling

# %%
css = """
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;600;700&display=swap');

    body {
        font-family: 'Inter', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    }

    .analysis-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 30px;
        border-radius: 15px;
        margin: 30px 0;
        text-align: center;
        font-size: 32px;
        font-weight: 700;
        box-shadow: 0 10px 30px rgba(102, 126, 234, 0.3);
        letter-spacing: -0.5px;
    }

    .section-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px 20px;
        border-radius: 10px;
        margin: 25px 0 15px 0;
        font-size: 20px;
        font-weight: 600;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.2);
    }

    .metric-grid {
        display: grid;
        grid-template-columns: repeat(auto-fit, minmax(250px, 1fr));
        gap: 20px;
        margin: 20px 0;
    }

    .metric-card {
        background: linear-gradient(135deg, #f5f7fa 0%, #c3cfe2 100%);
        border-radius: 12px;
        padding: 25px;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
        transition: transform 0.3s ease, box-shadow 0.3s ease;
    }

    .metric-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 8px 25px rgba(0, 0, 0, 0.15);
    }

    .metric-value {
        font-size: 36px;
        font-weight: 700;
        color: #2c3e50;
        margin: 10px 0;
    }

    .metric-label {
        font-size: 13px;
        color: #7f8c8d;
        text-transform: uppercase;
        letter-spacing: 1px;
        font-weight: 600;
    }

    .metric-change {
        font-size: 16px;
        font-weight: 600;
        margin-top: 8px;
    }

    .positive { color: #27ae60; }
    .negative { color: #e74c3c; }
    .neutral { color: #95a5a6; }

    .insight-box {
        background: #fff;
        border-left: 5px solid #3498db;
        padding: 20px;
        margin: 15px 0;
        border-radius: 8px;
        box-shadow: 0 2px 10px rgba(0, 0, 0, 0.05);
    }

    .insight-box.success {
        border-left-color: #27ae60;
        background: linear-gradient(to right, #e8f5e9 0%, #ffffff 100%);
    }

    .insight-box.warning {
        border-left-color: #f39c12;
        background: linear-gradient(to right, #fff3e0 0%, #ffffff 100%);
    }

    .insight-box.danger {
        border-left-color: #e74c3c;
        background: linear-gradient(to right, #ffebee 0%, #ffffff 100%);
    }

    .insight-title {
        font-size: 18px;
        font-weight: 600;
        margin-bottom: 10px;
        color: #2c3e50;
    }

    .insight-text {
        font-size: 15px;
        line-height: 1.6;
        color: #34495e;
    }

    .stats-table {
        width: 100%;
        border-collapse: separate;
        border-spacing: 0;
        margin: 20px 0;
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.1);
    }

    .stats-table thead th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        text-align: left;
        font-weight: 600;
        font-size: 14px;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }

    .stats-table tbody td {
        padding: 15px;
        border-bottom: 1px solid #ecf0f1;
        font-size: 14px;
    }

    .stats-table tbody tr {
        background: white;
        transition: background 0.2s ease;
    }

    .stats-table tbody tr:hover {
        background: #f8f9fa;
    }

    .decision-banner {
        padding: 30px;
        border-radius: 15px;
        text-align: center;
        font-size: 24px;
        font-weight: 700;
        margin: 30px 0;
        box-shadow: 0 10px 30px rgba(0, 0, 0, 0.15);
    }

    .decision-banner.recommend {
        background: linear-gradient(135deg, #4caf50 0%, #8bc34a 100%);
        color: white;
    }

    .decision-banner.reject {
        background: linear-gradient(135deg, #f44336 0%, #e91e63 100%);
        color: white;
    }

    .decision-banner.inconclusive {
        background: linear-gradient(135deg, #ff9800 0%, #ffc107 100%);
        color: white;
    }

    .key-finding {
        background: #fff;
        border: 2px solid #667eea;
        border-radius: 10px;
        padding: 20px;
        margin: 15px 0;
    }

    .key-finding-title {
        font-size: 16px;
        font-weight: 600;
        color: #667eea;
        margin-bottom: 10px;
    }

    .badge {
        display: inline-block;
        padding: 5px 12px;
        border-radius: 20px;
        font-size: 12px;
        font-weight: 600;
        margin: 5px;
    }

    .badge.success {
        background: #d4edda;
        color: #155724;
    }

    .badge.danger {
        background: #f8d7da;
        color: #721c24;
    }

    .badge.warning {
        background: #fff3cd;
        color: #856404;
    }

    .badge.info {
        background: #d1ecf1;
        color: #0c5460;
    }
</style>
"""

display(HTML(css))


# %% [markdown]
# ## 🛠️ Enhanced Helper Functions

# %%
def display_header(text, subtitle=""):
    """Display beautiful header with optional subtitle"""
    subtitle_html = f"<div style='font-size: 16px; margin-top: 10px; opacity: 0.9;'>{subtitle}</div>" if subtitle else ""
    html = f'''
    <div class="analysis-header">
        {text}
        {subtitle_html}
    </div>
    '''
    display(HTML(html))


def display_section_header(text):
    """Display section header"""
    html = f'<div class="section-header">{text}</div>'
    display(HTML(html))


def display_metrics_grid(metrics_list):
    """
    Display metrics in a grid layout
    metrics_list: list of dicts with keys: label, value, change, format_type
    """
    cards_html = ""
    for metric in metrics_list:
        label = metric.get('label', '')
        value = metric.get('value', 0)
        change = metric.get('change', None)
        format_type = metric.get('format_type', 'number')

        # Format value
        if format_type == 'percent':
            formatted_value = f"{value:.2f}%"
        elif format_type == 'currency':
            formatted_value = f"${value:,.2f}"
        elif format_type == 'number':
            formatted_value = f"{value:,.0f}"
        else:
            formatted_value = str(value)

        # Change indicator
        change_html = ""
        if change is not None:
            change_class = "positive" if change > 0 else "negative" if change < 0 else "neutral"
            change_symbol = "▲" if change > 0 else "▼" if change < 0 else "●"
            change_html = f'<div class="metric-change {change_class}">{change_symbol} {abs(change):.2f}%</div>'

        cards_html += f'''
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{formatted_value}</div>
            {change_html}
        </div>
        '''

    html = f'<div class="metric-grid">{cards_html}</div>'
    display(HTML(html))


def display_insight(title, text, insight_type='info'):
    """Display insight box"""
    html = f'''
    <div class="insight-box {insight_type}">
        <div class="insight-title">{title}</div>
        <div class="insight-text">{text}</div>
    </div>
    '''
    display(HTML(html))


def display_key_finding(title, metrics):
    """Display key finding with metrics"""
    metrics_html = ""
    for key, value in metrics.items():
        metrics_html += f"<strong>{key}:</strong> {value}<br>"

    html = f'''
    <div class="key-finding">
        <div class="key-finding-title">🔑 {title}</div>
        {metrics_html}
    </div>
    '''
    display(HTML(html))


def display_decision_banner(decision_type, text):
    """Display decision banner"""
    html = f'<div class="decision-banner {decision_type}">{text}</div>'
    display(HTML(html))


def create_comparison_badge(is_significant):
    """Create significance badge"""
    if is_significant:
        return '<span class="badge success">✓ Significant</span>'
    else:
        return '<span class="badge danger">✗ Not Significant</span>'


print("✅ Enhanced helper functions loaded!")

# %% [markdown]
# ## 🚀 Initialize Analysis

# %%
display_header(
    "🔬 A/B Test Analysis Dashboard",
    "Recommendation Engine v2 | Comprehensive Statistical Analysis"
)

print("🔄 Initializing analyzers...")
ab_analyzer = ABTestAnalyzer()
cohort_analyzer = CohortAnalyzer()
viz = ABTestVisualizer(save_dir="analysis_plots")

print("✅ Analyzers initialized successfully!")

# %% [markdown]
# ## 📊 Executive Summary

# %%
display_section_header("📊 Executive Summary")

# Load data
data = ab_analyzer.get_ab_test_data('recommendation_engine_v2')
cohort_data = cohort_analyzer.get_cohort_data('recommendation_engine_v2')

# Calculate key metrics
control_data = data[data['group_name'] == 'control']
treatment_data = data[data['group_name'] == 'treatment']

conv_test = ab_analyzer.z_test_proportions(control_data, treatment_data)
rev_test = ab_analyzer.t_test_revenue(control_data, treatment_data)

# Display overview metrics
display_metrics_grid([
    {
        'label': 'Total Users Analyzed',
        'value': len(data),
        'format_type': 'number'
    },
    {
        'label': 'Control Group',
        'value': len(control_data),
        'format_type': 'number'
    },
    {
        'label': 'Treatment Group',
        'value': len(treatment_data),
        'format_type': 'number'
    },
    {
        'label': 'Test Duration',
        'value': len(cohort_data),
        'format_type': 'number'
    }
])

# Key findings
display_key_finding(
    "Primary Outcome: Conversion Rate",
    {
        'Control': f"{conv_test['control_conversion_rate']:.2f}%",
        'Treatment': f"{conv_test['treatment_conversion_rate']:.2f}%",
        'Lift': f"{conv_test['lift_percentage']:.2f}%",
        'P-value': f"{conv_test['p_value']:.4f}",
        'Status': '✓ Significant' if conv_test['statistically_significant'] else '✗ Not Significant'
    }
)

display_key_finding(
    "Secondary Outcome: Revenue per User",
    {
        'Control': f"${rev_test['control_avg_revenue']:.2f}",
        'Treatment': f"${rev_test['treatment_avg_revenue']:.2f}",
        'Lift': f"{rev_test['lift_percentage']:.2f}%",
        'P-value': f"{rev_test['p_value']:.4f}",
        'Status': '✓ Significant' if rev_test['statistically_significant'] else '✗ Not Significant'
    }
)

# %% [markdown]
# ## 💰 Conversion Analysis (Deep Dive)

# %%
display_section_header("💰 Conversion Rate Analysis")

conversion_metrics = ab_analyzer.calculate_conversion_rate(data)

# Display metrics with lift
control_conv = conversion_metrics[conversion_metrics['group_name'] == 'control'].iloc[0]
treatment_conv = conversion_metrics[conversion_metrics['group_name'] == 'treatment'].iloc[0]

display_metrics_grid([
    {
        'label': 'Control Conversion Rate',
        'value': control_conv['conversion_rate'],
        'format_type': 'percent'
    },
    {
        'label': 'Treatment Conversion Rate',
        'value': treatment_conv['conversion_rate'],
        'format_type': 'percent',
        'change': conv_test['lift_percentage']
    },
    {
        'label': 'Absolute Difference',
        'value': conv_test['treatment_conversion_rate'] - conv_test['control_conversion_rate'],
        'format_type': 'percent'
    },
    {
        'label': 'Statistical Power',
        'value': (1 - conv_test['p_value']) * 100 if conv_test['p_value'] < 0.5 else 0,
        'format_type': 'percent'
    }
])

# Interpretation
if conv_test['statistically_significant']:
    if conv_test['lift_percentage'] > 0:
        display_insight(
            "✅ Positive Result",
            f"The treatment group shows a {conv_test['lift_percentage']:.2f}% improvement in conversion rate. "
            f"This result is statistically significant (p={conv_test['p_value']:.4f}), indicating that the "
            f"new recommendation engine is performing better than the control.",
            'success'
        )
    else:
        display_insight(
            "❌ Negative Result",
            f"The treatment group shows a {abs(conv_test['lift_percentage']):.2f}% decrease in conversion rate. "
            f"This result is statistically significant (p={conv_test['p_value']:.4f}), indicating that the "
            f"new recommendation engine is underperforming compared to the control.",
            'danger'
        )
else:
    display_insight(
        "⚠️ Inconclusive Result",
        f"The observed difference of {conv_test['lift_percentage']:.2f}% in conversion rate is not statistically "
        f"significant (p={conv_test['p_value']:.4f}). This could be due to insufficient sample size, high variance, "
        f"or the true effect being smaller than detectable. Consider extending the test or analyzing segments.",
        'warning'
    )

# Visualization
fig_conv = viz.plot_conversion_rate(conversion_metrics)
fig_conv.show()

# %% [markdown]
# ## 💵 Revenue Analysis (Deep Dive)

# %%
display_section_header("💵 Revenue per User Analysis")

revenue_metrics = ab_analyzer.calculate_revenue_metrics(data)

# Display metrics
control_rev = revenue_metrics[revenue_metrics['group_name'] == 'control'].iloc[0]
treatment_rev = revenue_metrics[revenue_metrics['group_name'] == 'treatment'].iloc[0]

display_metrics_grid([
    {
        'label': 'Control Revenue/User',
        'value': control_rev['avg_revenue_per_user'],
        'format_type': 'currency'
    },
    {
        'label': 'Treatment Revenue/User',
        'value': treatment_rev['avg_revenue_per_user'],
        'format_type': 'currency',
        'change': rev_test['lift_percentage']
    },
    {
        'label': 'Total Revenue Impact',
        'value': (treatment_rev['avg_revenue_per_user'] - control_rev['avg_revenue_per_user']) * len(treatment_data),
        'format_type': 'currency'
    },
    {
        'label': 'Effect Size (Cohen\'s d)',
        'value': abs(rev_test.get('cohens_d', 0)),
        'format_type': 'number'
    }
])

# Interpretation
if rev_test['statistically_significant']:
    revenue_impact = (treatment_rev['avg_revenue_per_user'] - control_rev['avg_revenue_per_user']) * len(data)

    if rev_test['lift_percentage'] > 0:
        display_insight(
            "💰 Significant Revenue Increase",
            f"The treatment group generates {rev_test['lift_percentage']:.2f}% more revenue per user "
            f"(${treatment_rev['avg_revenue_per_user']:.2f} vs ${control_rev['avg_revenue_per_user']:.2f}). "
            f"If rolled out to all users, this could result in approximately ${revenue_impact:,.2f} in additional revenue. "
            f"This difference is statistically significant (p={rev_test['p_value']:.4f}).",
            'success'
        )
    else:
        display_insight(
            "⚠️ Significant Revenue Decrease",
            f"The treatment group generates {abs(rev_test['lift_percentage']):.2f}% less revenue per user. "
            f"This represents a potential loss of ${abs(revenue_impact):,.2f}. "
            f"This difference is statistically significant (p={rev_test['p_value']:.4f}).",
            'danger'
        )
else:
    display_insight(
        "📊 No Significant Revenue Difference",
        f"While there's a {rev_test['lift_percentage']:.2f}% difference in revenue per user, "
        f"it's not statistically significant (p={rev_test['p_value']:.4f}). "
        f"The true effect might be smaller than observed or masked by high variance.",
        'warning'
    )

# Visualization
fig_rev = viz.plot_revenue_comparison(revenue_metrics)
fig_rev.show()

# %% [markdown]
# ## 👥 Cohort & Retention Analysis

# %%
display_section_header("👥 Cohort Retention Analysis")

retention = cohort_analyzer.calculate_cohort_retention(cohort_data)
retention = retention.dropna(subset=['retention_rate'])

# Retention summary
retention_summary = retention.groupby('group_name').agg({
    'retention_rate': ['mean', 'std', 'min', 'max']
}).round(2)

display(Markdown("### 📊 Retention Statistics"))
display(retention_summary)

# Cohort heatmaps
pivot_tables = cohort_analyzer.create_cohort_pivot_table(retention)

for group_name, pivot in pivot_tables.items():
    display(Markdown(f"### 🗺️ {group_name.capitalize()} Group - Cohort Retention Heatmap"))
    fig = viz.plot_cohort_heatmap(pivot, group_name)
    fig.show()

# Retention curves
display(Markdown("### 📈 Retention Curves Comparison"))
fig_retention = viz.plot_retention_curves(retention)
fig_retention.show()

# Insights
avg_retention_control = retention[retention['group_name'] == 'control']['retention_rate'].mean()
avg_retention_treatment = retention[retention['group_name'] == 'treatment']['retention_rate'].mean()
retention_diff = ((avg_retention_treatment - avg_retention_control) / avg_retention_control * 100)

if abs(retention_diff) > 5:
    insight_type = 'success' if retention_diff > 0 else 'warning'
    display_insight(
        f"{'📈' if retention_diff > 0 else '📉'} Retention Difference Detected",
        f"Treatment group shows {abs(retention_diff):.1f}% {'higher' if retention_diff > 0 else 'lower'} "
        f"average retention compared to control ({avg_retention_treatment:.1f}% vs {avg_retention_control:.1f}%).",
        insight_type
    )

# %% [markdown]
# ## 💰 Revenue Cohort Analysis

# %%
display_section_header("💰 Cohort Revenue Analysis")

revenue_cohort = cohort_analyzer.calculate_cohort_revenue(cohort_data)

# Revenue summary
revenue_summary = revenue_cohort.groupby('group_name')['arpu'].agg(['mean', 'std', 'min', 'max']).round(2)
display(Markdown("### 📊 ARPU (Average Revenue Per User) Statistics"))
display(revenue_summary)

# Cumulative revenue chart
display(Markdown("### 📈 Cumulative Revenue Growth"))
fig_cum_rev = viz.plot_cumulative_revenue(revenue_cohort)
fig_cum_rev.show()

# %% [markdown]
# ## 🎯 Final Decision & Recommendations

# %%
display_section_header("🎯 Final Decision & Recommendations")

# Decision logic
if conv_test['statistically_significant'] and conv_test['lift_percentage'] > 2:
    display_decision_banner('recommend', '✅ RECOMMEND ROLLOUT')
    decision_text = "Strong recommendation to proceed with full rollout of the new recommendation engine."
    decision_type = 'success'
elif conv_test['statistically_significant'] and conv_test['lift_percentage'] < -2:
    display_decision_banner('reject', '❌ DO NOT ROLLOUT')
    decision_text = "The new recommendation engine underperforms. Do not proceed with rollout."
    decision_type = 'danger'
else:
    display_decision_banner('inconclusive', '⚠️ INCONCLUSIVE - FURTHER TESTING RECOMMENDED')
    decision_text = "Results are inconclusive. Consider extended testing, segment analysis, or larger sample size."
    decision_type = 'warning'

display_insight("📋 Final Decision", decision_text, decision_type)

# Detailed recommendations
display(Markdown("### 📝 Detailed Recommendations"))

recommendations = []

# Based on conversion
if not conv_test['statistically_significant']:
    recommendations.append("🔍 **Extend test duration** to achieve statistical significance in conversion rate")
    recommendations.append("📊 **Analyze user segments** (device type, country, user tenure) for hidden effects")

# Based on revenue
if rev_test['statistically_significant'] and rev_test['lift_percentage'] > 5:
    recommendations.append("💰 **Strong revenue lift detected** - prioritize rollout to maximize revenue")
elif rev_test['statistically_significant'] and rev_test['lift_percentage'] < 0:
    recommendations.append("⚠️ **Revenue concerns** - investigate why revenue decreased despite test")

# Sample size
sample_size_needed = ab_analyzer.calculate_sample_size(
    baseline_rate=conv_test['control_conversion_rate'] / 100,
    mde=0.02
)
if len(control_data) < sample_size_needed:
    recommendations.append(
        f"📏 **Insufficient sample size** - need {sample_size_needed:,} users per group, currently have {len(control_data):,}")

# Retention
if abs(retention_diff) > 10:
    recommendations.append(
        f"👥 **Significant retention impact** - {retention_diff:+.1f}% change warrants deeper investigation")

for i, rec in enumerate(recommendations, 1):
    display(Markdown(f"{i}. {rec}"))

# Summary table
display(Markdown("### 📊 Test Summary"))

summary_html = f'''
<table class="stats-table">
    <thead>
        <tr>
            <th>Metric</th>
            <th>Control</th>
            <th>Treatment</th>
            <th>Lift</th>
            <th>P-value</th>
            <th>Significant</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td><strong>Conversion Rate</strong></td>
            <td>{conv_test['control_conversion_rate']:.2f}%</td>
            <td>{conv_test['treatment_conversion_rate']:.2f}%</td>
            <td class="{'positive' if conv_test['lift_percentage'] > 0 else 'negative'}">{conv_test['lift_percentage']:+.2f}%</td>
            <td>{conv_test['p_value']:.4f}</td>
            <td>{create_comparison_badge(conv_test['statistically_significant'])}</td>
        </tr>
        <tr>
            <td><strong>Revenue per User</strong></td>
            <td>${rev_test['control_avg_revenue']:.2f}</td>
            <td>${rev_test['treatment_avg_revenue']:.2f}</td>
            <td class="{'positive' if rev_test['lift_percentage'] > 0 else 'negative'}">{rev_test['lift_percentage']:+.2f}%</td>
            <td>{rev_test['p_value']:.4f}</td>
            <td>{create_comparison_badge(rev_test['statistically_significant'])}</td>
        </tr>
        <tr>
            <td><strong>Avg Retention</strong></td>
            <td>{avg_retention_control:.1f}%</td>
            <td>{avg_retention_treatment:.1f}%</td>
            <td class="{'positive' if retention_diff > 0 else 'negative'}">{retention_diff:+.1f}%</td>
            <td>N/A</td>
            <td><span class="badge info">Descriptive</span></td>
        </tr>
    </tbody>
</table>
'''

display(HTML(summary_html))

# %% [markdown]
# ---
# ### ✨ Analysis Complete!
#
# **Next Steps:**
# 1. Review all visualizations saved in `analysis_plots/`
# 2. Share findings with stakeholders
# 3. Make go/no-go decision based on business context
# 4. Document learnings for future experiments

# %%
print("\n" + "=" * 70)
print("✅ COMPREHENSIVE ANALYSIS COMPLETE!")
print("=" * 70)
print(f"\n📁 Visualizations: analysis_plots/")
print(f"📊 Users analyzed: {len(data):,}")
print(f"📈 Conversion lift: {conv_test['lift_percentage']:+.2f}% (p={conv_test['p_value']:.4f})")
print(f"💰 Revenue lift: {rev_test['lift_percentage']:+.2f}% (p={rev_test['p_value']:.4f})")
print(f"🎯 Decision: {decision_text}")
print("\nThank you for using the A/B Test Analysis Dashboard! 🚀")
print("=" * 70)