# analysis_extensions.py
"""
Extended analysis helpers for the A/B testing project.

Usage:
    from analysis_extensions import run_full_extended_report
    report = run_full_extended_report(ab, coh, viz, data, conversion, revenue, retention, cohort_df)
"""

from typing import Dict, Any, Optional
import os
import json
import subprocess
from datetime import datetime
import pandas as pd
import numpy as np

# Output folders (relative to current working dir)
REPORT_DIR = "analysis_report"
PLOTS_DIR = os.path.join(REPORT_DIR, "plots")
os.makedirs(PLOTS_DIR, exist_ok=True)


# ---------- Utilities ----------
def _fmt_pct(x: float, ndigits: int = 2) -> str:
    if pd.isna(x):
        return "N/A"
    return f"{x*100:.{ndigits}f}%"


def _save_fig(fig, name: str, fmt: str = "png") -> Optional[str]:
    """
    Save a Plotly figure. Try image export first (kaleido); fall back to HTML.
    Returns path or None on failure.
    """
    path = os.path.join(PLOTS_DIR, f"{name}.{fmt}")
    try:
        # prefer static export (requires kaleido)
        fig.write_image(path, scale=2)
        return path
    except Exception:
        # fallback to html
        html_path = os.path.join(PLOTS_DIR, f"{name}.html")
        try:
            fig.write_html(html_path)
            return html_path
        except Exception:
            return None


# ---------- 1) Summary ----------
def generate_summary(data: pd.DataFrame) -> Dict[str, Any]:
    """
    Create a compact summary of the A/B dataset.
    Expects columns: user_id, group_name, converted, transaction_count, total_revenue
    """
    summary: Dict[str, Any] = {}
    summary['total_rows'] = len(data)

    if 'group_name' not in data.columns or 'user_id' not in data.columns:
        summary['by_group'] = None
        summary['balance_ratio'] = None
        summary['overall_conversion_rate'] = None
        return summary

    by_group = (
        data.groupby('group_name')
        .agg(users=('user_id', 'count'),
             conversions=('converted', 'sum'),
             transaction_count=('transaction_count', 'sum'),
             total_revenue=('total_revenue', 'sum'))
        .reset_index()
    )
    # conversion rate 0..1
    by_group['conversion_rate'] = (by_group['conversions'] / by_group['users']).fillna(0)
    summary['by_group'] = by_group

    # balance measurement (relative diff)
    if set(['control', 'treatment']).issubset(set(by_group['group_name'])):
        control_users = int(by_group.loc[by_group['group_name'] == 'control', 'users'].iloc[0])
        treatment_users = int(by_group.loc[by_group['group_name'] == 'treatment', 'users'].iloc[0])
        summary['balance_ratio'] = abs(control_users - treatment_users) / max(control_users, treatment_users)
    else:
        summary['balance_ratio'] = None

    # overall
    total_users = by_group['users'].sum()
    total_conv = by_group['conversions'].sum()
    summary['overall_conversion_rate'] = (total_conv / total_users) if total_users > 0 else 0

    return summary


# ---------- 2) Interpretation ----------
def interpret_ab_test(ab, data: pd.DataFrame, conversion_df: pd.DataFrame, revenue_df: pd.DataFrame) -> Dict[str, Any]:
    """
    Run the statistical tests implemented in ABTestAnalyzer and produce a readable summary.
    Expects ab to have methods: z_test_proportions(control_df, treatment_df) and t_test_revenue(control_df, treatment_df)
    """
    control = data[data['group_name'] == 'control']
    treatment = data[data['group_name'] == 'treatment']

    conv_test = ab.z_test_proportions(control, treatment)
    rev_test = ab.t_test_revenue(control, treatment)

    conv_out = {k: v for k, v in conv_test.items()}
    rev_out = {k: v for k, v in rev_test.items()}

    plain_lines = []
    plain_lines.append("# A/B Test Interpretation")
    plain_lines.append("## Conversion")
    # guard keys presence
    plain_lines.append(f"Control conversion rate: {conv_out.get('control_conversion_rate', 'N/A')}")
    plain_lines.append(f"Treatment conversion rate: {conv_out.get('treatment_conversion_rate', 'N/A')}")
    plain_lines.append(f"Lift (treatment vs control): {conv_out.get('lift_percentage', 'N/A')}")
    plain_lines.append(f"Z-statistic: {conv_out.get('z_statistic', 'N/A')}")
    plain_lines.append(f"p-value: {conv_out.get('p_value', 'N/A')}")
    plain_lines.append(f"Statistically significant (α=0.05): {conv_out.get('statistically_significant', 'N/A')}")
    plain_lines.append("")
    plain_lines.append("## Revenue")
    plain_lines.append(f"Control avg revenue/user: {rev_out.get('control_avg_revenue', 'N/A')}")
    plain_lines.append(f"Treatment avg revenue/user: {rev_out.get('treatment_avg_revenue', 'N/A')}")
    plain_lines.append(f"Revenue lift: {rev_out.get('lift_percentage', 'N/A')}")
    plain_lines.append(f"T-statistic: {rev_out.get('t_statistic', 'N/A')}")
    plain_lines.append(f"p-value: {rev_out.get('p_value', 'N/A')}")
    plain_lines.append(f"Statistically significant: {rev_out.get('statistically_significant', 'N/A')}")
    plain_lines.append("")

    if conv_out.get('statistically_significant'):
        plain_lines.append("Recommendation: conversion uplift is statistically significant — consider rollout.")
    else:
        plain_lines.append("Recommendation: no significant uplift detected — investigate segments, data quality, or increase sample size.")

    return {
        'conversion_test': conv_out,
        'revenue_test': rev_out,
        'plain_text': "\n".join(plain_lines)
    }


# ---------- 3) Data quality ----------
def data_quality_checks(data: pd.DataFrame) -> Dict[str, Any]:
    checks: Dict[str, Any] = {}
    checks['rows'] = len(data)
    checks['nulls'] = data.isna().sum().to_dict()

    if 'total_revenue' in data.columns:
        s = data['total_revenue'].fillna(0)
        checks['revenue_mean'] = float(s.mean())
        checks['revenue_median'] = float(s.median())
        checks['revenue_q99'] = float(s.quantile(0.99))
    return checks


# ---------- 4) Segmentation ----------
def segmentation_analysis(data: pd.DataFrame, by: str = 'device_type', top_n_countries: int = 10) -> Dict[str, pd.DataFrame]:
    """
    Produce segment-level metrics. Returns dictionary of DataFrames or {'error': msg} values.
    """
    out = {}
    if by is None:
        return out

    if by == 'country':
        if 'country' not in data.columns:
            return {'error': 'country not in data'}
        top = data['country'].value_counts().head(top_n_countries).index.tolist()
        df = data[data['country'].isin(top)].copy()
        grp = (
            df.groupby(['group_name', 'country'])
            .agg(users=('user_id', 'count'), conversions=('converted', 'sum'))
            .reset_index()
        )
        grp['conversion_rate'] = grp['conversions'] / grp['users']
        out['by_country_top'] = grp
    else:
        if by not in data.columns:
            return {'error': f'{by} not in data'}
        df = data.copy()
        grp = (
            df.groupby(['group_name', by])
            .agg(users=('user_id', 'count'), conversions=('converted', 'sum'))
            .reset_index()
        )
        grp['conversion_rate'] = grp['conversions'] / grp['users']
        out[f'by_{by}'] = grp
    return out


# ---------- 5) Export plots ----------
def save_all_plots(viz, conversion_df: pd.DataFrame, revenue_df: pd.DataFrame, retention_df: pd.DataFrame,
                   cohort_df: pd.DataFrame, ab, data: pd.DataFrame, coh) -> Dict[str, Any]:
    """
    Save main visualizations. coh must be a CohortAnalyzer instance.
    Returns dict with saved file paths or error strings.
    """
    saved: Dict[str, Any] = {}

    # conversion
    try:
        fig = viz.plot_conversion_rate(conversion_df)
        saved['conversion'] = _save_fig(fig, 'conversion_rate')
    except Exception as e:
        saved['conversion_error'] = str(e)

    # revenue
    try:
        fig = viz.plot_revenue_comparison(revenue_df)
        saved['revenue'] = _save_fig(fig, 'revenue_comparison')
    except Exception as e:
        saved['revenue_error'] = str(e)

    # cohort heatmaps
    try:
        pivots = coh.create_cohort_pivot_table(retention_df)
        saved['cohort_heatmaps'] = {}
        for g, p in pivots.items():
            fig = viz.plot_cohort_heatmap(p, g)
            saved['cohort_heatmaps'][g] = _save_fig(fig, f'cohort_heatmap_{g}')
    except Exception as e:
        saved['cohort_heatmaps_error'] = str(e)

    # retention curves
    try:
        fig = viz.plot_retention_curves(retention_df)
        saved['retention_curves'] = _save_fig(fig, 'retention_curves')
    except Exception as e:
        saved['retention_curves_error'] = str(e)

    # cumulative revenue by cohort
    try:
        rev = coh.calculate_cohort_revenue(cohort_df)
        fig = viz.plot_cumulative_revenue(rev)
        saved['cumulative_revenue'] = _save_fig(fig, 'cumulative_revenue')
    except Exception as e:
        saved['cumulative_revenue_error'] = str(e)

    # funnel (optional)
    try:
        funnel_df = pd.read_sql("SELECT * FROM conversion_funnel", ab.engine)
        fig = viz.plot_funnel_analysis(funnel_df)
        saved['funnel'] = _save_fig(fig, 'funnel')
    except Exception as e:
        saved['funnel_error'] = str(e)

    # sequential testing
    try:
        seq = ab.sequential_testing(data)
        fig = viz.plot_sequential_testing(seq)
        saved['sequential'] = _save_fig(fig, 'sequential_testing')
    except Exception as e:
        saved['sequential_error'] = str(e)

    return saved


# ---------- 6) Report generation ----------
def generate_report_md(summary: Dict[str, Any], interp: Dict[str, Any], checks: Dict[str, Any],
                       segment_results: Dict[str, pd.DataFrame], saved_plots: Dict[str, Any],
                       filename: Optional[str] = None) -> str:
    if filename is None:
        filename = os.path.join(REPORT_DIR, f"ab_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md")

    lines = []
    lines.append(f"# A/B Test Report — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("\n## 1. Summary of data\n")
    lines.append(f"- Total rows: **{summary.get('total_rows', 0):,}**")

    by = summary.get('by_group')
    if by is not None and isinstance(by, pd.DataFrame):
        for _, r in by.iterrows():
            lines.append(f"- **{r['group_name'].capitalize()}**: users={int(r['users']):,}, conversions={int(r['conversions'])}, conversion_rate={r['conversion_rate']:.4f}")

    bal = summary.get('balance_ratio')
    if bal is not None:
        lines.append(f"\n- Balance deviation: {bal*100:.2f}%\n")

    lines.append("## 2. Data quality checks\n")
    lines.append("```")
    lines.append(json.dumps(checks, indent=2, default=str))
    lines.append("```")

    lines.append("## 3. Interpretation\n")
    lines.append(interp.get('plain_text', ''))

    lines.append("## 4. Segment analysis (sample)\n")
    # robustly handle segment_results values (DataFrame or error str)
    if isinstance(segment_results, dict):
        for k, df in segment_results.items():
            lines.append(f"### {k}")
            if isinstance(df, pd.DataFrame):
                try:
                    lines.append(df.head(10).to_markdown(index=False))
                except Exception:
                    lines.append(str(df.head(10)))
            else:
                # df is probably an error message
                lines.append(f"⚠️ {df}")

    lines.append("## 5. Saved plots\n")
    for k, v in (saved_plots or {}).items():
        if isinstance(v, dict):
            lines.append(f"### {k}")
            for subk, subv in v.items():
                lines.append(f"- {subk}: {subv}")
        else:
            lines.append(f"- {k}: {v}")

    lines.append("## 6. Recommendations\n")
    lines.append("- No statistically significant conversion uplift detected.\n")
    lines.append("- Investigate segments where lift might be concentrated (device, country).\n")
    lines.append("- Verify data generation; check for synthetic data artifacts in retention.\n")
    lines.append("- Consider increasing sample size or experiment duration.\n")

    # write file
    os.makedirs(REPORT_DIR, exist_ok=True)
    with open(filename, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    return filename


# ---------- 7) Convert MD -> PDF (best-effort) ----------
def convert_md_to_pdf(md_path: str, pdf_path: Optional[str] = None) -> Optional[str]:
    if pdf_path is None:
        pdf_path = md_path.rsplit('.', 1)[0] + '.pdf'
    # try pandoc
    try:
        subprocess.run(['pandoc', md_path, '-o', pdf_path], check=True)
        return pdf_path
    except Exception:
        pass

    # try nbconvert fallback
    try:
        tmp_nb = md_path.rsplit('.', 1)[0] + '_tmp.ipynb'
        nb = {
            'cells': [
                {
                    'cell_type': 'markdown',
                    'metadata': {},
                    'source': [open(md_path, 'r', encoding='utf-8').read()]
                }
            ],
            'metadata': {'kernelspec': {'display_name': 'Python 3', 'language': 'python', 'name': 'python3'}},
            'nbformat': 4,
            'nbformat_minor': 5
        }
        with open(tmp_nb, 'w', encoding='utf-8') as f:
            json.dump(nb, f)
        subprocess.run(['jupyter', 'nbconvert', '--to', 'pdf', tmp_nb, '--output', pdf_path.rsplit('.', 1)[0]], check=True)
        return pdf_path
    except Exception:
        return None


# ---------- 8) Orchestrator ----------
def run_full_extended_report(ab, coh, viz, data: pd.DataFrame, conversion_df: pd.DataFrame,
                             revenue_df: pd.DataFrame, retention_df: pd.DataFrame, cohort_df: pd.DataFrame,
                             seg_by: str = 'device_type') -> Dict[str, Any]:
    """
    Run full extended reporting pipeline.
    Returns dict with md_path, pdf_path, saved_plots, summary, interp, checks, segments
    """
    # 1. summary
    summary = generate_summary(data)
    # 2. interpretation
    interp = interpret_ab_test(ab, data, conversion_df, revenue_df)
    # 3. quality checks
    checks = data_quality_checks(data)
    # 4. segmentation
    segments = segmentation_analysis(data, by=seg_by)
    # 5. export plots (pass coh explicitly)
    saved = save_all_plots(viz, conversion_df, revenue_df, retention_df, cohort_df, ab, data, coh)
    # 6. generate md
    md_path = generate_report_md(summary, interp, checks, segments, saved)
    # 7. try pdf
    pdf_path = convert_md_to_pdf(md_path)

    return {
        'md_path': md_path,
        'pdf_path': pdf_path,
        'saved_plots': saved,
        'summary': summary,
        'interp': interp,
        'checks': checks,
        'segments': segments
    }


