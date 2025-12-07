# analysis_extensions.py
"""
Extended, project-adapted report generator for A/B testing.

Compatibility assumptions (based on project):
- `ab` is ABTestAnalyzer with:
    - z_test_proportions(control_df, treatment_df) -> dict (see keys below)
    - t_test_revenue(control_df, treatment_df) -> dict
    - sequential_testing(data) -> DataFrame
    - engine -> SQLAlchemy engine
- `coh` is CohortAnalyzer with:
    - create_cohort_pivot_table(retention_df) -> dict(group_name -> pivot_df)
    - calculate_cohort_revenue(cohort_df) -> DataFrame
- `viz` is ABTestVisualizer returning Plotly figures for the named methods.

This module:
- generates summary metrics
- runs tests and interprets results
- saves visualizations (png via kaleido; html fallback)
- writes a markdown report and optionally converts to pdf
"""

from __future__ import annotations
import os
import json
import subprocess
from datetime import datetime
from dataclasses import dataclass, asdict
from typing import Dict, Any, Optional, List, Tuple
from pathlib import Path

import pandas as pd
import numpy as np

# ---------- Config / Data models ----------

@dataclass
class ReportConfig:
    report_dir: str = "analysis_report"
    plots_dir: str = "plots"
    include_pdf: bool = False       # Pandoc often missing; default off
    segment_by: str = "device_type"
    top_n_countries: int = 10
    alpha: float = 0.05

    @property
    def plots_path(self) -> Path:
        return Path(self.report_dir) / self.plots_dir

@dataclass
class SummaryMetrics:
    total_rows: int
    total_users: int
    control_users: int
    treatment_users: int
    balance_ratio: float
    overall_conversion_rate: float
    data_quality_score: float

    def to_dict(self):
        return asdict(self)

@dataclass
class TestResults:
    control_metric: float
    treatment_metric: float
    lift_percentage: float
    absolute_difference: float
    test_statistic: float
    p_value: float
    is_significant: bool
    confidence_interval: Tuple[float, float]
    effect_size: Optional[float] = None

    def to_dict(self):
        d = asdict(self)
        # ensure numeric types are python-native
        for k,v in d.items():
            if isinstance(v, (np.floating, np.integer)):
                d[k] = float(v)
        return d

# ---------- Utilities ----------

def _ensure_dir(path: Path):
    path.mkdir(parents=True, exist_ok=True)

def _safe_write_image(fig, path: Path):
    # Attempt PNG via kaleido; fallback to HTML
    try:
        import kaleido  # ensure available
        fig.write_image(str(path), scale=2)
        return str(path)
    except Exception:
        # fallback to html
        html_path = path.with_suffix('.html')
        try:
            fig.write_html(str(html_path))
            return str(html_path)
        except Exception as e:
            return None

def _fmt_pct(v):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "N/A"
    try:
        return f"{v*100:.2f}%"
    except Exception:
        # if already percent (rare), try direct format
        try:
            return f"{float(v):.2f}%"
        except Exception:
            return str(v)

# ---------- Main class ----------

class ExtendedReportGenerator:
    def __init__(self, ab, coh, viz, config: Optional[ReportConfig] = None):
        self.ab = ab
        self.coh = coh
        self.viz = viz
        self.config = config or ReportConfig()
        # prepare directories
        _ensure_dir(Path(self.config.report_dir))
        _ensure_dir(self.config.plots_path)
        # single timestamp for consistent filenames
        self._ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    # ---------------- Summary ----------------
    def generate_summary(self, data: pd.DataFrame) -> SummaryMetrics:
        if data is None or data.empty:
            return SummaryMetrics(0, 0, 0, 0, 0.0, 0.0, 0.0)

        total_rows = len(data)
        total_users = int(data['user_id'].nunique()) if 'user_id' in data.columns else total_rows

        # group counts
        if 'group_name' in data.columns:
            grp = data.groupby('group_name').agg(user_count=('user_id', 'count'),
                                                 conversions=('converted', 'sum' if 'converted' in data.columns else 'count'))
            control_users = int(grp.loc['control','user_count']) if 'control' in grp.index else 0
            treatment_users = int(grp.loc['treatment','user_count']) if 'treatment' in grp.index else 0
        else:
            control_users = 0
            treatment_users = 0

        balance_ratio = 0.0
        if control_users and treatment_users:
            balance_ratio = abs(control_users - treatment_users) / max(control_users, treatment_users)

        overall_conversion_rate = (data['converted'].sum() / total_users) if ('converted' in data.columns and total_users>0) else 0.0

        data_quality_score = self._calc_data_quality(data, balance_ratio)

        return SummaryMetrics(
            total_rows=total_rows,
            total_users=total_users,
            control_users=control_users,
            treatment_users=treatment_users,
            balance_ratio=balance_ratio,
            overall_conversion_rate=overall_conversion_rate,
            data_quality_score=data_quality_score
        )

    def _calc_data_quality(self, data: pd.DataFrame, balance_ratio: float) -> float:
        score = 100.0
        score -= min(balance_ratio * 100, 20)
        if 'total_revenue' in data.columns:
            missing_pct = data['total_revenue'].isna().sum() / max(len(data),1)
            score -= missing_pct * 30
        if len(data) < 1000:
            score -= 20
        elif len(data) < 10000:
            score -= 10
        return max(0.0, score)

    # ---------------- Interpretation ----------------
    def interpret_results(self, data: pd.DataFrame, conversion_df: pd.DataFrame, revenue_df: pd.DataFrame) -> Dict[str, Any]:
        # ensure groups exist
        control = data[data['group_name']=='control'] if 'group_name' in data.columns else data
        treatment = data[data['group_name']=='treatment'] if 'group_name' in data.columns else data

        conv_raw = self.ab.z_test_proportions(control, treatment)
        rev_raw = self.ab.t_test_revenue(control, treatment)

        conv = TestResults(
            control_metric = conv_raw.get('control_conversion_rate', 0.0),
            treatment_metric = conv_raw.get('treatment_conversion_rate', 0.0),
            lift_percentage = conv_raw.get('lift_percentage', 0.0),
            absolute_difference = conv_raw.get('treatment_conversion_rate', 0.0) - conv_raw.get('control_conversion_rate', 0.0),
            test_statistic = conv_raw.get('z_statistic', 0.0),
            p_value = conv_raw.get('p_value', 1.0),
            is_significant = conv_raw.get('statistically_significant', False),
            confidence_interval = conv_raw.get('confidence_interval_95', (0.0,0.0)),
            effect_size = conv_raw.get('cohens_d', None)
        )

        rev = TestResults(
            control_metric = rev_raw.get('control_avg_revenue', 0.0),
            treatment_metric = rev_raw.get('treatment_avg_revenue', 0.0),
            lift_percentage = rev_raw.get('lift_percentage', 0.0),
            absolute_difference = rev_raw.get('treatment_avg_revenue', 0.0) - rev_raw.get('control_avg_revenue', 0.0),
            test_statistic = rev_raw.get('t_statistic', 0.0),
            p_value = rev_raw.get('p_value', 1.0),
            is_significant = rev_raw.get('statistically_significant', False),
            confidence_interval = (0.0, 0.0),
            effect_size = rev_raw.get('cohens_d', None)
        )

        recommendations = self._build_recommendations(conv, rev, data)
        return {'conversion_test': conv, 'revenue_test': rev, 'recommendations': recommendations}

    def _build_recommendations(self, conv: TestResults, rev: TestResults, data: pd.DataFrame) -> List[str]:
        recs: List[str] = []
        n = len(data)
        if n < 1000:
            recs.append("⚠️ Small sample (<1000). Results may be unreliable.")
        if conv.is_significant:
            if conv.lift_percentage > 0:
                recs.append(f"✅ Conversion up {conv.lift_percentage:.2f}% — consider rollout.")
            else:
                recs.append(f"❌ Conversion down {abs(conv.lift_percentage):.2f}% — do not rollout.")
        else:
            recs.append(f"⚪ No significant conversion difference (p={conv.p_value:.4f}).")
        if rev.is_significant:
            recs.append(f"💰 Revenue per user change significant (lift {rev.lift_percentage:.2f}%).")
        else:
            recs.append(f"⚪ No significant revenue difference (p={rev.p_value:.4f}).")
        if (not conv.is_significant) and (not rev.is_significant):
            recs.append("Next steps: segment-level analysis, verify data, longer test or larger sample.")
        return recs

    # ---------------- Data quality ----------------
    def check_data_quality(self, data: pd.DataFrame) -> Dict[str, Any]:
        checks = {'total_rows': len(data), 'missing_values': {}, 'outliers': {}, 'distribution': {}}
        for col in data.columns:
            miss = int(data[col].isna().sum())
            if miss > 0:
                checks['missing_values'][col] = {'count': miss, 'percentage': round(miss / max(len(data),1) * 100, 2)}
        if 'total_revenue' in data.columns:
            rev = data['total_revenue'].dropna()
            if len(rev) > 0:
                q1, q3 = rev.quantile([0.25, 0.75])
                iqr = q3 - q1
                out = rev[(rev < q1 - 1.5*iqr) | (rev > q3 + 1.5*iqr)]
                checks['outliers']['revenue'] = {'count': int(len(out)), 'percentage': round(len(out)/len(rev)*100,2),
                                                'max_value': float(rev.max()), 'p99_value': float(rev.quantile(0.99))}
                checks['distribution']['revenue'] = {'mean': float(rev.mean()), 'median': float(rev.median()),
                                                    'std': float(rev.std()), 'skewness': float(rev.skew())}
        return checks

    # ---------------- Segmentation ----------------
    def analyze_segments(self, data: pd.DataFrame, segment_by: Optional[str] = None) -> Dict[str, Any]:
        seg_col = segment_by or self.config.segment_by
        if seg_col is None:
            return {}
        if seg_col not in data.columns:
            return {'error': f"Column '{seg_col}' not found in data"}
        df = data.copy()
        if seg_col == 'country':
            top = df['country'].value_counts().head(self.config.top_n_countries).index.tolist()
            df = df[df['country'].isin(top)].copy()
        metrics = (df.groupby(['group_name', seg_col])
                    .agg(users=('user_id','count'),
                         conversions=('converted','sum'),
                         total_revenue=('total_revenue', lambda s: s.sum() if 'total_revenue' in df.columns else 0))
                    .reset_index())
        metrics['conversion_rate'] = metrics['conversions'] / metrics['users']
        metrics['avg_revenue'] = metrics['total_revenue'] / metrics['users'].replace({0: np.nan})
        # lift table
        lifts = []
        for val in metrics[seg_col].unique():
            sub = metrics[metrics[seg_col]==val]
            if set(['control','treatment']).issubset(sub['group_name'].values):
                c = float(sub[sub['group_name']=='control']['conversion_rate'].iloc[0])
                t = float(sub[sub['group_name']=='treatment']['conversion_rate'].iloc[0])
                lift_pct = ((t - c) / c * 100) if c > 0 else 0.0
                lifts.append({seg_col: val, 'control_conversion': c, 'treatment_conversion': t, 'lift_pct': lift_pct})
        return {f'segments_{seg_col}': metrics, f'lift_by_{seg_col}': pd.DataFrame(lifts)}

    # ---------------- Visualizations export ----------------
    def save_all_visualizations(self, conversion_df: pd.DataFrame, revenue_df: pd.DataFrame,
                                retention_df: pd.DataFrame, cohort_df: pd.DataFrame, data: pd.DataFrame) -> Dict[str, Any]:
        saved: Dict[str, Any] = {}
        plots_path = self.config.plots_path
        _ensure_dir(plots_path)

        # conversion
        try:
            fig = self.viz.plot_conversion_rate(conversion_df)
            p = plots_path / f"conversion_rate_{self._ts}.png"
            saved['conversion_rate'] = _safe_write_image(fig, p)
        except Exception as e:
            saved['conversion_rate_error'] = str(e)

        # revenue
        try:
            fig = self.viz.plot_revenue_comparison(revenue_df)
            p = plots_path / f"revenue_comparison_{self._ts}.png"
            saved['revenue_comparison'] = _safe_write_image(fig, p)
        except Exception as e:
            saved['revenue_comparison_error'] = str(e)

        # cohort heatmaps
        try:
            pivots = self.coh.create_cohort_pivot_table(retention_df)
            saved['cohort_heatmaps'] = {}
            for g, pivot in pivots.items():
                fig = self.viz.plot_cohort_heatmap(pivot, g)
                p = plots_path / f"cohort_heatmap_{g}_{self._ts}.png"
                saved['cohort_heatmaps'][g] = _safe_write_image(fig, p)
        except Exception as e:
            saved['cohort_heatmaps_error'] = str(e)

        # retention curves
        try:
            fig = self.viz.plot_retention_curves(retention_df)
            p = plots_path / f"retention_curves_{self._ts}.png"
            saved['retention_curves'] = _safe_write_image(fig, p)
        except Exception as e:
            saved['retention_curves_error'] = str(e)

        # cumulative revenue
        try:
            rev_cohort = self.coh.calculate_cohort_revenue(cohort_df)
            fig = self.viz.plot_cumulative_revenue(rev_cohort)
            p = plots_path / f"cumulative_revenue_{self._ts}.png"
            saved['cumulative_revenue'] = _safe_write_image(fig, p)
        except Exception as e:
            saved['cumulative_revenue_error'] = str(e)

        # funnel - guard existence
        try:
            funnel_df = None
            if hasattr(self.ab, 'engine') and self.ab.engine is not None:
                # quick test: try reading, but catch errors if view/table absent
                try:
                    funnel_df = pd.read_sql("SELECT * FROM conversion_funnel", self.ab.engine)
                except Exception:
                    funnel_df = None
            if funnel_df is not None and not funnel_df.empty:
                fig = self.viz.plot_funnel_analysis(funnel_df)
                p = plots_path / f"funnel_{self._ts}.png"
                saved['funnel'] = _safe_write_image(fig, p)
            else:
                saved['funnel'] = None
        except Exception as e:
            saved['funnel_error'] = str(e)

        # sequential
        try:
            seq = self.ab.sequential_testing(data)
            fig = self.viz.plot_sequential_testing(seq)
            p = plots_path / f"sequential_testing_{self._ts}.png"
            saved['sequential_testing'] = _safe_write_image(fig, p)
        except Exception as e:
            saved['sequential_testing_error'] = str(e)

        return saved

    # ---------------- Markdown report ----------------
    def generate_markdown_report(self, summary: SummaryMetrics, interpretation: Dict[str,Any],
                                 quality_checks: Dict[str,Any], segments: Dict[str,Any],
                                 visualizations: Dict[str,Any]) -> str:
        now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        report_path = Path(self.config.report_dir)
        _ensure_dir(report_path)
        md_file = report_path / f"ab_report_{self._ts}.md"

        conv = interpretation['conversion_test']
        rev = interpretation['revenue_test']

        lines: List[str] = []
        lines.append(f"# A/B Test Report")
        lines.append(f"_Generated: {now}_\n")
        # Decision
        decision = "RECOMMEND ROLLOUT" if conv.is_significant and conv.lift_percentage > 0 else "DO NOT ROLLOUT"
        lines.append(f"## Decision: **{decision}**\n")

        # Key metrics (percentages: multiply by 100)
        lines.append("### Key metrics")
        lines.append("| Metric | Control | Treatment | Lift | Significant |")
        lines.append("|---|---:|---:|---:|---:|")
        lines.append(f"| Conversion rate | {_fmt_pct(conv.control_metric)} | {_fmt_pct(conv.treatment_metric)} | {conv.lift_percentage:+.2f}% | {'✅' if conv.is_significant else '❌'} |")
        lines.append(f"| Revenue per user | ${convify(rev.control_metric):s} | ${convify(rev.treatment_metric):s} | {rev.lift_percentage:+.2f}% | {'✅' if rev.is_significant else '❌'} |")
        lines.append("")

        # Dataset overview
        lines.append("## Dataset overview")
        lines.append(f"- Total rows: {summary.total_rows:,}")
        lines.append(f"- Total users: {summary.total_users:,}")
        lines.append(f"- Control users: {summary.control_users:,}")
        lines.append(f"- Treatment users: {summary.treatment_users:,}")
        lines.append(f"- Balance deviation: {summary.balance_ratio*100:.2f}%")
        lines.append(f"- Overall conversion: {summary.overall_conversion_rate*100:.2f}%")
        lines.append(f"- Data quality score: {summary.data_quality_score:.1f}/100\n")

        # Data quality
        lines.append("## Data quality checks\n")
        lines.append("```json")
        lines.append(json.dumps(quality_checks, indent=2, default=str))
        lines.append("```\n")

        # Segments
        lines.append("## Segmentation (sample)\n")
        if isinstance(segments, dict):
            for k, v in segments.items():
                lines.append(f"### {k}")
                if isinstance(v, pd.DataFrame):
                    try:
                        lines.append(v.head(10).to_markdown(index=False))
                    except Exception:
                        lines.append(str(v.head(10)))
                else:
                    lines.append(f"⚠️ {v}")
                lines.append("")
        else:
            lines.append("No segmentation results.\n")

        # Recommendations
        lines.append("## Recommendations\n")
        for r in interpretation.get('recommendations', []):
            lines.append(f"- {r}")
        lines.append("")

        # Visualizations list
        lines.append("## Visualizations\n")
        for name, path in (visualizations or {}).items():
            if isinstance(path, dict):
                lines.append(f"### {name}")
                for sub, p in path.items():
                    lines.append(f"- {sub}: {p}")
            else:
                lines.append(f"- {name}: {path}")
        lines.append("")

        # Appendix
        lines.append("## Appendix")
        lines.append("- Conversion test: two-proportion z-test")
        lines.append("- Revenue test: Welch's t-test")
        lines.append("- Significance: alpha = 0.05\n")

        # write file
        with open(md_file, 'w', encoding='utf-8') as f:
            f.write("\n".join(lines))

        return str(md_file)

    # ---------------- PDF (optional) ----------------
    def convert_to_pdf(self, md_path: str) -> Optional[str]:
        if not self.config.include_pdf:
            return None
        pdf_path = md_path.replace('.md', '.pdf')
        try:
            subprocess.run(['pandoc', md_path, '-o', pdf_path], check=True, capture_output=True)
            return pdf_path
        except Exception:
            return None

    # ---------------- Main orchestrator ----------------
    def generate_full_report(self, data: pd.DataFrame, conversion_df: pd.DataFrame,
                             revenue_df: pd.DataFrame, retention_df: pd.DataFrame,
                             cohort_df: pd.DataFrame, segment_by: Optional[str]=None) -> Dict[str,Any]:
        """
        Run full pipeline and return a dict with paths & serializable results.
        """
        print("Starting full report generation...")
        summary = self.generate_summary(data)
        interpretation = self.interpret_results(data, conversion_df, revenue_df)
        quality = self.check_data_quality(data)
        segments = self.analyze_segments(data, segment_by)
        visualizations = self.save_all_visualizations(conversion_df, revenue_df, retention_df, cohort_df, data)
        md_path = self.generate_markdown_report(summary, interpretation, quality, segments, visualizations)
        pdf_path = self.convert_to_pdf(md_path)
        result = {
            'markdown_report': md_path,
            'pdf_report': pdf_path,
            'summary': summary.to_dict(),
            'interpretation': {
                'conversion_test': interpretation['conversion_test'].to_dict(),
                'revenue_test': interpretation['revenue_test'].to_dict(),
                'recommendations': interpretation['recommendations']
            },
            'quality_checks': quality,
            'segments': {k: (v.to_dict(orient='records') if isinstance(v, pd.DataFrame) else v) for k,v in (segments.items() if isinstance(segments, dict) else [])},
            'visualizations': visualizations
        }
        print("Report generation finished.")
        return result

# ---------- Helper local functions ----------
def convify(val) -> str:
    """Format revenue numbers with commas and two decimals (string)."""
    try:
        return f"{float(val):,.2f}"
    except Exception:
        return str(val)

