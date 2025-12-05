"""
Visualizations for A/B testing and cohort analysis
"""
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import os
from sqlalchemy import create_engine
from dotenv import load_dotenv

load_dotenv()


class ABTestVisualizer:
    """Class for creating A/B test visualizations"""

    def __init__(self, save_dir="plots"):
        # Color scheme for groups
        self.colors = {
            'control': '#3498db',  # Blue
            'treatment': '#e74c3c'  # Red
        }
        # Create a folder for graphs if one does not exist
        self.save_dir = save_dir
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

    def save_plot(self, fig, filename, format="html"):
        """
        Method for saving a plot
        """
        path = os.path.join(self.save_dir, filename)

        if format == "html":
            fig.write_html(f"{path}.html")
            print(f"Plot saved: {path}.html")
        elif format in ["png", "jpg", "jpeg", "svg"]:
            # Requires pip install kaleido
            try:
                fig.write_image(f"{path}.{format}", scale=2)
                print(f"Plot saved: {path}.{format}")
            except ValueError:
                print("Error saving image. Install kaleido: pip install kaleido")
        else:
            print("Unknown file format")

    def plot_conversion_rate(self, conversion_data, title="Conversion Rate Comparison"):
        """
        Visualization of conversion rate comparison
        """
        fig = go.Figure()

        for _, row in conversion_data.iterrows():
            fig.add_trace(go.Bar(
                name=row['group_name'].capitalize(),
                x=[row['group_name'].capitalize()],
                y=[row['conversion_rate']],
                text=[f"{row['conversion_rate']:.2f}%"],
                textposition='outside',
                marker_color=self.colors.get(row['group_name'], '#95a5a6'),
                hovertemplate='<b>%{x}</b><br>' +
                              'Conversion Rate: %{y:.2f}%<br>' +
                              f"Users: {row['users']:,}<br>" +
                              f"Conversions: {row['conversions']:,}<extra></extra>"
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Group",
            yaxis_title="Conversion Rate (%)",
            showlegend=False,
            height=500,
            template='plotly_white'
        )

        return fig

    def plot_revenue_comparison(self, revenue_data, title="Revenue Per User Comparison"):
        """
        Revenue comparison visualization
        """
        fig = go.Figure()

        for _, row in revenue_data.iterrows():
            fig.add_trace(go.Bar(
                name=row['group_name'].capitalize(),
                x=[row['group_name'].capitalize()],
                y=[row['avg_revenue_per_user']],
                text=[f"${row['avg_revenue_per_user']:.2f}"],
                textposition='outside',
                marker_color=self.colors.get(row['group_name'], '#95a5a6'),
                error_y=dict(
                    type='data',
                    array=[row['std_revenue']],
                    visible=True
                ),
                hovertemplate='<b>%{x}</b><br>' +
                              'Avg Revenue: $%{y:.2f}<br>' +
                              f"Total Revenue: ${row['total_revenue']:,.2f}<br>" +
                              f"Std Dev: ${row['std_revenue']:.2f}<extra></extra>"
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Group",
            yaxis_title="Average Revenue per User ($)",
            showlegend=False,
            height=500,
            template='plotly_white'
        )

        return fig

    def plot_cohort_heatmap(self, cohort_pivot, group_name, title=None):
        """
        Heatmap for cohort analysis
        """
        if title is None:
            title = f"Cohort Retention Heatmap - {group_name.capitalize()} Group"

        # Convert the index to year-month format
        cohort_pivot.index = cohort_pivot.index.strftime('%Y-%m')

        fig = go.Figure(data=go.Heatmap(
            z=cohort_pivot.values,
            x=[f"Month {int(col)}" for col in cohort_pivot.columns],
            y=cohort_pivot.index,
            colorscale='RdYlGn',
            text=cohort_pivot.values,
            texttemplate='%{text:.1f}%',
            textfont={"size": 10},
            colorbar=dict(title="Retention %")
        ))

        fig.update_layout(
            title=title,
            xaxis_title="Period After Registration",
            yaxis_title="Cohort Month",
            height=600,
            template='plotly_white'
        )

        return fig

    def plot_retention_curves(self, retention_data, title="Retention Curves by Group"):
        """
        Retention curves by group
        """
        fig = go.Figure()

        for group in retention_data['group_name'].unique():
            group_data = retention_data[retention_data['group_name'] == group]

            # Aggregate by periods
            avg_retention = group_data.groupby('period_number')['retention_rate'].mean()

            fig.add_trace(go.Scatter(
                x=avg_retention.index,
                y=avg_retention.values,
                mode='lines+markers',
                name=group.capitalize(),
                line=dict(
                    color=self.colors.get(group, '#95a5a6'),
                    width=3
                ),
                marker=dict(size=8),
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'Period: %{x}<br>' +
                              'Retention: %{y:.2f}%<extra></extra>'
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Months After Registration",
            yaxis_title="Retention Rate (%)",
            height=500,
            template='plotly_white',
            hovermode='x unified'
        )

        return fig

    def plot_cumulative_revenue(self, revenue_data, title="Cumulative Revenue by Cohort"):
        """
        Кумулятивний revenue по когортам
        """
        fig = go.Figure()

        for group in revenue_data['group_name'].unique():
            group_data = revenue_data[revenue_data['group_name'] == group]

            # Aggregate by periods
            avg_cumulative = group_data.groupby('period_number')['cumulative_revenue'].mean()

            fig.add_trace(go.Scatter(
                x=avg_cumulative.index,
                y=avg_cumulative.values,
                mode='lines+markers',
                name=group.capitalize(),
                line=dict(
                    color=self.colors.get(group, '#95a5a6'),
                    width=3
                ),
                marker=dict(size=8),
                fill='tonexty' if group == 'treatment' else None,
                hovertemplate='<b>%{fullData.name}</b><br>' +
                              'Period: %{x}<br>' +
                              'Cumulative Revenue: $%{y:,.2f}<extra></extra>'
            ))

        fig.update_layout(
            title=title,
            xaxis_title="Months After Registration",
            yaxis_title="Cumulative Revenue ($)",
            height=500,
            template='plotly_white',
            hovermode='x unified'
        )

        return fig

    def plot_funnel_analysis(self, funnel_data, title="Conversion Funnel by Group"):
        """
        Conversion funnel analysis
        """
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Control Group', 'Treatment Group'),
            specs=[[{'type': 'funnel'}, {'type': 'funnel'}]]
        )

        stages = ['Views', 'Add to Cart', 'Checkout', 'Purchase']

        for idx, group in enumerate(['control', 'treatment'], start=1):
            if group in funnel_data['group_name'].values:
                group_row = funnel_data[funnel_data['group_name'] == group].iloc[0]

                values = [
                    group_row['views'],
                    group_row['add_to_cart'],
                    group_row['checkout'],
                    group_row['purchases']
                ]

                fig.add_trace(
                    go.Funnel(
                        y=stages,
                        x=values,
                        textinfo="value+percent initial",
                        marker=dict(color=self.colors[group])
                    ),
                    row=1, col=idx
                )

        fig.update_layout(
            title_text=title,
            height=500,
            showlegend=False
        )

        return fig

    def plot_statistical_power(self, sample_sizes, effect_sizes, alpha=0.05):
        """
        Power analysis graph
        """
        from scipy.stats import norm

        fig = go.Figure()

        for effect_size in effect_sizes:
            powers = []
            for n in sample_sizes:
                # Simplified power calculation
                z_alpha = norm.ppf(1 - alpha / 2)
                z_beta = effect_size * np.sqrt(n / 2) - z_alpha
                power = norm.cdf(z_beta)
                powers.append(power)

            fig.add_trace(go.Scatter(
                x=sample_sizes,
                y=powers,
                mode='lines',
                name=f'Effect Size: {effect_size:.2f}',
                line=dict(width=2)
            ))

        # Add a horizontal line at 0.8 (standard power)
        fig.add_hline(
            y=0.8,
            line_dash="dash",
            line_color="red",
            annotation_text="Target Power (0.8)"
        )

        fig.update_layout(
            title="Statistical Power Analysis",
            xaxis_title="Sample Size per Group",
            yaxis_title="Statistical Power",
            height=500,
            template='plotly_white',
            hovermode='x unified'
        )

        return fig

    def plot_sequential_testing(self, sequential_data, title="Sequential Testing Results"):
        """
        Visualization of sequential testing results
        """
        fig = make_subplots(
            rows=2, cols=1,
            subplot_titles=('P-value Over Time', 'Lift Over Time'),
            vertical_spacing=0.15
        )

        # P-value plot
        fig.add_trace(
            go.Scatter(
                x=sequential_data['sample_size'],
                y=sequential_data['p_value'],
                mode='lines+markers',
                name='P-value',
                line=dict(color='#3498db', width=2),
                marker=dict(size=6)
            ),
            row=1, col=1
        )

        # Add a significance line (alpha = 0.05)
        fig.add_hline(
            y=0.05,
            line_dash="dash",
            line_color="red",
            row=1, col=1,
            annotation_text="α = 0.05"
        )

        # Lift plot
        colors = ['#2ecc71' if sig else '#95a5a6'
                  for sig in sequential_data['significant']]

        fig.add_trace(
            go.Scatter(
                x=sequential_data['sample_size'],
                y=sequential_data['lift'],
                mode='lines+markers',
                name='Lift %',
                line=dict(color='#e74c3c', width=2),
                marker=dict(size=6, color=colors)
            ),
            row=2, col=1
        )

        fig.update_xaxes(title_text="Sample Size", row=2, col=1)
        fig.update_yaxes(title_text="P-value", row=1, col=1)
        fig.update_yaxes(title_text="Lift (%)", row=2, col=1)

        fig.update_layout(
            title_text=title,
            height=700,
            showlegend=True,
            template='plotly_white'
        )

        return fig

    def create_dashboard(self, conversion_data, revenue_data, retention_data,
                         funnel_data=None):
        """
        Creating an interactive dashboard with all metrics
        """
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=(
                'Conversion Rate Comparison',
                'Revenue Per User',
                'Retention Curves',
                'Cumulative Revenue'
            ),
            specs=[
                [{'type': 'bar'}, {'type': 'bar'}],
                [{'type': 'scatter'}, {'type': 'scatter'}]
            ],
            vertical_spacing=0.12,
            horizontal_spacing=0.1
        )

        # 1. Conversion Rate
        for _, row in conversion_data.iterrows():
            fig.add_trace(
                go.Bar(
                    name=row['group_name'].capitalize(),
                    x=[row['group_name'].capitalize()],
                    y=[row['conversion_rate']],
                    marker_color=self.colors.get(row['group_name'], '#95a5a6'),
                    showlegend=False
                ),
                row=1, col=1
            )

        # 2. Revenue
        for _, row in revenue_data.iterrows():
            fig.add_trace(
                go.Bar(
                    name=row['group_name'].capitalize(),
                    x=[row['group_name'].capitalize()],
                    y=[row['avg_revenue_per_user']],
                    marker_color=self.colors.get(row['group_name'], '#95a5a6'),
                    showlegend=False
                ),
                row=1, col=2
            )

        # 3. Retention Curves
        for group in retention_data['group_name'].unique():
            group_data = retention_data[retention_data['group_name'] == group]
            avg_retention = group_data.groupby('period_number')['retention_rate'].mean()

            fig.add_trace(
                go.Scatter(
                    x=avg_retention.index,
                    y=avg_retention.values,
                    mode='lines+markers',
                    name=group.capitalize(),
                    line=dict(color=self.colors.get(group, '#95a5a6'), width=2),
                    showlegend=True
                ),
                row=2, col=1
            )

        # 4. Cumulative Revenue
        for group in retention_data['group_name'].unique():
            group_data = retention_data[retention_data['group_name'] == group]

            if 'cumulative_revenue' in group_data.columns:
                avg_cumulative = group_data.groupby('period_number')['cumulative_revenue'].mean()

                fig.add_trace(
                    go.Scatter(
                        x=avg_cumulative.index,
                        y=avg_cumulative.values,
                        mode='lines+markers',
                        name=group.capitalize(),
                        line=dict(color=self.colors.get(group, '#95a5a6'), width=2),
                        showlegend=False
                    ),
                    row=2, col=2
                )

        fig.update_layout(
            title_text="A/B Test Dashboard",
            height=800,
            template='plotly_white'
        )

        return fig


if __name__ == "__main__":
    # Example of use with data from a DB
    from cohort_builder import CohortAnalyzer
    from ab_test import ABTestAnalyzer

    print("The visualization module is ready to use!")
    print("Import ABTestVisualizer into your notebooks to create graphs.")

    # Automatic generation of graphs from metrics
    cohort_analyzer = CohortAnalyzer()
    ab_analyzer = ABTestAnalyzer()

    cohort_data = cohort_analyzer.get_cohort_data()
    retention = cohort_analyzer.calculate_cohort_retention(cohort_data)
    revenue = cohort_analyzer.calculate_cohort_revenue(cohort_data)

    report = ab_analyzer.full_analysis_report()
    conversion_metrics = report['conversion_metrics']
    revenue_metrics = report['revenue_metrics']

    # Funnel з view
    funnel_query = "SELECT * FROM conversion_funnel"
    funnel_data = pd.read_sql(funnel_query, ab_analyzer.engine)

    viz = ABTestVisualizer(save_dir="result_plots")

    # Generation of all graphs
    figs = {}

    figs['conversion'] = viz.plot_conversion_rate(conversion_metrics)
    figs['conversion'].show()
    viz.save_plot(figs['conversion'], "conversion_rate", "html")

    figs['revenue'] = viz.plot_revenue_comparison(revenue_metrics)
    figs['revenue'].show()
    viz.save_plot(figs['revenue'], "revenue_comparison", "png")

    pivot_tables = cohort_analyzer.create_cohort_pivot_table(retention)
    for group, pivot in pivot_tables.items():
        figs[f'heatmap_{group}'] = viz.plot_cohort_heatmap(pivot, group)
        figs[f'heatmap_{group}'].show()
        viz.save_plot(figs[f'heatmap_{group}'], f"cohort_heatmap_{group}", "html")

    figs['retention_curves'] = viz.plot_retention_curves(retention)
    figs['retention_curves'].show()
    viz.save_plot(figs['retention_curves'], "retention_curves", "png")

    figs['cumulative_revenue'] = viz.plot_cumulative_revenue(revenue)
    figs['cumulative_revenue'].show()
    viz.save_plot(figs['cumulative_revenue'], "cumulative_revenue", "html")

    figs['funnel'] = viz.plot_funnel_analysis(funnel_data)
    figs['funnel'].show()
    viz.save_plot(figs['funnel'], "funnel_analysis", "png")

    sample_sizes = list(range(100, 10000, 100))
    effect_sizes = [0.01, 0.02, 0.05]
    figs['power'] = viz.plot_statistical_power(sample_sizes, effect_sizes)
    figs['power'].show()
    viz.save_plot(figs['power'], "statistical_power", "html")

    data = ab_analyzer.get_ab_test_data()
    sequential_data = ab_analyzer.sequential_testing(data)
    figs['sequential'] = viz.plot_sequential_testing(sequential_data)
    figs['sequential'].show()
    viz.save_plot(figs['sequential'], "sequential_testing", "png")

    figs['dashboard'] = viz.create_dashboard(conversion_metrics, revenue_metrics, retention, funnel_data)
    figs['dashboard'].show()
    viz.save_plot(figs['dashboard'], "ab_dashboard", "html")