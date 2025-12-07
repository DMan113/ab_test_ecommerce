"""
Cohort Analysis for E-Commerce A/B Testing
Refactored for stability, numeric correctness, and clean visualization.
"""

import pandas as pd
import numpy as np
from sqlalchemy import create_engine
from datetime import datetime
import os
from dotenv import load_dotenv

load_dotenv()


class CohortAnalyzer:
    """Class for constructing and analyzing cohorts with stable, clean outputs."""

    def __init__(self, db_connection_string=None):
        if db_connection_string is None:
            db_connection_string = os.getenv(
                'DATABASE_URL',
                'postgresql://ab_test_user:1234@localhost:5432/ecommerce_ab_test?client_encoding=utf8'
            )
        self.engine = create_engine(db_connection_string)

    # -------------------------------------------------------------------------
    # 1) Load cohort data
    # -------------------------------------------------------------------------
    def get_cohort_data(self, experiment_name='recommendation_engine_v2'):
        query = """
            SELECT 
                u.user_id,
                u.user_uuid,
                u.registration_date,
                DATE_TRUNC('month', u.registration_date) as cohort_month,
                ua.group_id,
                eg.group_name,
                t.transaction_id,
                t.transaction_date,
                t.total_amount,
                DATE_TRUNC('month', t.transaction_date) as transaction_month
            FROM users u
            LEFT JOIN user_assignments ua ON u.user_id = ua.user_id
            LEFT JOIN experiment_groups eg ON ua.group_id = eg.group_id
            LEFT JOIN transactions t ON u.user_id = t.user_id
            WHERE ua.experiment_name = %(exp_name)s
            ORDER BY u.user_id, t.transaction_date
        """

        df = pd.read_sql(query, self.engine, params={'exp_name': experiment_name})

        # Normalize types once here
        df['cohort_month'] = pd.to_datetime(df['cohort_month'])
        df['transaction_month'] = pd.to_datetime(df['transaction_month'])

        print(f"Loaded {len(df)} records for cohort analysis")
        return df

    # -------------------------------------------------------------------------
    # 2) Retention calculation
    # -------------------------------------------------------------------------
    def calculate_cohort_retention(self, cohort_data):
        data = cohort_data.copy()

        # Ensure datetimes
        data['cohort_month'] = pd.to_datetime(data['cohort_month'])
        data['transaction_month'] = pd.to_datetime(data['transaction_month'])

        # Period number in months
        data['period_number'] = (
            (data['transaction_month'].dt.year - data['cohort_month'].dt.year) * 12 +
            (data['transaction_month'].dt.month - data['cohort_month'].dt.month)
        )

        # Remove users who have no transactions (period = NaN)
        active = data[data['period_number'].notna() & (data['period_number'] >= 0)].copy()
        active['period_number'] = active['period_number'].astype(int)

        # Count unique active users per period
        retention = active.groupby(
            ['cohort_month', 'group_name', 'period_number']
        )['user_id'].nunique().reset_index(name='users')

        # Cohort sizes
        cohort_sizes = data.groupby(
            ['cohort_month', 'group_name']
        )['user_id'].nunique().reset_index(name='cohort_size')

        # Merge
        retention = retention.merge(cohort_sizes, on=['cohort_month', 'group_name'], how='left')

        # Retention %
        retention['retention_rate'] = (
            retention['users'] / retention['cohort_size'] * 100
        ).round(2)

        # Ensure numeric
        retention['retention_rate'] = pd.to_numeric(retention['retention_rate'], errors="coerce")

        return retention

    # -------------------------------------------------------------------------
    # 3) Revenue per cohort
    # -------------------------------------------------------------------------
    def calculate_cohort_revenue(self, cohort_data):
        data = cohort_data.copy()

        data['cohort_month'] = pd.to_datetime(data['cohort_month'])
        data['transaction_month'] = pd.to_datetime(data['transaction_month'])

        data['period_number'] = (
            (data['transaction_month'].dt.year - data['cohort_month'].dt.year) * 12 +
            (data['transaction_month'].dt.month - data['cohort_month'].dt.month)
        )

        active = data[data['period_number'].notna() & (data['period_number'] >= 0)].copy()
        active['period_number'] = active['period_number'].astype(int)

        revenue = active.groupby(
            ['cohort_month', 'group_name', 'period_number']
        ).agg(
            revenue=('total_amount', 'sum'),
            active_users=('user_id', 'nunique')
        ).reset_index()

        revenue['arpu'] = (
            revenue['revenue'] / revenue['active_users']
        ).round(2)

        # cumulative revenue
        revenue = revenue.sort_values(['cohort_month', 'group_name', 'period_number'])
        revenue['cumulative_revenue'] = revenue.groupby(
            ['cohort_month', 'group_name']
        )['revenue'].cumsum()

        return revenue

    # -------------------------------------------------------------------------
    # 4) LTV
    # -------------------------------------------------------------------------
    def calculate_ltv(self, cohort_data):
        revenue_data = self.calculate_cohort_revenue(cohort_data)

        cohort_sizes = cohort_data.groupby(
            ['cohort_month', 'group_name']
        )['user_id'].nunique().reset_index(name='cohort_size')

        ltv = revenue_data.groupby(
            ['cohort_month', 'group_name']
        )['cumulative_revenue'].max().reset_index()

        ltv = ltv.merge(cohort_sizes, on=['cohort_month', 'group_name'])
        ltv['ltv'] = (ltv['cumulative_revenue'] / ltv['cohort_size']).round(2)

        return ltv

    # -------------------------------------------------------------------------
    # 5) Cohort pivot (used by heatmaps)
    # -------------------------------------------------------------------------
    def create_cohort_pivot_table(self, retention_data):
        """
        Produce numeric, clean cohort matrices for each group.
        Ensures:
        - sorted cohorts
        - sorted periods
        - numeric pivot values
        """

        pivot_tables = {}

        for group in retention_data['group_name'].unique():
            group_df = retention_data[retention_data['group_name'] == group].copy()

            pivot = group_df.pivot_table(
                index='cohort_month',
                columns='period_number',
                values='retention_rate',
                aggfunc='mean'
            )

            # Make sure everything is numeric
            pivot = pivot.apply(pd.to_numeric, errors="coerce")

            # Sort rows & columns
            pivot = pivot.sort_index()
            pivot = pivot.reindex(sorted(pivot.columns), axis=1)

            pivot_tables[group] = pivot

        return pivot_tables

    # -------------------------------------------------------------------------
    # 6) A/B cohort comparison
    # -------------------------------------------------------------------------
    def compare_cohorts_ab(self, cohort_data):
        retention = self.calculate_cohort_retention(cohort_data)

        mean_retention = retention.groupby(
            ['period_number', 'group_name']
        )['retention_rate'].mean().reset_index()

        retention_pivot = mean_retention.pivot(
            index='period_number',
            columns='group_name',
            values='retention_rate'
        )

        if {'control', 'treatment'}.issubset(retention_pivot.columns):
            retention_pivot['lift'] = (
                (retention_pivot['treatment'] - retention_pivot['control']) /
                retention_pivot['control'] * 100
            ).round(2)

        # Revenue comparison
        revenue = self.calculate_cohort_revenue(cohort_data)

        mean_arpu = revenue.groupby(
            ['period_number', 'group_name']
        )['arpu'].mean().reset_index()

        revenue_pivot = mean_arpu.pivot(
            index='period_number',
            columns='group_name',
            values='arpu'
        )

        if {'control', 'treatment'}.issubset(revenue_pivot.columns):
            revenue_pivot['lift'] = (
                (revenue_pivot['treatment'] - revenue_pivot['control']) /
                revenue_pivot['control'] * 100
            ).round(2)

        return {
            'retention_comparison': retention_pivot,
            'revenue_comparison': revenue_pivot
        }
