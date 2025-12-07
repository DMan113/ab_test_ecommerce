"""
Cohort Analysis for E-Commerce A/B Testing
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
            db_connection_string = os.getenv('DATABASE_URL')
        self.engine = create_engine(db_connection_string)

    def get_cohort_data(self, experiment_name='recommendation_engine_v2'):
        """Loading cohort data from the database"""
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

        # data type conversion
        df['cohort_month'] = pd.to_datetime(df['cohort_month'], errors='coerce')
        df['transaction_month'] = pd.to_datetime(df['transaction_month'], errors='coerce')
        df['registration_date'] = pd.to_datetime(df['registration_date'], errors='coerce')
        df['transaction_date'] = pd.to_datetime(df['transaction_date'], errors='coerce')

        # Delete rows with invalid dates
        df = df.dropna(subset=['cohort_month', 'registration_date'])

        print(f"✅ Loaded {len(df)} records for cohort analysis")
        print(f"📊 Date range: {df['cohort_month'].min()} to {df['cohort_month'].max()}")

        return df

    def calculate_cohort_retention(self, cohort_data):
        """
        Retention rate calculation with correct NaN handling
        """
        data = cohort_data.copy()

        # Make sure the dates are in the correct format
        data['cohort_month'] = pd.to_datetime(data['cohort_month'])
        data['transaction_month'] = pd.to_datetime(data['transaction_month'])

        # Period calculation
        data['period_number'] = (
            (data['transaction_month'].dt.year - data['cohort_month'].dt.year) * 12 +
            (data['transaction_month'].dt.month - data['cohort_month'].dt.month)
        )

        # Delete invalid periods
        data = data[data['period_number'].notna()].copy()
        data = data[data['period_number'] >= 0].copy()
        data['period_number'] = data['period_number'].astype(int)

        # Calculating the number of active users
        retention = data.groupby(
            ['cohort_month', 'group_name', 'period_number']
        )['user_id'].nunique().reset_index(name='active_users')

        # Calculating cohort size (all users who signed up this month)
        cohort_sizes = cohort_data.groupby(
            ['cohort_month', 'group_name']
        )['user_id'].nunique().reset_index(name='cohort_size')

        # Merging
        retention = retention.merge(
            cohort_sizes,
            on=['cohort_month', 'group_name'],
            how='left'
        )

        # Preventing division by zero
        retention['cohort_size'] = retention['cohort_size'].replace(0, np.nan)
        retention['retention_rate'] = (
            (retention['active_users'] / retention['cohort_size']) * 100
        ).round(2)

        # Replace inf and negative values ​​with NaN
        retention['retention_rate'] = retention['retention_rate'].replace([np.inf, -np.inf], np.nan)

        # Delete rows with NaN retention_rate
        retention = retention.dropna(subset=['retention_rate'])

        print(f"✅ Calculated retention for {len(retention)} cohort-period combinations")
        print(f"📈 Retention range: {retention['retention_rate'].min():.1f}% to {retention['retention_rate'].max():.1f}%")

        return retention

    def calculate_cohort_revenue(self, cohort_data):
        """Calculation of revenue per cohort"""
        data = cohort_data.copy()

        data['cohort_month'] = pd.to_datetime(data['cohort_month'])
        data['transaction_month'] = pd.to_datetime(data['transaction_month'])

        # Period calculation
        data['period_number'] = (
            (data['transaction_month'].dt.year - data['cohort_month'].dt.year) * 12 +
            (data['transaction_month'].dt.month - data['cohort_month'].dt.month)
        )

        # Filtration
        active = data[data['period_number'].notna() & (data['period_number'] >= 0)].copy()
        active['period_number'] = active['period_number'].astype(int)

        # Aggregation
        revenue = active.groupby(
            ['cohort_month', 'group_name', 'period_number']
        ).agg(
            revenue=('total_amount', 'sum'),
            active_users=('user_id', 'nunique')
        ).reset_index()

        # ARPU (Average Revenue Per User)
        revenue['arpu'] = (revenue['revenue'] / revenue['active_users']).round(2)

        # Cumulative revenue
        revenue = revenue.sort_values(['cohort_month', 'group_name', 'period_number'])
        revenue['cumulative_revenue'] = revenue.groupby(
            ['cohort_month', 'group_name']
        )['revenue'].cumsum()

        print(f"✅ Calculated revenue for {len(revenue)} cohort-period combinations")

        return revenue

    def calculate_ltv(self, cohort_data):
        """Lifetime Value Calculation"""
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

    def create_cohort_pivot_table(self, retention_data):
        """
        Creating a pivot table without NaN
        """
        pivot_tables = {}

        for group in retention_data['group_name'].unique():
            group_df = retention_data[retention_data['group_name'] == group].copy()

            # Remove NaN before creating pivot
            group_df = group_df.dropna(subset=['retention_rate'])

            # Creating a pivot table
            pivot = group_df.pivot_table(
                index='cohort_month',
                columns='period_number',
                values='retention_rate',
                aggfunc='mean'  # Use mean for aggregation
            )

            # Fill NaN with zeros (if there is no data for the period)
            pivot = pivot.fillna(0)

            # Convert all values ​​to numeric
            pivot = pivot.apply(pd.to_numeric, errors='coerce')

            # Sorting
            pivot = pivot.sort_index()
            pivot = pivot.reindex(sorted(pivot.columns), axis=1)

            # Final check: replace all NaN with 0
            pivot = pivot.fillna(0)

            print(f"✅ Created pivot for {group}: shape {pivot.shape}")
            print(f"   NaN values: {pivot.isna().sum().sum()}")

            pivot_tables[group] = pivot

        return pivot_tables

    def compare_cohorts_ab(self, cohort_data):
        """Comparison of cohorts between groups A/B"""
        retention = self.calculate_cohort_retention(cohort_data)

        # Mean retention by periods
        mean_retention = retention.groupby(
            ['period_number', 'group_name']
        )['retention_rate'].mean().reset_index()

        retention_pivot = mean_retention.pivot(
            index='period_number',
            columns='group_name',
            values='retention_rate'
        )

        # Lift calculation
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