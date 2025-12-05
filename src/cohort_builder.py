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
    """Class for constructing and analyzing cohorts"""

    def __init__(self, db_connection_string=None):
        if db_connection_string is None:
            db_connection_string = os.getenv(
                'DATABASE_URL',
                'postgresql://ab_test_user:1234@localhost:5432/ecommerce_ab_test?client_encoding=utf8'
            )
        self.engine = create_engine(db_connection_string)

    def get_cohort_data(self, experiment_name='recommendation_engine_v2'):
        """
        Retrieving data for cohort analysis from the database

        Args:
            experiment_name: experiment name

        Returns:
            DataFrame with cohort data
        """
        query = f"""
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
        WHERE ua.experiment_name = '{experiment_name}'
        ORDER BY u.user_id, t.transaction_date
        """

        df = pd.read_sql(query, self.engine)
        print(f"Loaded {len(df)} records for cohort analysis")
        return df

    def calculate_cohort_retention(self, cohort_data):
        """
        Calculating retention rate by cohort

        Args:
            cohort_data: DataFrame with cohort data

        Returns:
            DataFrame with retention metrics
        """
        # Filter only users with transactions
        cohort_data = cohort_data.dropna(subset=['transaction_date']).copy()

        # Calculate the period (in months) after registration
        cohort_data['cohort_month'] = pd.to_datetime(cohort_data['cohort_month'])
        cohort_data['transaction_month'] = pd.to_datetime(cohort_data['transaction_month'])

        cohort_data['period_number'] = (
                (cohort_data['transaction_month'].dt.year - cohort_data['cohort_month'].dt.year) * 12 +
                (cohort_data['transaction_month'].dt.month - cohort_data['cohort_month'].dt.month)
        )

        # Group by cohort, A/B test group, and period
        retention_data = cohort_data.groupby([
            'cohort_month', 'group_name', 'period_number'
        ])['user_id'].nunique().reset_index()

        retention_data.columns = ['cohort_month', 'group_name', 'period_number', 'users']

        # Calculating the initial cohort size
        cohort_sizes = cohort_data.groupby([
            'cohort_month', 'group_name'
        ])['user_id'].nunique().reset_index()
        cohort_sizes.columns = ['cohort_month', 'group_name', 'cohort_size']

        # Merge and calculate retention rate
        retention_data = retention_data.merge(
            cohort_sizes,
            on=['cohort_month', 'group_name']
        )

        retention_data['retention_rate'] = (
                retention_data['users'] / retention_data['cohort_size'] * 100
        ).round(2)

        return retention_data

    def calculate_cohort_revenue(self, cohort_data):
        """
        Calculation of revenue metrics by cohorts

        Args:
            cohort_data: DataFrame with cohort data

        Returns:
            DataFrame with revenue metrics
        """
        cohort_data = cohort_data.dropna(subset=['transaction_date']).copy()

        # Calculate the period
        cohort_data['cohort_month'] = pd.to_datetime(cohort_data['cohort_month'])
        cohort_data['transaction_month'] = pd.to_datetime(cohort_data['transaction_month'])

        cohort_data['period_number'] = (
                (cohort_data['transaction_month'].dt.year - cohort_data['cohort_month'].dt.year) * 12 +
                (cohort_data['transaction_month'].dt.month - cohort_data['cohort_month'].dt.month)
        )

        # Cumulative revenue by cohort
        revenue_data = cohort_data.groupby([
            'cohort_month', 'group_name', 'period_number'
        ]).agg({
            'total_amount': 'sum',
            'user_id': 'nunique'
        }).reset_index()

        revenue_data.columns = [
            'cohort_month', 'group_name', 'period_number',
            'revenue', 'active_users'
        ]

        # ARPU (Average Revenue Per User)
        revenue_data['arpu'] = (
                revenue_data['revenue'] / revenue_data['active_users']
        ).round(2)

        # Cumulative revenue
        revenue_data = revenue_data.sort_values(['cohort_month', 'group_name', 'period_number'])
        revenue_data['cumulative_revenue'] = revenue_data.groupby([
            'cohort_month', 'group_name'
        ])['revenue'].cumsum()

        return revenue_data

    def calculate_ltv(self, cohort_data, prediction_periods=12):
        """
        Calculating Lifetime Value (LTV) by cohort

        Args:
            cohort data: DataFrame with cohort data
            prediction periods: number of months for the prediction

        Returns:
            DataFrame with LTV metrics
        """
        # Get revenue data
        revenue_data = self.calculate_cohort_revenue(cohort_data)

        # Group by cohort and group
        ltv_data = revenue_data.groupby([
            'cohort_month', 'group_name'
        ]).agg({
            'cumulative_revenue': 'max',
            'active_users': 'first'
        }).reset_index()

        # Average LTV
        ltv_data['ltv'] = (
                ltv_data['cumulative_revenue'] / ltv_data['active_users']
        ).round(2)

        return ltv_data

    def create_cohort_pivot_table(self, retention_data):
        """
        Creating a pivot table to visualize cohorts

        Args:
            retention_data: DataFrame with retention data

        Returns:
            Pivot table for each group
        """
        pivot_tables = {}

        for group in retention_data['group_name'].unique():
            group_data = retention_data[retention_data['group_name'] == group]

            pivot = group_data.pivot_table(
                values='retention_rate',
                index='cohort_month',
                columns='period_number',
                aggfunc='first'
            )

            pivot_tables[group] = pivot

        return pivot_tables

    def compare_cohorts_ab(self, cohort_data):
        """
        Cohort comparison between control and test groups

        Args:
            cohort_data: DataFrame with cohort data

        Returns:
            DataFrame with comparative statistics
        """
        # Retention comparison
        retention_data = self.calculate_cohort_retention(cohort_data)

        retention_comparison = retention_data.groupby([
            'period_number', 'group_name'
        ])['retention_rate'].mean().reset_index()

        retention_pivot = retention_comparison.pivot(
            index='period_number',
            columns='group_name',
            values='retention_rate'
        )

        if 'control' in retention_pivot.columns and 'treatment' in retention_pivot.columns:
            retention_pivot['lift'] = (
                    (retention_pivot['treatment'] - retention_pivot['control']) /
                    retention_pivot['control'] * 100
            ).round(2)

        # Revenue comparison
        revenue_data = self.calculate_cohort_revenue(cohort_data)

        revenue_comparison = revenue_data.groupby([
            'period_number', 'group_name'
        ])['arpu'].mean().reset_index()

        revenue_pivot = revenue_comparison.pivot(
            index='period_number',
            columns='group_name',
            values='arpu'
        )

        if 'control' in revenue_pivot.columns and 'treatment' in revenue_pivot.columns:
            revenue_pivot['lift'] = (
                    (revenue_pivot['treatment'] - revenue_pivot['control']) /
                    revenue_pivot['control'] * 100
            ).round(2)

        return {
            'retention_comparison': retention_pivot,
            'revenue_comparison': revenue_pivot
        }


if __name__ == "__main__":
    # Usage example
    analyzer = CohortAnalyzer()

    print("Loading cohort data...")
    cohort_data = analyzer.get_cohort_data()

    print("\nRetention calculation...")
    retention = analyzer.calculate_cohort_retention(cohort_data)
    print(retention.head())

    print("\nCalculation of revenue metrics...")
    revenue = analyzer.calculate_cohort_revenue(cohort_data)
    print(revenue.head())

    print("\nLTV calculation...")
    ltv = analyzer.calculate_ltv(cohort_data)
    print(ltv.head())

    print("\nA/B group comparison...")
    comparison = analyzer.compare_cohorts_ab(cohort_data)
    print("\nRetention comparison:")
    print(comparison['retention_comparison'])
    print("\nRevenue comparison:")
    print(comparison['revenue_comparison'])