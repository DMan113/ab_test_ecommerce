"""
Statistical Analysis of A/B Tests
"""

import pandas as pd
import numpy as np
from scipy import stats
from sqlalchemy import create_engine
import os
from dotenv import load_dotenv

load_dotenv()


class ABTestAnalyzer:
    """Class for statistical analysis of A/B tests"""

    def __init__(self, db_connection_string=None):
        if db_connection_string is None:
            db_connection_string = os.getenv(
                'DATABASE_URL',
                'postgresql://ab_test_user:1234@localhost:5432/ecommerce_ab_test?client_encoding=utf8'
            )
        self.engine = create_engine(db_connection_string)

    def get_ab_test_data(self, experiment_name='recommendation_engine_v2'):
        """
        Retrieving A/B test data from the database

        Args:
        expriment_name: experiment name

        Returns:
        DataFrame with test datae
        """
        query = f"""
        SELECT 
            u.user_id,
            ua.group_id,
            eg.group_name,
            COUNT(DISTINCT t.transaction_id) as transaction_count,
            SUM(t.total_amount) as total_revenue,
            CASE WHEN COUNT(t.transaction_id) > 0 THEN 1 ELSE 0 END as converted
        FROM users u
        JOIN user_assignments ua ON u.user_id = ua.user_id
        JOIN experiment_groups eg ON ua.group_id = eg.group_id
        LEFT JOIN transactions t ON u.user_id = t.user_id
        WHERE ua.experiment_name = '{experiment_name}'
        GROUP BY u.user_id, ua.group_id, eg.group_name
        """

        df = pd.read_sql(query, self.engine)
        print(f"Loaded data for {len(df)} users")
        return df

    def calculate_conversion_rate(self, data):
        """
        Calculate conversion rate for each group

        Args:
            data: DataFrame with A/B test data

        Returns:
            DataFrame with conversion metrics
        """
        metrics = data.groupby('group_name').agg({
            'user_id': 'count',
            'converted': 'sum'
        }).reset_index()

        metrics.columns = ['group_name', 'users', 'conversions']
        metrics['conversion_rate'] = (
                metrics['conversions'] / metrics['users'] * 100
        ).round(2)

        return metrics

    def calculate_revenue_metrics(self, data):
        """
        Calculating revenue metrics

        Args:
            data: DataFrame with A/B test data

        Returns:
            DataFrame with revenue metrics
        """
        metrics = data.groupby('group_name').agg({
            'user_id': 'count',
            'total_revenue': ['sum', 'mean', 'std']
        }).reset_index()

        metrics.columns = [
            'group_name', 'users', 'total_revenue',
            'avg_revenue_per_user', 'std_revenue'
        ]

        metrics['total_revenue'] = metrics['total_revenue'].round(2)
        metrics['avg_revenue_per_user'] = metrics['avg_revenue_per_user'].round(2)
        metrics['std_revenue'] = metrics['std_revenue'].round(2)

        return metrics

    def z_test_proportions(self, control_data, treatment_data):
        """
        Z-test для порівняння conversion rates (пропорцій)

        Args:
            control_data: дані контрольної групи
            treatment_data: дані тестової групи

        Returns:
            dict з результатами тесту
        """
        # Кількість конверсій
        control_conversions = control_data['converted'].sum()
        treatment_conversions = treatment_data['converted'].sum()

        # Кількість користувачів
        control_users = len(control_data)
        treatment_users = len(treatment_data)

        # Conversion rates
        p_control = control_conversions / control_users if control_users > 0 else 0
        p_treatment = treatment_conversions / treatment_users if treatment_users > 0 else 0

        # Pooled proportion
        p_pooled = (control_conversions + treatment_conversions) / (control_users + treatment_users) if (control_users + treatment_users) > 0 else 0

        # Standard error
        se = np.sqrt(p_pooled * (1 - p_pooled) * (1 / control_users + 1 / treatment_users)) if (control_users > 0 and treatment_users > 0) else 0

        # Z-statistic
        z_stat = (p_treatment - p_control) / se if se > 0 else 0

        # P-value (two-tailed)
        p_value = 2 * (1 - stats.norm.cdf(abs(z_stat))) if se > 0 else 1.0

        # Confidence interval (95%)
        ci_lower = (p_treatment - p_control) - 1.96 * se
        ci_upper = (p_treatment - p_control) + 1.96 * se

        # Lift
        lift = ((p_treatment - p_control) / p_control * 100) if p_control > 0 else 0

        return {
            'control_conversion_rate': p_control * 100,
            'treatment_conversion_rate': p_treatment * 100,
            'lift_percentage': lift,
            'z_statistic': z_stat,
            'p_value': p_value,
            'statistically_significant': p_value < 0.05,
            'confidence_interval_95': (ci_lower * 100, ci_upper * 100)
        }

    def t_test_revenue(self, control_data, treatment_data):
        """
        T-test to compare mean revenue

        Args:
            control_data: control group data

        treatment_data: test group data

        Returns:
            dict with test results
        """
        control_revenue = control_data['total_revenue'].fillna(0)
        treatment_revenue = treatment_data['total_revenue'].fillna(0)

        # T-test
        t_stat, p_value = stats.ttest_ind(treatment_revenue, control_revenue)

        # Means
        control_mean = control_revenue.mean()
        treatment_mean = treatment_revenue.mean()

        # Lift
        lift = ((treatment_mean - control_mean) / control_mean * 100) if control_mean != 0 else 0

        # Cohen's d (effect size)
        pooled_std = np.sqrt(
            ((len(control_revenue) - 1) * control_revenue.std() ** 2 +
             (len(treatment_revenue) - 1) * treatment_revenue.std() ** 2) /
            (len(control_revenue) + len(treatment_revenue) - 2)
        ) if (len(control_revenue) + len(treatment_revenue) - 2) > 0 else 0
        cohens_d = (treatment_mean - control_mean) / pooled_std if pooled_std != 0 else 0

        return {
            'control_avg_revenue': control_mean,
            'treatment_avg_revenue': treatment_mean,
            'lift_percentage': lift,
            't_statistic': t_stat if not np.isnan(t_stat) else 0,
            'p_value': p_value if not np.isnan(p_value) else 1.0,
            'statistically_significant': p_value < 0.05 if not np.isnan(p_value) else False,
            'cohens_d': cohens_d
        }

    def calculate_sample_size(self, baseline_rate, mde, alpha=0.05, power=0.8):
        """
        Calculating the required sample size

        Args:
            baseline_rate: baseline conversion rate (0-1)
            mde: minimum detectable effect (0-1)
            alpha: significance level (type I error)
            power: statistical power (1 - type II error)

        Returns:
            the required sample size for each group
        """
        # Z-scores
        z_alpha = stats.norm.ppf(1 - alpha / 2)
        z_beta = stats.norm.ppf(power)

        # Treatment rate
        treatment_rate = baseline_rate + mde

        # Standard deviations
        sd_control = np.sqrt(baseline_rate * (1 - baseline_rate))
        sd_treatment = np.sqrt(treatment_rate * (1 - treatment_rate))

        # Sample size per group
        n = ((z_alpha * np.sqrt(2 * baseline_rate * (1 - baseline_rate)) +
              z_beta * np.sqrt(sd_control ** 2 + sd_treatment ** 2)) / mde) ** 2

        return int(np.ceil(n))

    def sequential_testing(self, data, check_interval=100):
        """
        Sequential testing for early detection

        Args:
            data: DataFrame with A/B test data
            check_interval: check interval (number of users)

        Returns:
            DataFrame with sequential test results
        """
        results = []

        control_data = data[data['group_name'] == 'control']
        treatment_data = data[data['group_name'] == 'treatment']

        max_users = min(len(control_data), len(treatment_data))

        for n in range(check_interval, max_users, check_interval):
            control_subset = control_data.iloc[:n]
            treatment_subset = treatment_data.iloc[:n]

            test_result = self.z_test_proportions(control_subset, treatment_subset)

            results.append({
                'sample_size': n,
                'p_value': test_result['p_value'],
                'lift': test_result['lift_percentage'],
                'significant': test_result['statistically_significant']
            })

        return pd.DataFrame(results)

    def full_analysis_report(self, experiment_name='recommendation_engine_v2'):
        """
        Full A/B test report

        Args:
            experiment_name: experiment name

        Returns:
            dict with all metrics and test results
        """
        # Data acquisition
        data = self.get_ab_test_data(experiment_name)

        control_data = data[data['group_name'] == 'control']
        treatment_data = data[data['group_name'] == 'treatment']

        # Basic metrics
        conversion_metrics = self.calculate_conversion_rate(data)
        revenue_metrics = self.calculate_revenue_metrics(data)

        # Statistical tests
        conversion_test = self.z_test_proportions(control_data, treatment_data)
        revenue_test = self.t_test_revenue(control_data, treatment_data)

        # Sample size calculation
        baseline_rate = control_data['converted'].mean()
        recommended_sample_size = self.calculate_sample_size(
            baseline_rate=baseline_rate,
            mde=0.02  # 2% minimum detected effect
        )

        return {
            'experiment_name': experiment_name,
            'sample_sizes': {
                'control': len(control_data),
                'treatment': len(treatment_data)
            },
            'recommended_sample_size': recommended_sample_size,
            'conversion_metrics': conversion_metrics,
            'revenue_metrics': revenue_metrics,
            'conversion_test': conversion_test,
            'revenue_test': revenue_test
        }


if __name__ == "__main__":
    analyzer = ABTestAnalyzer()

    print("Conducting a full A/B test analysis...")
    report = analyzer.full_analysis_report()

    print("\n" + "=" * 60)
    print(f"A/BTEST REPORT: {report['experiment_name']}")
    print("=" * 60)

    print("\n📊 SAMPLE SIZES:")
    print(f"Control: {report['sample_sizes']['control']}")
    print(f"Treatment: {report['sample_sizes']['treatment']}")
    print(f"Recommended size: {report['recommended_sample_size']}")

    print("\n📈 CONVERSION RATE:")
    print(report['conversion_metrics'])

    print("\n💰 REVENUE METRICS:")
    print(report['revenue_metrics'])

    print("\n🧪 CONVERSION TEST:")
    for key, value in report['conversion_test'].items():
        print(f"{key}: {value}")

    print("\n💵 REVENUE TEST:")
    for key, value in report['revenue_test'].items():
        print(f"{key}: {value}")

