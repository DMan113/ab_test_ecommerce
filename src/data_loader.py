"""
Downloading data for e-commerce A/B testing
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import uuid

load_dotenv()

class DataLoader:
    """Class for loading data from CSV into database"""

    def __init__(self, db_connection_string=None, csv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw', 'ecommerce_transactions.csv')):
        if db_connection_string is None:
            db_connection_string = os.getenv(
                'DATABASE_URL',
                'postgresql://DB_USER:DB_PASSWORD@DB_HOST:DB_PORT/DB_NAME?client_encoding=utf8'
            )
        print(f"Connection string used: {db_connection_string}")
        print(f"The way to CSV: {csv_path}")
        self.engine = create_engine(db_connection_string)
        self.csv_path = csv_path

    def initialize_database(self):
        """
        Database initialization: creating tables, indexes, and views
        """
        schema_sql = """
-- Database schema for e-commerce A/B testing
-- Users table
CREATE TABLE IF NOT EXISTS users (
    user_id SERIAL PRIMARY KEY,
    user_uuid VARCHAR(100) UNIQUE NOT NULL,
    registration_date TIMESTAMP,
    country VARCHAR(50),
    device_type VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table of experimental groups
CREATE TABLE IF NOT EXISTS experiment_groups (
    group_id SERIAL PRIMARY KEY,
    group_name VARCHAR(50) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- User group assignment table
CREATE TABLE IF NOT EXISTS user_assignments (
    assignment_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    group_id INTEGER REFERENCES experiment_groups(group_id),
    experiment_name VARCHAR(100),
    assigned_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(user_id, experiment_name)
);

-- Product table
CREATE TABLE IF NOT EXISTS products (
    product_id SERIAL PRIMARY KEY,
    product_name VARCHAR(255),
    category VARCHAR(100),
    price DECIMAL(10, 2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Transaction table
CREATE TABLE IF NOT EXISTS transactions (
    transaction_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    product_id INTEGER REFERENCES products(product_id),
    transaction_date TIMESTAMP NOT NULL,
    quantity INTEGER,
    total_amount DECIMAL(10, 2),
    session_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Events table for funnel analysis
CREATE TABLE IF NOT EXISTS events (
    event_id SERIAL PRIMARY KEY,
    user_id INTEGER REFERENCES users(user_id),
    event_type VARCHAR(50) NOT NULL, -- 'view', 'add_to_cart', 'checkout', 'purchase'
    product_id INTEGER REFERENCES products(product_id),
    event_timestamp TIMESTAMP NOT NULL,
    session_id VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- A/B test metrics table
CREATE TABLE IF NOT EXISTS ab_test_metrics (
    metric_id SERIAL PRIMARY KEY,
    experiment_name VARCHAR(100),
    group_id INTEGER REFERENCES experiment_groups(group_id),
    metric_name VARCHAR(100), -- 'conversion_rate', 'avg_order_value', 'revenue_per_user'
    metric_value DECIMAL(12, 4),
    user_count INTEGER,
    calculation_date DATE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for query optimization
CREATE INDEX IF NOT EXISTS idx_transactions_user_date ON transactions(user_id, transaction_date);
CREATE INDEX IF NOT EXISTS idx_events_user_type ON events(user_id, event_type);
CREATE INDEX IF NOT EXISTS idx_user_assignments_experiment ON user_assignments(experiment_name, group_id);
CREATE INDEX IF NOT EXISTS idx_transactions_date ON transactions(transaction_date);
CREATE INDEX IF NOT EXISTS idx_events_timestamp ON events(event_timestamp);

-- View for quick access to cohorts
CREATE OR REPLACE VIEW cohort_analysis AS
SELECT
    u.user_id,
    u.user_uuid,
    DATE_TRUNC('month', u.registration_date) as cohort_month,
    ua.group_id,
    eg.group_name,
    COUNT(DISTINCT t.transaction_id) as transaction_count,
    SUM(t.total_amount) as total_revenue,
    MIN(t.transaction_date) as first_purchase_date
FROM users u
LEFT JOIN user_assignments ua ON u.user_id = ua.user_id
LEFT JOIN experiment_groups eg ON ua.group_id = eg.group_id
LEFT JOIN transactions t ON u.user_id = t.user_id
GROUP BY u.user_id, u.user_uuid, cohort_month, ua.group_id, eg.group_name;

-- View for conversion funnel
CREATE OR REPLACE VIEW conversion_funnel AS
SELECT
    ua.experiment_name,
    eg.group_name,
    COUNT(DISTINCT CASE WHEN e.event_type = 'view' THEN e.user_id END) as views,
    COUNT(DISTINCT CASE WHEN e.event_type = 'add_to_cart' THEN e.user_id END) as add_to_cart,
    COUNT(DISTINCT CASE WHEN e.event_type = 'checkout' THEN e.user_id END) as checkout,
    COUNT(DISTINCT CASE WHEN e.event_type = 'purchase' THEN e.user_id END) as purchases
FROM events e
JOIN user_assignments ua ON e.user_id = ua.user_id
JOIN experiment_groups eg ON ua.group_id = eg.group_id
GROUP BY ua.experiment_name, eg.group_name;
        """
        with self.engine.connect() as conn:
            conn.execute(text(schema_sql))
            conn.commit()
        print("The database is initialized with tables, indexes, and views.")

    def load_data(self):
        """
        Loading data from CSV, processing and inserting into the database
        """
        #  CSV reading
        df = pd.read_csv(self.csv_path)
        print(f"Loaded {len(df)} transactions from CSV.")

        # Renaming columns to match the schema
        df = df.rename(columns={
            'Transaction_ID': 'transaction_id',
            'User_Name': 'user_name',
            'Age': 'age',
            'Country': 'country',
            'Product_Category': 'category',
            'Purchase_Amount': 'total_amount',
            'Payment_Method': 'payment_method',
            'Transaction_Date': 'transaction_date'
        })

        # Date processing
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])

        # Generation of unique users
        unique_users = df[['user_name', 'age', 'country']].drop_duplicates().reset_index(drop=True)
        unique_users['user_id'] = unique_users.index + 1
        unique_users['user_uuid'] = [str(uuid.uuid4()) for _ in range(len(unique_users))]

        # Generation registration_date: min transaction_date per user - random 1-365 days
        min_dates = df.groupby('user_name')['transaction_date'].min().reset_index()
        unique_users = unique_users.merge(min_dates, on='user_name')
        unique_users['registration_date'] = unique_users['transaction_date'] - pd.to_timedelta(np.random.randint(1, 366, len(unique_users)), unit='D')
        unique_users = unique_users.drop(columns=['transaction_date'])

        # Generation device_type
        unique_users['device_type'] = np.random.choice(['mobile', 'desktop', 'tablet'], len(unique_users))

        # Preparing users_df
        users_df = unique_users[['user_id', 'user_uuid', 'registration_date', 'country', 'device_type']]
        users_df.to_sql('users', self.engine, if_exists='append', index=False)
        print(f"Inserted {len(users_df)} users.")

        # Mapping user_id back to df
        user_id_map = dict(zip(unique_users['user_name'], unique_users['user_id']))
        df['user_id'] = df['user_name'].map(user_id_map)

        # Product and transaction generation
        df['product_id'] = df.index + 1
        df['product_name'] = 'Product ' + df['category'] + ' ' + df['transaction_id'].astype(str)
        df['price'] = df['total_amount']  # Assume quantity=1
        df['quantity'] = 1
        df['session_id'] = [str(uuid.uuid4()) for _ in range(len(df))]

        # Підготовка products_df
        products_df = df[['product_id', 'product_name', 'category', 'price']].drop_duplicates()
        products_df.to_sql('products', self.engine, if_exists='append', index=False)
        print(f"Inserted {len(products_df)} products.")

        # Preparing trans_df
        trans_df = df[['transaction_id', 'user_id', 'product_id', 'transaction_date', 'quantity', 'total_amount', 'session_id']]
        trans_df.to_sql('transactions', self.engine, if_exists='append', index=False)
        print(f"Inserted {len(trans_df)} transactions.")

        # Transaction-based event generation (funnel simulation)
        events = []
        for _, row in df.iterrows():
            base_time = row['transaction_date']
            session_id = row['session_id']
            user_id = row['user_id']
            product_id = row['product_id']

            # View
            events.append({
                'user_id': user_id,
                'event_type': 'view',
                'product_id': product_id,
                'event_timestamp': base_time - timedelta(minutes=np.random.randint(10, 30)),
                'session_id': session_id
            })

            # Add to cart
            events.append({
                'user_id': user_id,
                'event_type': 'add_to_cart',
                'product_id': product_id,
                'event_timestamp': base_time - timedelta(minutes=np.random.randint(5, 10)),
                'session_id': session_id
            })

            # Checkout
            events.append({
                'user_id': user_id,
                'event_type': 'checkout',
                'product_id': product_id,
                'event_timestamp': base_time - timedelta(minutes=np.random.randint(1, 5)),
                'session_id': session_id
            })

            # Purchase
            events.append({
                'user_id': user_id,
                'event_type': 'purchase',
                'product_id': product_id,
                'event_timestamp': base_time,
                'session_id': session_id
            })

        events_df = pd.DataFrame(events)
        events_df.to_sql('events', self.engine, if_exists='append', index=False)
        print(f"Generated and inserted {len(events_df)} подій.")

        # Creating experimental groups
        groups_df = pd.DataFrame([
            {'group_name': 'control', 'description': 'Control group no changes'},
            {'group_name': 'treatment', 'description': 'Test group with new recommendation system'}
        ])
        groups_df.to_sql('experiment_groups', self.engine, if_exists='append', index=False)
        print("Experimental groups created.")

        # Assigning users to groups (random 50/50)
        users = pd.read_sql("SELECT user_id FROM users", self.engine)
        group_ids = pd.read_sql("SELECT group_id, group_name FROM experiment_groups", self.engine)

        control_id = group_ids[group_ids['group_name'] == 'control']['group_id'].iloc[0]
        treatment_id = group_ids[group_ids['group_name'] == 'treatment']['group_id'].iloc[0]

        users['group_id'] = np.random.choice([control_id, treatment_id], size=len(users), p=[0.5, 0.5])
        users['experiment_name'] = 'recommendation_engine_v2'

        users[['user_id', 'group_id', 'experiment_name']].to_sql(
            'user_assignments', self.engine, if_exists='append', index=False
        )
        print(f"Assigned {len(users)} users to groups.")

    def run_full_load(self):
        """
        Full process: initialization + data loading
        """
        self.initialize_database()
        self.load_data()
        print("Data loading complete!")

if __name__ == "__main__":
    # Example of use
    # First, download the dataset from Kaggle and specify the path to the CSV
    loader = DataLoader()  # Now the path is calculated automatically relative to src
    loader.run_full_load()