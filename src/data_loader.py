"""
Downloading data for e-commerce A/B testing
"""
import pandas as pd
import numpy as np
from sqlalchemy import create_engine, text, update, MetaData
import os
from dotenv import load_dotenv
from datetime import datetime, timedelta
import uuid
import random

load_dotenv()

class DataLoader:
    """Class for loading data from CSV into database"""

    def __init__(self, db_connection_string=None, csv_path=os.path.join(os.path.dirname(os.path.dirname(__file__)), 'data', 'raw', 'ecommerce_transactions.csv')):
        if db_connection_string is None:
            db_connection_string = os.getenv(
                'DATABASE_URL',
                'postgresql://ab_test_user:1234@localhost:5432/ecommerce_ab_test?client_encoding=utf8'
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
-- Database Schema for E-Commerce A/B Testing

-- Users Table
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
        # Clear all tables before loading to avoid conflicts
        truncate_sql = """
TRUNCATE TABLE ab_test_metrics CASCADE;
TRUNCATE TABLE events CASCADE;
TRUNCATE TABLE transactions CASCADE;
TRUNCATE TABLE products CASCADE;
TRUNCATE TABLE user_assignments CASCADE;
TRUNCATE TABLE experiment_groups CASCADE;
TRUNCATE TABLE users CASCADE;
        """
        with self.engine.connect() as conn:
            conn.execute(text(truncate_sql))
            conn.commit()
        print("Cleared all tables before loading new data.")

        # CSV reading (replace with real if exists; here fake for simulation)
        # df = pd.read_csv(self.csv_path)
        # For simulation:
        num_transactions = 50000
        # Random num_buyers pool: between 5000-20000 for repeat rate ~2.5-10 trans/user
        buyer_pool_size = random.randint(5000, 20000)
        fake_data = {
            'transaction_id': range(1, num_transactions + 1),
            'user_name': [f'user_{random.randint(1, buyer_pool_size)}' for _ in range(num_transactions)],  # Random assignment to users
            'age': np.random.randint(18, 70, num_transactions),
            'country': np.random.choice(['USA', 'UK', 'Germany', 'France', 'Canada'], num_transactions),
            'category': np.random.choice(['Electronics', 'Books', 'Clothing', 'Home'], num_transactions),
            'total_amount': np.random.uniform(10, 500, num_transactions),
            'payment_method': np.random.choice(['Credit Card', 'PayPal', 'Debit Card'], num_transactions),
            'transaction_date': [datetime.now() - timedelta(days=random.randint(1, 365)) for _ in range(num_transactions)]
        }
        df = pd.DataFrame(fake_data)
        print(f"Loaded {len(df)} transactions from CSV (simulated).")

        # Date processing
        df['transaction_date'] = pd.to_datetime(df['transaction_date'])

        # Unique buyers
        unique_buyers = df[['user_name', 'age', 'country']].drop_duplicates().reset_index(drop=True)
        num_buyers = len(unique_buyers)
        unique_buyers['user_uuid'] = [str(uuid.uuid4()) for _ in range(num_buyers)]

        # Registration date for buyers
        min_dates = df.groupby('user_name')['transaction_date'].min().reset_index()
        unique_buyers = unique_buyers.merge(min_dates, on='user_name')
        unique_buyers['registration_date'] = unique_buyers['transaction_date'] - pd.to_timedelta(np.random.randint(1, 366, num_buyers), unit='D')
        unique_buyers = unique_buyers.drop(columns=['transaction_date'])

        # Device type for buyers
        unique_buyers['device_type'] = np.random.choice(['mobile', 'desktop', 'tablet'], num_buyers)

        # Non-converted users (4x buyers)
        num_non_buyers = num_buyers * 4
        non_buyers = pd.DataFrame({
            'user_uuid': [str(uuid.uuid4()) for _ in range(num_non_buyers)],
            'registration_date': [datetime.now() - timedelta(days=random.randint(1, 365)) for _ in range(num_non_buyers)],
            'country': np.random.choice(['USA', 'UK', 'Germany', 'France', 'Canada'], num_non_buyers),
            'device_type': np.random.choice(['mobile', 'desktop', 'tablet'], num_non_buyers)
        })

        # All users (without user_id, let DB generate)
        all_users = pd.concat([unique_buyers[['user_uuid', 'registration_date', 'country', 'device_type']], non_buyers], ignore_index=True)
        all_users.to_sql('users', self.engine, if_exists='append', index=False)
        print(f"Inserted {len(all_users)} users (buyers: {num_buyers}, non-buyers: {num_non_buyers}).")

        # Get generated user_ids
        users_generated = pd.read_sql("SELECT user_id, user_uuid FROM users", self.engine)
        user_uuid_to_id = dict(zip(users_generated['user_uuid'], users_generated['user_id']))

        # Map user_id to buyers (for transactions)
        unique_buyers['user_id'] = unique_buyers['user_uuid'].map(user_uuid_to_id)
        user_name_to_id = dict(zip(unique_buyers['user_name'], unique_buyers['user_id']))
        df['user_id'] = df['user_name'].map(user_name_to_id)

        # Products (without product_id, let DB generate)
        df['product_name'] = 'Product ' + df['category'] + ' ' + df['transaction_id'].astype(str)
        df['price'] = df['total_amount']  # Assume quantity=1
        df['quantity'] = 1
        df['session_id'] = [str(uuid.uuid4()) for _ in range(len(df))]

        products_df = df[['product_name', 'category', 'price']].drop_duplicates()
        products_df.to_sql('products', self.engine, if_exists='append', index=False)
        print(f"Inserted {len(products_df)} products.")

        # Get generated product_ids
        products_generated = pd.read_sql("SELECT product_id, product_name FROM products", self.engine)
        product_name_to_id = dict(zip(products_generated['product_name'], products_generated['product_id']))
        df['product_id'] = df['product_name'].map(product_name_to_id)

        # Transactions (without transaction_id)
        trans_df = df[['user_id', 'product_id', 'transaction_date', 'quantity', 'total_amount', 'session_id']]
        trans_df.to_sql('transactions', self.engine, if_exists='append', index=False)
        print(f"Inserted {len(trans_df)} transactions.")

        # Events with drop-offs
        events = []
        product_ids = pd.read_sql("SELECT product_id FROM products", self.engine)['product_id'].tolist()

        # For buyers (from transactions)
        for _, row in df.iterrows():
            base_time = row['transaction_date']
            session_id = row['session_id']
            user_id = row['user_id']
            product_id = row['product_id']

            # View (always)
            events.append({
                'user_id': user_id,
                'event_type': 'view',
                'product_id': product_id,
                'event_timestamp': base_time - timedelta(minutes=random.randint(10, 30)),
                'session_id': session_id
            })

            # Add to cart (90%)
            if random.random() < 0.9:
                events.append({
                    'user_id': user_id,
                    'event_type': 'add_to_cart',
                    'product_id': product_id,
                    'event_timestamp': base_time - timedelta(minutes=random.randint(5, 10)),
                    'session_id': session_id
                })

            # Checkout (70%)
            if random.random() < 0.7:
                events.append({
                    'user_id': user_id,
                    'event_type': 'checkout',
                    'product_id': product_id,
                    'event_timestamp': base_time - timedelta(minutes=random.randint(1, 5)),
                    'session_id': session_id
                })

            # Purchase (always for buyers)
            events.append({
                'user_id': user_id,
                'event_type': 'purchase',
                'product_id': product_id,
                'event_timestamp': base_time,
                'session_id': session_id
            })

        # For non-buyers (users without transactions)
        non_buyers_query = """
SELECT u.user_id, u.user_uuid, u.registration_date
FROM users u
LEFT JOIN transactions t ON u.user_id = t.user_id
WHERE t.user_id IS NULL
        """
        non_buyers_df = pd.read_sql(non_buyers_query, self.engine)
        for _, non_buyer in non_buyers_df.iterrows():
            num_sessions = random.randint(1, 5)
            for _ in range(num_sessions):
                session_id = str(uuid.uuid4())
                base_time = non_buyer['registration_date'] + timedelta(days=random.randint(1, 180))
                product_id = random.choice(product_ids)

                # View (always)
                events.append({
                    'user_id': non_buyer['user_id'],
                    'event_type': 'view',
                    'product_id': product_id,
                    'event_timestamp': base_time - timedelta(minutes=random.randint(10, 30)),
                    'session_id': session_id
                })

                # Add to cart (50%)
                if random.random() < 0.5:
                    events.append({
                        'user_id': non_buyer['user_id'],
                        'event_type': 'add_to_cart',
                        'product_id': product_id,
                        'event_timestamp': base_time - timedelta(minutes=random.randint(5, 10)),
                        'session_id': session_id
                    })

                # Checkout (30%)
                if random.random() < 0.3:
                    events.append({
                        'user_id': non_buyer['user_id'],
                        'event_type': 'checkout',
                        'product_id': product_id,
                        'event_timestamp': base_time - timedelta(minutes=random.randint(1, 5)),
                        'session_id': session_id
                    })

        events_df = pd.DataFrame(events)
        events_df.to_sql('events', self.engine, if_exists='append', index=False)
        print(f"Generated and inserted {len(events_df)} events with drop-offs.")

        # Experimental groups (without group_id)
        groups_df = pd.DataFrame([
            {'group_name': 'control', 'description': 'Control group no changes'},
            {'group_name': 'treatment', 'description': 'Test group with new recommendation system'}
        ])
        groups_df.to_sql('experiment_groups', self.engine, if_exists='append', index=False)
        print("Experimental groups created.")

        # Assign users to groups (random 50/50)
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

        # Simulate treatment effect (+10% revenue for treatment)
        metadata = MetaData()
        metadata.reflect(self.engine)
        transactions_table = metadata.tables['transactions']

        treatment_users = users[users['group_id'] == treatment_id]['user_id'].tolist()
        if treatment_users:
            stmt = update(transactions_table).where(transactions_table.c.user_id.in_(treatment_users)).values(total_amount=transactions_table.c.total_amount * 1.1)
            with self.engine.connect() as conn:
                conn.execute(stmt)
                conn.commit()
        print("Simulated treatment effect: +10% revenue for treatment group.")

    def run_full_load(self):
        """
        Full process: initialization + data loading
        """
        self.initialize_database()
        self.load_data()
        print("Data loading complete!")

if __name__ == "__main__":
    loader = DataLoader()
    loader.run_full_load()