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
CREATE INDEX idx_transactions_user_date ON transactions(user_id, transaction_date);
CREATE INDEX idx_events_user_type ON events(user_id, event_type);
CREATE INDEX idx_user_assignments_experiment ON user_assignments(experiment_name, group_id);
CREATE INDEX idx_transactions_date ON transactions(transaction_date);
CREATE INDEX idx_events_timestamp ON events(event_timestamp);

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
