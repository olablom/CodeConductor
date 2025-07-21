-- CodeConductor Database Initialization
-- This script sets up the initial database schema

-- Create tables for AI/ML data
CREATE TABLE IF NOT EXISTS q_learning_states (
    id SERIAL PRIMARY KEY,
    state_key VARCHAR(255) UNIQUE NOT NULL,
    q_values JSONB NOT NULL DEFAULT '{}',
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS bandit_arms (
    id SERIAL PRIMARY KEY,
    arm_name VARCHAR(255) UNIQUE NOT NULL,
    feature_vector JSONB NOT NULL DEFAULT '[]',
    reward_history JSONB NOT NULL DEFAULT '[]',
    total_pulls INTEGER DEFAULT 0,
    total_reward FLOAT DEFAULT 0.0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS policy_decisions (
    id SERIAL PRIMARY KEY,
    code_hash VARCHAR(64) NOT NULL,
    risk_level VARCHAR(50) NOT NULL,
    policy_compliant BOOLEAN NOT NULL,
    analysis_result JSONB NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Create indexes for performance
CREATE INDEX IF NOT EXISTS idx_q_learning_states_key ON q_learning_states(state_key);
CREATE INDEX IF NOT EXISTS idx_bandit_arms_name ON bandit_arms(arm_name);
CREATE INDEX IF NOT EXISTS idx_policy_decisions_hash ON policy_decisions(code_hash);

-- Insert some initial data
INSERT INTO q_learning_states (state_key, q_values) VALUES 
    ('default_state', '{"action1": 0.0, "action2": 0.0, "action3": 0.0}')
ON CONFLICT (state_key) DO NOTHING;

INSERT INTO bandit_arms (arm_name, feature_vector) VALUES 
    ('conservative_strategy', '[0.5, 0.3, 0.7, 0.4, 0.6]'),
    ('experimental_strategy', '[0.8, 0.9, 0.6, 0.7, 0.8]'),
    ('hybrid_approach', '[0.6, 0.7, 0.5, 0.8, 0.7]')
ON CONFLICT (arm_name) DO NOTHING;
