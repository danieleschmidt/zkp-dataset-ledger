-- PostgreSQL initialization script for ZKP Dataset Ledger
-- This script sets up the database schema for PostgreSQL storage backend

BEGIN;

-- Create schema for ZKP ledger data
CREATE SCHEMA IF NOT EXISTS zkp_ledger;

-- Set search path
SET search_path TO zkp_ledger, public;

-- Create transactions table
CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    dataset_name VARCHAR(255) NOT NULL,
    transaction_hash CHAR(64) NOT NULL UNIQUE,
    merkle_root CHAR(64) NOT NULL,
    proof_data JSONB NOT NULL,
    dataset_hash CHAR(64) NOT NULL,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Metadata
    operation_type VARCHAR(50) NOT NULL DEFAULT 'notarize',
    dataset_size_bytes BIGINT,
    row_count INTEGER,
    column_count INTEGER,
    proof_generation_time_ms INTEGER,
    
    -- Indexing for performance
    CONSTRAINT valid_operation_type CHECK (operation_type IN ('notarize', 'transform', 'split', 'merge'))
);

-- Create indexes for common queries
CREATE INDEX IF NOT EXISTS idx_transactions_dataset_name ON transactions(dataset_name);
CREATE INDEX IF NOT EXISTS idx_transactions_hash ON transactions(transaction_hash);
CREATE INDEX IF NOT EXISTS idx_transactions_merkle_root ON transactions(merkle_root);
CREATE INDEX IF NOT EXISTS idx_transactions_created_at ON transactions(created_at);
CREATE INDEX IF NOT EXISTS idx_transactions_operation_type ON transactions(operation_type);

-- Create Merkle tree nodes table for efficient tree operations
CREATE TABLE IF NOT EXISTS merkle_nodes (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    node_hash CHAR(64) NOT NULL UNIQUE,
    parent_hash CHAR(64),
    left_child_hash CHAR(64),
    right_child_hash CHAR(64),
    level INTEGER NOT NULL,
    position INTEGER NOT NULL,
    is_leaf BOOLEAN NOT NULL DEFAULT false,
    transaction_id UUID REFERENCES transactions(id) ON DELETE CASCADE,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Ensure proper tree structure
    CONSTRAINT valid_level CHECK (level >= 0),
    CONSTRAINT valid_position CHECK (position >= 0)
);

-- Indexes for Merkle tree operations
CREATE INDEX IF NOT EXISTS idx_merkle_nodes_hash ON merkle_nodes(node_hash);
CREATE INDEX IF NOT EXISTS idx_merkle_nodes_parent ON merkle_nodes(parent_hash);
CREATE INDEX IF NOT EXISTS idx_merkle_nodes_level_position ON merkle_nodes(level, position);
CREATE INDEX IF NOT EXISTS idx_merkle_nodes_transaction ON merkle_nodes(transaction_id);

-- Create audit log table for compliance
CREATE TABLE IF NOT EXISTS audit_log (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    transaction_id UUID REFERENCES transactions(id) ON DELETE CASCADE,
    event_type VARCHAR(50) NOT NULL,
    event_data JSONB NOT NULL,
    user_id VARCHAR(255),
    ip_address INET,
    user_agent TEXT,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    
    -- Event type constraints
    CONSTRAINT valid_event_type CHECK (event_type IN ('create', 'verify', 'query', 'export', 'delete'))
);

-- Index for audit queries
CREATE INDEX IF NOT EXISTS idx_audit_log_transaction ON audit_log(transaction_id);
CREATE INDEX IF NOT EXISTS idx_audit_log_event_type ON audit_log(event_type);
CREATE INDEX IF NOT EXISTS idx_audit_log_created_at ON audit_log(created_at);
CREATE INDEX IF NOT EXISTS idx_audit_log_user ON audit_log(user_id);

-- Create proof verification cache table
CREATE TABLE IF NOT EXISTS proof_verification_cache (
    proof_hash CHAR(64) PRIMARY KEY,
    is_valid BOOLEAN NOT NULL,
    verification_time_ms INTEGER NOT NULL,
    verifier_version VARCHAR(50) NOT NULL,
    cached_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    expires_at TIMESTAMP WITH TIME ZONE DEFAULT (NOW() + INTERVAL '24 hours'),
    
    -- Performance constraints
    CONSTRAINT positive_verification_time CHECK (verification_time_ms > 0)
);

-- Index for cache expiration cleanup
CREATE INDEX IF NOT EXISTS idx_proof_cache_expires ON proof_verification_cache(expires_at);

-- Create stored procedures for common operations

-- Function to get audit trail for a dataset
CREATE OR REPLACE FUNCTION get_dataset_audit_trail(dataset_name_param VARCHAR(255))
RETURNS TABLE(
    transaction_id UUID,
    transaction_hash CHAR(64),
    operation_type VARCHAR(50),
    created_at TIMESTAMP WITH TIME ZONE,
    dataset_size_bytes BIGINT,
    row_count INTEGER
) AS $$
BEGIN
    RETURN QUERY
    SELECT 
        t.id,
        t.transaction_hash,
        t.operation_type,
        t.created_at,
        t.dataset_size_bytes,
        t.row_count
    FROM transactions t
    WHERE t.dataset_name = dataset_name_param
    ORDER BY t.created_at ASC;
END;
$$ LANGUAGE plpgsql;

-- Function to verify Merkle tree integrity
CREATE OR REPLACE FUNCTION verify_merkle_tree_integrity(root_hash_param CHAR(64))
RETURNS BOOLEAN AS $$
DECLARE
    node_count INTEGER;
    expected_leaves INTEGER;
    actual_leaves INTEGER;
BEGIN
    -- Count total nodes
    SELECT COUNT(*) INTO node_count 
    FROM merkle_nodes 
    WHERE node_hash IN (
        WITH RECURSIVE tree_nodes AS (
            SELECT node_hash, left_child_hash, right_child_hash, level
            FROM merkle_nodes 
            WHERE node_hash = root_hash_param
            
            UNION ALL
            
            SELECT m.node_hash, m.left_child_hash, m.right_child_hash, m.level
            FROM merkle_nodes m
            INNER JOIN tree_nodes t ON (m.node_hash = t.left_child_hash OR m.node_hash = t.right_child_hash)
        )
        SELECT node_hash FROM tree_nodes
    );
    
    -- Count leaf nodes
    SELECT COUNT(*) INTO actual_leaves
    FROM merkle_nodes
    WHERE is_leaf = true
    AND node_hash IN (
        WITH RECURSIVE tree_nodes AS (
            SELECT node_hash, left_child_hash, right_child_hash, level
            FROM merkle_nodes 
            WHERE node_hash = root_hash_param
            
            UNION ALL
            
            SELECT m.node_hash, m.left_child_hash, m.right_child_hash, m.level
            FROM merkle_nodes m
            INNER JOIN tree_nodes t ON (m.node_hash = t.left_child_hash OR m.node_hash = t.right_child_hash)
        )
        SELECT node_hash FROM tree_nodes
    );
    
    -- Basic integrity check: if we have n leaves, we should have 2n-1 total nodes
    RETURN (node_count = 2 * actual_leaves - 1) AND (actual_leaves > 0);
END;
$$ LANGUAGE plpgsql;

-- Function to clean expired cache entries
CREATE OR REPLACE FUNCTION cleanup_expired_cache()
RETURNS INTEGER AS $$
DECLARE
    deleted_count INTEGER;
BEGIN
    DELETE FROM proof_verification_cache
    WHERE expires_at < NOW();
    
    GET DIAGNOSTICS deleted_count = ROW_COUNT;
    RETURN deleted_count;
END;
$$ LANGUAGE plpgsql;

-- Create trigger to update updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_transactions_updated_at
    BEFORE UPDATE ON transactions
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Create trigger for audit logging
CREATE OR REPLACE FUNCTION log_transaction_changes()
RETURNS TRIGGER AS $$
BEGIN
    IF TG_OP = 'INSERT' THEN
        INSERT INTO audit_log (transaction_id, event_type, event_data)
        VALUES (NEW.id, 'create', to_jsonb(NEW));
        RETURN NEW;
    ELSIF TG_OP = 'UPDATE' THEN
        INSERT INTO audit_log (transaction_id, event_type, event_data)
        VALUES (NEW.id, 'update', jsonb_build_object('old', to_jsonb(OLD), 'new', to_jsonb(NEW)));
        RETURN NEW;
    ELSIF TG_OP = 'DELETE' THEN
        INSERT INTO audit_log (transaction_id, event_type, event_data)
        VALUES (OLD.id, 'delete', to_jsonb(OLD));
        RETURN OLD;
    END IF;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER transaction_audit_trigger
    AFTER INSERT OR UPDATE OR DELETE ON transactions
    FOR EACH ROW
    EXECUTE FUNCTION log_transaction_changes();

-- Create views for common queries

-- View for transaction summary
CREATE OR REPLACE VIEW transaction_summary AS
SELECT 
    dataset_name,
    COUNT(*) as transaction_count,
    MIN(created_at) as first_transaction,
    MAX(created_at) as last_transaction,
    SUM(dataset_size_bytes) as total_size_bytes,
    AVG(proof_generation_time_ms) as avg_proof_time_ms
FROM transactions
GROUP BY dataset_name;

-- View for recent activity
CREATE OR REPLACE VIEW recent_activity AS
SELECT 
    t.dataset_name,
    t.operation_type,
    t.created_at,
    t.transaction_hash,
    t.dataset_size_bytes,
    t.row_count
FROM transactions t
ORDER BY t.created_at DESC
LIMIT 100;

-- Create extension for better performance (if available)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
CREATE EXTENSION IF NOT EXISTS "pg_trgm";

-- Grant permissions to the application user
GRANT USAGE ON SCHEMA zkp_ledger TO zkpuser;
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA zkp_ledger TO zkpuser;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA zkp_ledger TO zkpuser;
GRANT EXECUTE ON ALL FUNCTIONS IN SCHEMA zkp_ledger TO zkpuser;

-- Set default privileges for future tables
ALTER DEFAULT PRIVILEGES IN SCHEMA zkp_ledger
    GRANT ALL PRIVILEGES ON TABLES TO zkpuser;
ALTER DEFAULT PRIVILEGES IN SCHEMA zkp_ledger
    GRANT ALL PRIVILEGES ON SEQUENCES TO zkpuser;
ALTER DEFAULT PRIVILEGES IN SCHEMA zkp_ledger
    GRANT EXECUTE ON FUNCTIONS TO zkpuser;

COMMIT;

-- Insert sample data for testing (only in development)
DO $$
BEGIN
    IF current_setting('server_version_num')::int >= 120000 AND 
       EXISTS (SELECT 1 FROM information_schema.schemata WHERE schema_name = 'zkp_ledger') THEN
        
        -- Only insert if we're in a development environment
        -- (This could be controlled by environment variables in a real setup)
        INSERT INTO transactions (
            dataset_name, 
            transaction_hash, 
            merkle_root, 
            proof_data, 
            dataset_hash,
            operation_type,
            dataset_size_bytes,
            row_count,
            column_count
        ) VALUES (
            'sample-dataset-v1',
            '1234567890abcdef1234567890abcdef1234567890abcdef1234567890abcdef',
            'fedcba0987654321fedcba0987654321fedcba0987654321fedcba0987654321',
            '{"proof": "sample_proof_data", "public_inputs": ["1000", "5"]}',
            'abcdef1234567890abcdef1234567890abcdef1234567890abcdef1234567890',
            'notarize',
            1024576,
            1000,
            5
        ) ON CONFLICT DO NOTHING;
        
    END IF;
END $$;