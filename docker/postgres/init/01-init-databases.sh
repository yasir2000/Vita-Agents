#!/bin/bash
set -e

# Create multiple databases
psql -v ON_ERROR_STOP=1 --username "$POSTGRES_USER" --dbname "$POSTGRES_DB" <<-EOSQL
    -- Create additional databases
    CREATE DATABASE vita_analytics;
    CREATE DATABASE vita_audit;
    
    -- Grant permissions
    GRANT ALL PRIVILEGES ON DATABASE vita_analytics TO $POSTGRES_USER;
    GRANT ALL PRIVILEGES ON DATABASE vita_audit TO $POSTGRES_USER;
    
    -- Create extensions for main database
    \c vita_agents;
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    CREATE EXTENSION IF NOT EXISTS "pgcrypto";
    CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
    
    -- Create extensions for analytics database
    \c vita_analytics;
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    CREATE EXTENSION IF NOT EXISTS "pg_stat_statements";
    
    -- Create extensions for audit database
    \c vita_audit;
    CREATE EXTENSION IF NOT EXISTS "uuid-ossp";
    CREATE EXTENSION IF NOT EXISTS "pgcrypto";
EOSQL

echo "Multiple databases created successfully!"