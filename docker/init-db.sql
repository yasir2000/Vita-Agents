-- Initialize Vita Agents Database
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Create schemas
CREATE SCHEMA IF NOT EXISTS agents;
CREATE SCHEMA IF NOT EXISTS workflows;
CREATE SCHEMA IF NOT EXISTS audit;

-- Agents table
CREATE TABLE IF NOT EXISTS agents.agent_instances (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    type VARCHAR(100) NOT NULL,
    status VARCHAR(50) NOT NULL,
    config JSONB,
    capabilities JSONB,
    metrics JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Tasks table
CREATE TABLE IF NOT EXISTS agents.tasks (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    task_id VARCHAR(255) UNIQUE NOT NULL,
    agent_id VARCHAR(255) NOT NULL,
    task_type VARCHAR(100) NOT NULL,
    parameters JSONB,
    status VARCHAR(50) NOT NULL,
    result JSONB,
    error_message TEXT,
    execution_time_ms INTEGER,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    FOREIGN KEY (agent_id) REFERENCES agents.agent_instances(agent_id)
);

-- Workflows table
CREATE TABLE IF NOT EXISTS workflows.definitions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    workflow_id VARCHAR(255) UNIQUE NOT NULL,
    name VARCHAR(255) NOT NULL,
    description TEXT,
    steps JSONB NOT NULL,
    metadata JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Workflow executions table
CREATE TABLE IF NOT EXISTS workflows.executions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    execution_id VARCHAR(255) UNIQUE NOT NULL,
    workflow_id VARCHAR(255) NOT NULL,
    status VARCHAR(50) NOT NULL,
    input_data JSONB,
    step_results JSONB,
    current_step VARCHAR(255),
    error_message TEXT,
    started_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    completed_at TIMESTAMP WITH TIME ZONE,
    FOREIGN KEY (workflow_id) REFERENCES workflows.definitions(workflow_id)
);

-- Audit logs table
CREATE TABLE IF NOT EXISTS audit.logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    data_type VARCHAR(100),
    details JSONB,
    user_id VARCHAR(255),
    ip_address INET,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (agent_id) REFERENCES agents.agent_instances(agent_id)
);

-- Healthcare data processing logs (HIPAA compliance)
CREATE TABLE IF NOT EXISTS audit.healthcare_logs (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    agent_id VARCHAR(255),
    action VARCHAR(100) NOT NULL,
    resource_type VARCHAR(100),
    resource_id VARCHAR(255),
    patient_id VARCHAR(255), -- Encrypted or hashed
    access_reason VARCHAR(255),
    compliance_flags JSONB,
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    FOREIGN KEY (agent_id) REFERENCES agents.agent_instances(agent_id)
);

-- Indexes for performance
CREATE INDEX IF NOT EXISTS idx_tasks_agent_id ON agents.tasks(agent_id);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON agents.tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_created_at ON agents.tasks(created_at);
CREATE INDEX IF NOT EXISTS idx_executions_workflow_id ON workflows.executions(workflow_id);
CREATE INDEX IF NOT EXISTS idx_executions_status ON workflows.executions(status);
CREATE INDEX IF NOT EXISTS idx_audit_logs_agent_id ON audit.logs(agent_id);
CREATE INDEX IF NOT EXISTS idx_audit_logs_created_at ON audit.logs(created_at);
CREATE INDEX IF NOT EXISTS idx_healthcare_logs_patient_id ON audit.healthcare_logs(patient_id);
CREATE INDEX IF NOT EXISTS idx_healthcare_logs_created_at ON audit.healthcare_logs(created_at);

-- Create user for application
DO $$
BEGIN
    IF NOT EXISTS (SELECT 1 FROM pg_user WHERE usename = 'vita_app') THEN
        CREATE USER vita_app WITH PASSWORD 'vita_app_password';
    END IF;
END
$$;

-- Grant permissions
GRANT USAGE ON SCHEMA agents TO vita_app;
GRANT USAGE ON SCHEMA workflows TO vita_app;
GRANT USAGE ON SCHEMA audit TO vita_app;

GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA agents TO vita_app;
GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA workflows TO vita_app;
GRANT SELECT, INSERT ON ALL TABLES IN SCHEMA audit TO vita_app;

-- Grant sequence permissions
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA agents TO vita_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA workflows TO vita_app;
GRANT USAGE, SELECT ON ALL SEQUENCES IN SCHEMA audit TO vita_app;