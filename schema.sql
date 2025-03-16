CREATE TABLE sources (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    type TEXT NOT NULL,
    created_at TEXT
);

CREATE TABLE analysis_queue (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    author TEXT NOT NULL,
    lifestage TEXT NOT NULL,
    filename TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'document',  -- document, qa, etc.
    priority INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'processing', 'completed', 'failed')),
    created_at TEXT NOT NULL,
    updated_at TEXT,
    retry_count INTEGER DEFAULT 0,
    last_attempt TEXT,
    error_log JSON,
    metadata JSON,  -- For additional type-specific data
    expert_status JSON,  -- Track each expert's status
    fact_status JSON,  -- Track each fact extractor's status
    dependencies JSON  -- List of queue items this depends on
);

-- Create index for efficient queue queries
CREATE INDEX idx_analysis_queue_status ON analysis_queue(status);
CREATE INDEX idx_analysis_queue_priority ON analysis_queue(priority);
CREATE INDEX idx_analysis_queue_type ON analysis_queue(type);

-- Create processing_log table for detailed tracking
CREATE TABLE processing_log (
    id TEXT PRIMARY KEY,
    queue_id TEXT NOT NULL,
    processor TEXT NOT NULL,  -- expert name or fact extractor
    status TEXT NOT NULL,
    started_at TEXT NOT NULL,
    completed_at TEXT,
    error_message TEXT,
    metadata JSON,
    FOREIGN KEY (queue_id) REFERENCES analysis_queue(id)
);

-- Create index for efficient log queries
CREATE INDEX idx_processing_log_queue ON processing_log(queue_id);
CREATE INDEX idx_processing_log_processor ON processing_log(processor);
CREATE INDEX idx_processing_log_status ON processing_log(status);

CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    title TEXT,
    description TEXT NULL,
    transcript_file TEXT NULL,
    created_at TEXT,
    metadata JSON NULL
);

CREATE TABLE qa_pairs (
    id TEXT PRIMARY KEY,
    session_id TEXT NOT NULL,
    question TEXT NOT NULL,
    answer TEXT NOT NULL,
    sequence INTEGER NOT NULL,
    created_at TEXT,
    embedding BLOB,
    FOREIGN KEY (session_id) REFERENCES sessions(id)
);

CREATE TABLE questions (
    id TEXT PRIMARY KEY,
    text TEXT NOT NULL,
    category TEXT,
    tags TEXT,
    created_at TEXT,
    embedding BLOB
);

CREATE TABLE domains (
    id TEXT PRIMARY KEY,
    name TEXT NOT NULL,
    weight REAL
);

CREATE TABLE question_domains (
    question_id TEXT,
    domain_id TEXT,
    relevance REAL,
    PRIMARY KEY (question_id, domain_id),
    FOREIGN KEY (question_id) REFERENCES questions(id),
    FOREIGN KEY (domain_id) REFERENCES domains(id)
);

CREATE TABLE knowledge_records (
    id TEXT PRIMARY KEY,
    type TEXT,
    domain TEXT,
    author TEXT,
    content JSON,
    created_at TEXT,
    version TEXT,
    embedding BLOB,
    consensus_id TEXT,
    session_id TEXT NULL,
    qa_id TEXT NULL,
    source_id TEXT NULL,
    keywords TEXT,
    FOREIGN KEY (qa_id) REFERENCES qa_pairs(id),
    FOREIGN KEY (session_id) REFERENCES sessions(id),
    FOREIGN KEY (source_id) REFERENCES sources(id)
);
