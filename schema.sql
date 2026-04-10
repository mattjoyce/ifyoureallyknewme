CREATE TABLE sources (
    id TEXT PRIMARY KEY,
    created_at TEXT
, description TEXT, content_path TEXT, title TEXT);
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
CREATE INDEX idx_processing_log_queue ON processing_log(queue_id);
CREATE INDEX idx_processing_log_processor ON processing_log(processor);
CREATE INDEX idx_processing_log_status ON processing_log(status);
CREATE TABLE sessions (
    id TEXT PRIMARY KEY,
    title TEXT,
    description TEXT NULL,
    created_at TEXT,
    metadata JSON NULL
, source_id TEXT, content_type TEXT, author text, lifestage text, file_path text);
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
CREATE TABLE analysis_queue (
    id TEXT PRIMARY KEY,
    author TEXT NOT NULL,
    lifestage TEXT NOT NULL,
    type TEXT NOT NULL DEFAULT 'document',
    priority INTEGER DEFAULT 0,
    status TEXT DEFAULT 'pending' CHECK(status IN ('pending', 'processing', 'completed', 'failed', 'cancelled')),
    created_at TEXT NOT NULL,
    source_id TEXT
);
