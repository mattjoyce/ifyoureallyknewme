-- ResetConsensus.sql
-- Script to reset consensus records in the KnowMe database
-- This script:
-- 1. Updates all knowledge records to remove their consensus_id references
-- 2. Deletes all consensus records (where author = 'ConsensusMaker')

-- Start a transaction for safety
BEGIN TRANSACTION;

-- First, update all knowledge records to reset consensus_id to NULL
UPDATE knowledge_records
SET consensus_id = NULL
WHERE consensus_id IS NOT NULL;

-- Then, delete all consensus records
DELETE FROM knowledge_records
WHERE author = 'ConsensusMaker'
AND type = 'consensus';

-- Commit the transaction if everything succeeded
COMMIT;

-- Optional verification queries (comment out in production use)
-- SELECT COUNT(*) FROM knowledge_records WHERE consensus_id IS NOT NULL;
-- SELECT COUNT(*) FROM knowledge_records WHERE author = 'ConsensusMaker' AND type = 'consensus';