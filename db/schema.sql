-- schema.sql
-- Relational database schema for storing fintech app reviews

-- Drop tables if they exist to allow clean replication
DROP TABLE IF EXISTS reviews;
DROP TABLE IF EXISTS banks;

-- 1. Banks Table: Stores metadata about the banks
CREATE TABLE banks (
    bank_id SERIAL PRIMARY KEY,
    bank_name VARCHAR(100) UNIQUE NOT NULL,
    app_name VARCHAR(100) NOT NULL
);

-- 2. Reviews Table: Stores the scraped and processed review data
CREATE TABLE reviews (
    review_id SERIAL PRIMARY KEY,
    bank_id INT REFERENCES banks(bank_id) ON DELETE CASCADE,
    review_text TEXT NOT NULL,
    rating INT NOT NULL,
    review_date DATE NOT NULL,
    sentiment_label VARCHAR(20),
    sentiment_score NUMERIC(5, 4),
    identified_theme VARCHAR(100),
    source VARCHAR(50) NOT NULL
);
