PRAGMA foreign_keys = ON;

DROP TABLE IF EXISTS reviews;
DROP TABLE IF EXISTS tasters;
DROP TABLE IF EXISTS wineries;
DROP TABLE IF EXISTS varieties;
DROP TABLE IF EXISTS locations;

CREATE TABLE tasters (
    taster_id INTEGER PRIMARY KEY,
    taster_name TEXT NOT NULL UNIQUE,
    twitter_handle TEXT
);

CREATE TABLE wineries (
    winery_id INTEGER PRIMARY KEY,
    winery TEXT NOT NULL UNIQUE
);

CREATE TABLE varieties (
    variety_id INTEGER PRIMARY KEY,
    variety TEXT NOT NULL UNIQUE
);

CREATE TABLE locations (
    location_id INTEGER PRIMARY KEY,
    country TEXT,
    province TEXT,
    region_1 TEXT,
    region_2 TEXT
);

CREATE TABLE reviews (
    review_id INTEGER PRIMARY KEY,
    title TEXT,
    designation TEXT,
    description TEXT NOT NULL,
    points INTEGER CHECK (points BETWEEN 80 AND 100),
    price REAL,
    source TEXT CHECK (source IN ('130k', '150k')),
    taster_id INTEGER REFERENCES tasters(taster_id),
    winery_id INTEGER REFERENCES wineries(winery_id),
    variety_id INTEGER REFERENCES varieties(variety_id),
    location_id INTEGER REFERENCES locations(location_id)
);

CREATE INDEX idx_reviews_points ON reviews(points);
CREATE INDEX idx_reviews_price ON reviews(price);
CREATE INDEX idx_reviews_source ON reviews(source);
CREATE INDEX idx_reviews_taster ON reviews(taster_id);
CREATE INDEX idx_reviews_winery ON reviews(winery_id);
CREATE INDEX idx_reviews_variety ON reviews(variety_id);
CREATE INDEX idx_reviews_location ON reviews(location_id);
