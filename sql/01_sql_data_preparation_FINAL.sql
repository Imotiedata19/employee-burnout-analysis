/* ============================================================================
   EMPLOYEE BURNOUT ANALYSIS — PHASE 1: SQL DATA PREPARATION & EXPLORATION
   ============================================================================

Engine       : PostgreSQL 13+

Source table : employee_burnout_raw
Raw rows     : 22,750

Prerequisite:
The employee_burnout_raw table is already loaded in PostgreSQL with
standardized snake_case column names.

Current schema:
    employee_id              VARCHAR
    date_of_joining          DATE
    gender                   VARCHAR
    company_type             VARCHAR
    wfh_setup_available      TEXT
    designation              INTEGER
    resource_allocation      NUMERIC
    mental_fatigue_score     NUMERIC
    burn_rate                NUMERIC

This script:
    1. validates the raw table,
    2. checks missing values, duplicates, inconsistencies, and outliers,
    3. creates employee_burnout_clean,
    4. removes rows missing burn_rate, resource_allocation, or
       mental_fatigue_score,
    5. derives joining_month and risk,
    6. produces the analytical table used by Python and Power BI.

Expected output:
    employee_burnout_clean — 18,590 rows, 11 columns.
   ============================================================================ */


-- ============================================================================
-- STEP 1 — INSPECT THE DATASET
-- Purpose: confirm the table loaded correctly and check overall size.
-- ============================================================================
SELECT COUNT(*) AS total_rows
FROM employee_burnout_raw;
-- Expected: 22,750 rows


-- ============================================================================
-- STEP 1.5 — VERIFY STANDARDIZED COLUMN HEADERS
-- Purpose: confirm that employee_burnout_raw already uses snake_case columns.
-- The PostgreSQL table was imported with standardized column names,
-- so no additional RENAME COLUMN operations are required.
-- ============================================================================

SELECT
    column_name,
    data_type
FROM information_schema.columns
WHERE table_schema = 'public'
  AND table_name = 'employee_burnout_raw'
ORDER BY ordinal_position;


-- ============================================================================
-- STEP 1.6 — VERIFY employee_id FORMAT
-- Purpose: earlier runs of this pipeline used a raw source file where
-- Employee ID arrived as mangled UTF-16-hex strings (e.g.
-- "fffe32003000360033003200") and required a decode step here. The current
-- source file (employee_burnout_postgres_fixed.csv) already provides clean,
-- sequential IDs like "EMP000001", so no transformation is needed — this
-- step only verifies that remains true, so the pipeline fails loudly if a
-- future source file regresses to the old mangled format.
-- ============================================================================
SELECT COUNT(*) AS total_ids, COUNT(DISTINCT employee_id) AS unique_ids,
       MIN(length(employee_id)) AS min_len, MAX(length(employee_id)) AS max_len,
       COUNT(*) FILTER (WHERE employee_id !~ '^EMP[0-9]+$') AS non_conforming_ids
FROM employee_burnout_raw;
-- Expected: total_ids = unique_ids = 22,750, non_conforming_ids = 0


-- ============================================================
-- STEP 1.7 - VERIFY WFH SETUP VALUES
-- Purpose: confirm that wfh_setup_available is already stored
-- as standardized text labels ('No'/'Yes').
-- No conversion is required.
-- ============================================================

SELECT
    wfh_setup_available,
    COUNT(*) AS employee_count
FROM employee_burnout_raw
GROUP BY wfh_setup_available
ORDER BY wfh_setup_available;


-- ============================================================================
-- STEP 2 — CHECK DATA STRUCTURE AND DATA TYPES
-- Purpose: confirm each column's stored type before writing type-sensitive
-- logic (date parsing, numeric aggregation) later in the script.
-- ============================================================================
SELECT column_name, data_type, is_nullable
FROM information_schema.columns
WHERE table_name = 'employee_burnout_raw'
ORDER BY ordinal_position;
-- Key finding: date_of_joining is stored as PostgreSQL DATE.
-- No text parsing is required; Step 7 only formats it for the analytical output.


-- ============================================================================
-- STEP 3 — IDENTIFY MISSING VALUES
-- Purpose: conditional-aggregation pattern counts NULLs across every column
-- in a single pass, rather than one query per column.
-- ============================================================================
SELECT
  COUNT(*) FILTER (WHERE employee_id          IS NULL) AS missing_employee_id,
  COUNT(*) FILTER (WHERE date_of_joining       IS NULL) AS missing_date_of_joining,
  COUNT(*) FILTER (WHERE gender                IS NULL) AS missing_gender,
  COUNT(*) FILTER (WHERE company_type          IS NULL) AS missing_company_type,
  COUNT(*) FILTER (WHERE wfh_setup_available   IS NULL) AS missing_wfh,
  COUNT(*) FILTER (WHERE designation           IS NULL) AS missing_designation,
  COUNT(*) FILTER (WHERE resource_allocation   IS NULL) AS missing_resource_allocation,
  COUNT(*) FILTER (WHERE mental_fatigue_score  IS NULL) AS missing_mental_fatigue,
  COUNT(*) FILTER (WHERE burn_rate             IS NULL) AS missing_burn_rate
FROM employee_burnout_raw;
-- Expected: resource_allocation = 1,381 | mental_fatigue_score = 2,117 | burn_rate = 1,124
-- All other columns: 0 missing.


-- ============================================================================
-- STEP 4 — IDENTIFY DUPLICATE RECORDS
-- Purpose: check both the unique key (employee_id) and full-row duplication.
-- ============================================================================
-- Duplicate IDs
SELECT employee_id, COUNT(*) AS cnt
FROM employee_burnout_raw
GROUP BY employee_id
HAVING COUNT(*) > 1;
-- Expected: 0 rows returned (no duplicate IDs)

-- Full-row duplicates
SELECT employee_id, date_of_joining, gender, company_type, wfh_setup_available,
       designation, resource_allocation, mental_fatigue_score, burn_rate, COUNT(*) AS cnt
FROM employee_burnout_raw
GROUP BY employee_id, date_of_joining, gender, company_type, wfh_setup_available,
         designation, resource_allocation, mental_fatigue_score, burn_rate
HAVING COUNT(*) > 1;
-- Expected: 0 rows returned (no full duplicates)


-- ============================================================================
-- STEP 5 — DETECT INCONSISTENT VALUES
-- Purpose: DISTINCT on categoricals surfaces spelling variants/typos;
-- range checks confirm numeric fields respect their logical bounds.
-- ============================================================================
SELECT DISTINCT gender              FROM employee_burnout_raw;   -- {Female, Male}
SELECT DISTINCT company_type        FROM employee_burnout_raw;   -- {Service, Product}
SELECT DISTINCT wfh_setup_available FROM employee_burnout_raw;   -- {No, Yes}

SELECT designation, COUNT(*)
FROM employee_burnout_raw
GROUP BY designation
ORDER BY designation;                                            -- 6 clean levels: 0-5

SELECT COUNT(*) AS invalid_burn_rate
FROM employee_burnout_raw
WHERE burn_rate < 0 OR burn_rate > 1;                             -- Expected: 0

-- Conclusion: no inconsistent categorical entries or out-of-range values found.


-- ============================================================================
-- STEP 6 — DETECT POSSIBLE OUTLIERS (IQR METHOD)
-- Purpose: PostgreSQL's native percentile_cont() computes exact quartiles
-- directly — no manual ROW_NUMBER/window-function workaround needed. The
-- standard 1.5*IQR fence rule is then applied on top.
-- ============================================================================
WITH quartiles AS (
  SELECT
    percentile_cont(0.25) WITHIN GROUP (ORDER BY burn_rate) AS q1,
    percentile_cont(0.75) WITHIN GROUP (ORDER BY burn_rate) AS q3
  FROM employee_burnout_raw
  WHERE burn_rate IS NOT NULL
)
SELECT q1, q3, (q3 - q1) AS iqr,
       q1 - 1.5 * (q3 - q1) AS lower_bound,
       q3 + 1.5 * (q3 - q1) AS upper_bound
FROM quartiles;
-- burn_rate: Q1=0.31, Q3=0.59, bounds [-0.11, 1.01] -> 0 outliers (natural range covers it)

-- Repeat for mental_fatigue_score and resource_allocation:
WITH quartiles AS (
  SELECT
    percentile_cont(0.25) WITHIN GROUP (ORDER BY mental_fatigue_score) AS q1,
    percentile_cont(0.75) WITHIN GROUP (ORDER BY mental_fatigue_score) AS q3
  FROM employee_burnout_raw
  WHERE mental_fatigue_score IS NOT NULL
)
SELECT q1, q3, q1 - 1.5*(q3-q1) AS lower_bound, q3 + 1.5*(q3-q1) AS upper_bound,
       (SELECT COUNT(*) FROM employee_burnout_raw WHERE mental_fatigue_score < (q1 - 1.5*(q3-q1))) AS low_side_outliers
FROM quartiles;
-- mental_fatigue_score: bounds [0.85, 10.85] -> 347 low-side points flagged
-- (valid low-fatigue employees, not data errors -- retained, not dropped)

WITH quartiles AS (
  SELECT
    percentile_cont(0.25) WITHIN GROUP (ORDER BY resource_allocation) AS q1,
    percentile_cont(0.75) WITHIN GROUP (ORDER BY resource_allocation) AS q3
  FROM employee_burnout_raw
  WHERE resource_allocation IS NOT NULL
)
SELECT q1, q3, q1 - 1.5*(q3-q1) AS lower_bound, q3 + 1.5*(q3-q1) AS upper_bound
FROM quartiles;
-- resource_allocation: bounds [-1.5, 10.5] -> 0 outliers


-- ============================================================================
-- STEP 7, 8, 9, 11 — BUILD THE CLEANED ANALYTICAL TABLE
-- Combines in one pass:
--   Step 7  - retain date_of_joining as a native PostgreSQL DATE
--   Step 8  - derive a joining_month label
--   Step 9  - remove rows with a missing burn_rate, resource_allocation, OR
--             mental_fatigue_score (complete-case / listwise deletion)
--   Step 11 - create the Risk column from the business rule
--
-- Note on Step 9: earlier versions of this pipeline only dropped rows
-- missing burn_rate (the target variable) and left resource_allocation /
-- mental_fatigue_score gaps for Phase 3 to median-impute at modeling time.
-- This version instead removes all three categories of missing data here,
-- at the source, so employee_burnout_clean is a complete-case table with
-- zero NULLs in any numeric field. This is a deliberate trade-off: it
-- costs substantially more rows (3,036 additional vs. the burn_rate-only
-- filter) in exchange for a cleaned table that never requires downstream
-- imputation — every phase after this one reads guaranteed-complete data.
--
-- Note: an ISO-format date_of_joining_iso column was evaluated during
-- development as a "sortable datetime" convenience column, but it was never
-- consumed by any downstream Python, modeling, or dashboard step — only
-- joining_month was — so it is intentionally excluded here to keep the
-- schema limited to columns that are actually used in later phases.
-- ============================================================================
-- Note: PostgreSQL gives no row-order guarantee without an explicit ORDER BY.
-- The analytical export is therefore ordered by employee_id so repeated exports
-- have deterministic row order before Python applies train_test_split with
-- random_state=42. This strengthens reproducibility across reruns.
DROP TABLE IF EXISTS employee_burnout_clean;

CREATE TABLE employee_burnout_clean AS
SELECT
  employee_id,
  date_of_joining,

   -- STEP 8: derived label
  trim(to_char(date_of_joining, 'Month')) AS joining_month,

  gender,
  company_type,
  wfh_setup_available,
  designation,
  resource_allocation,
  mental_fatigue_score,
  burn_rate,
  CASE WHEN burn_rate >= 0.70 THEN 'High Risk' ELSE 'Low Risk' END AS risk                 -- STEP 11: business rule
FROM employee_burnout_raw
WHERE burn_rate IS NOT NULL                                                               -- STEP 9: drop rows missing the target
  AND resource_allocation IS NOT NULL                                                     -- STEP 9: drop rows missing this feature
  AND mental_fatigue_score IS NOT NULL                                                    -- STEP 9: drop rows missing this feature
ORDER BY employee_id;

SELECT COUNT(*) AS clean_row_count FROM employee_burnout_clean;
-- Expected: 18,590 rows retained (4,160 removed across all three missing-value
-- categories combined, ~18.3% of source — up from 4.9% when only burn_rate
-- was filtered)


-- ============================================================================
-- STEP 10 — DESCRIPTIVE STATISTICS
-- ============================================================================
SELECT 'burn_rate' AS metric,
       ROUND(MIN(burn_rate),4)  AS min_val, ROUND(AVG(burn_rate),4)  AS avg_val,
       ROUND(MAX(burn_rate),4)  AS max_val, COUNT(burn_rate)         AS n
FROM employee_burnout_clean
UNION ALL
SELECT 'mental_fatigue_score',
       ROUND(MIN(mental_fatigue_score),4), ROUND(AVG(mental_fatigue_score),4),
       ROUND(MAX(mental_fatigue_score),4), COUNT(mental_fatigue_score)
FROM employee_burnout_clean
UNION ALL
SELECT 'resource_allocation',
       ROUND(MIN(resource_allocation),4), ROUND(AVG(resource_allocation),4),
       ROUND(MAX(resource_allocation),4), COUNT(resource_allocation)
FROM employee_burnout_clean;
/*
 burn_rate             : min 0.00 | avg 0.4524 | max 1.00 | n 18,590
 mental_fatigue_score   : min 0.00 | avg 5.7322 | max 10.00| n 18,590
 resource_allocation    : min 1.00 | avg 4.4866 | max 10.00| n 18,590
*/


-- ============================================================================
-- BUSINESS QUESTIONS
-- ============================================================================

-- How many employees are High Risk vs Low Risk? What % is High Risk?
SELECT risk, COUNT(*) AS employee_count,
       ROUND(100.0 * COUNT(*) / (SELECT COUNT(*) FROM employee_burnout_clean), 2) AS pct
FROM employee_burnout_clean
GROUP BY risk;
-- High Risk: 2,071 (11.14%) | Low Risk: 16,519 (88.86%)

-- Which company type has the highest burnout?
SELECT company_type, ROUND(AVG(burn_rate),4) AS avg_burn_rate
FROM employee_burnout_clean GROUP BY company_type ORDER BY avg_burn_rate DESC;
-- Service: 0.4533 | Product: 0.4508 (effectively tied)

-- Does Work From Home affect burnout?
SELECT wfh_setup_available, ROUND(AVG(burn_rate),4) AS avg_burn_rate
FROM employee_burnout_clean GROUP BY wfh_setup_available ORDER BY avg_burn_rate DESC;
-- No: 0.5182 | Yes: 0.3963 (large, actionable gap)

-- Which designation experiences the highest burnout?
SELECT designation, ROUND(AVG(burn_rate),4) AS avg_burn_rate
FROM employee_burnout_clean GROUP BY designation ORDER BY avg_burn_rate DESC;
-- Rises almost linearly: 0 -> 0.1513 ... 5 -> 0.8565

-- Which gender has the highest average burnout?
SELECT gender, ROUND(AVG(burn_rate),4) AS avg_burn_rate
FROM employee_burnout_clean GROUP BY gender ORDER BY avg_burn_rate DESC;
-- Male: 0.4850 | Female: 0.4229

-- Which employees have the highest mental fatigue?
-- NOTE: PostgreSQL defaults to NULLS FIRST on DESC sorts (the opposite of
-- SQLite), so NULLS LAST must be stated explicitly or NULL rows wrongly
-- appear "on top" as if they were the highest values.
SELECT employee_id, mental_fatigue_score, burn_rate, designation
FROM employee_burnout_clean
WHERE mental_fatigue_score IS NOT NULL
ORDER BY mental_fatigue_score DESC
LIMIT 5;
-- All top-5 are Designation 4-5, burn_rate 0.88-0.99

-- Does resource allocation influence burnout?
SELECT resource_allocation, ROUND(AVG(burn_rate),4) AS avg_burn_rate
FROM employee_burnout_clean
WHERE resource_allocation IS NOT NULL
GROUP BY resource_allocation
ORDER BY resource_allocation;
-- Near-perfectly monotonic: 1 -> 0.139 ... 10 -> 0.900


/* ============================================================================
   END OF PHASE 1
   Output table `employee_burnout_clean` (18,590 rows, 11 columns) feeds
   directly into Phase 2 (Python EDA) and Phase 3 (predictive modeling).
   To export it for those phases:  \copy employee_burnout_clean TO
   'employee_burnout_clean.csv' WITH (FORMAT csv, HEADER, DELIMITER ',');
   ============================================================================ */


SELECT
    column_name,
    data_type
FROM information_schema.columns
WHERE table_name = 'employee_burnout_clean'
ORDER BY ordinal_position;