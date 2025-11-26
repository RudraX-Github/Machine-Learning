USE weather_history;

-- Step 1: Combine all tables into a single master table
CREATE TABLE combined_weather_data AS (
    SELECT * FROM Bardoli
    UNION ALL
    SELECT * FROM Bardoli_Kar
    UNION ALL
    SELECT * FROM Bardoli_Ma
    UNION ALL
    SELECT * FROM Kosamba
    UNION ALL
    SELECT * FROM Kosamba_D
    UNION ALL
    SELECT * FROM Mahuva
    UNION ALL
    SELECT * FROM Mahuva_Am
    UNION ALL
    SELECT * FROM Mandvi
    UNION ALL
    SELECT * FROM Nizar
    UNION ALL
    SELECT * FROM Nizar_Kuka
    UNION ALL
    SELECT * FROM Nizar_Pumb
    UNION ALL
    SELECT * FROM S_Mandvi
    UNION ALL
    SELECT * FROM Songadh
    UNION ALL
    SELECT * FROM Songadh_B
    UNION ALL
    SELECT * FROM Songadh_U
    UNION ALL
    SELECT * FROM Surat
    UNION ALL
    SELECT * FROM Uchhal
    UNION ALL
    SELECT * FROM Valod
    UNION ALL
    SELECT * FROM Valod_Buh
    UNION ALL
    SELECT * FROM Vyara_Pan
    UNION ALL
    SELECT * FROM Vyra
);

-- **********************************************************************************************************************
-- ----------------------------------------------------------------------------------------------------------------------
-- **********************************************************************************************************************

-- Step 2: TEST THIS QUERY FIRST!
SELECT * 
FROM ( SELECT *, ROW_NUMBER() OVER 
(PARTITION BY `market_name`, `time` ORDER BY `time`) as row_num 
FROM combined_weather_data ) AS temp WHERE row_num = 1 LIMIT 10;

-- **********************************************************************************************************************
-- ----------------------------------------------------------------------------------------------------------------------
-- **********************************************************************************************************************

-- Step 3: Run this ONLY after the test query works
DROP TABLE IF EXISTS cleaned_weather_data;

CREATE TABLE cleaned_weather_data AS
SELECT
    market_name, latitude, longitude, time,
    temperature_max, temperature_min, temperature_mean,
    precipitation_sum, rain_sum,
    wind_speed_max, wind_gusts_max,
    weather_code, cloud_cover
FROM (
    -- finds all unique rows
    SELECT
        *,
        ROW_NUMBER() OVER(
            PARTITION BY market_name, time
            ORDER BY time
        ) as row_num
    FROM
        combined_weather_data
) AS temp
WHERE
    row_num = 1;
    
    Drop table combined_weather_data;
    
-- **********************************************************************************************************************
-- ----------------------------------------------------------------------------------------------------------------------
-- **********************************************************************************************************************

-- Disable safe update mode for this session
SET SQL_SAFE_UPDATES = 0;
    
 -- **********************************************************************************************************************
-- ----------------------------------------------------------------------------------------------------------------------
-- **********************************************************************************************************************

-- Step 4 Impute (fill in) NULLs
-- Assumes that NULL for rain/precipitation means 0
UPDATE cleaned_weather_data
SET
    precipitation_sum = COALESCE(precipitation_sum, 0),
    rain_sum = COALESCE(rain_sum, 0);

-- Step 4a: Delete rows with critical missing data
-- If we don't have a time or a mean temperature, the row is not useful
DELETE FROM cleaned_weather_data
WHERE
    time IS NULL
    OR temperature_mean IS NULL;

-- Step 2c: Remove illogical data
-- Deletes any rows where min temperature is greater than max
DELETE FROM cleaned_weather_data
WHERE
    temperature_min > temperature_max;
    UPDATE cleaned_weather_data 
    SET precipitation_sum = COALESCE(precipitation_sum, 0),
				rain_sum = COALESCE(rain_sum, 0);

-- **********************************************************************************************************************
-- ----------------------------------------------------------------------------------------------------------------------
-- **********************************************************************************************************************
-- enable safe update mode for this session
UPDATE cleaned_weather_data
SET
    precipitation_sum = COALESCE(precipitation_sum, 0),
    rain_sum = COALESCE(rain_sum, 0);

-- Turn safe update mode back on
SET SQL_SAFE_UPDATES = 1;

-- **********************************************************************************************************************
-- ----------------------------------------------------------------------------------------------------------------------
-- **********************************************************************************************************************

-- Step 5: Add a column for daily temperature range
ALTER TABLE cleaned_weather_data
ADD COLUMN temp_range DECIMAL(5, 2);

-- Populate the new column
UPDATE cleaned_weather_data
SET temp_range = temperature_max - temperature_min;

-- Step 5a: Add columns for year, month, and day for easy grouping
ALTER TABLE cleaned_weather_data
ADD COLUMN data_year INT,
ADD COLUMN data_month INT,
ADD COLUMN data_day INT;

-- Populate the new date columns
UPDATE cleaned_weather_data
SET
    data_year = YEAR(time),
    data_month = MONTH(time),
    data_day = DAY(time);
    
    