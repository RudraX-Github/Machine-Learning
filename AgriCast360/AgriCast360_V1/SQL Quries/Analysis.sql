-- 1. Check Data Distribution

SELECT market_name, COUNT(*) as data_points, MIN(time) as earliest_date, MAX(time) as latest_date
FROM cleaned_weather_data
GROUP BY market_name
ORDER BY data_points DESC;
-- ********************************************************************************************************************************************************************
-- 2. Find Overall Extreme Weather
-- Find the hottest day

SELECT market_name, time, temperature_max
FROM cleaned_weather_data
ORDER BY temperature_max DESC
LIMIT 1;

-- Find the coldest day
SELECT market_name, time, temperature_min
FROM cleaned_weather_data
ORDER BY temperature_min ASC
LIMIT 1;

-- *********************************************************************************************************************************************************************
-- 3. Calculate Average Conditions per Market

SELECT market_name, AVG(temperature_mean) as avg_temp, AVG(temp_range) as avg_temp_range, SUM(precipitation_sum) as total_precipitation, AVG(wind_speed_max) as avg_wind_speed
FROM cleaned_weather_data
GROUP BY market_name
ORDER BY avg_temp DESC;
    
-- *******************************************************************************************************************************************************
-- 4. Find the Top 10 Rainiest Days
SELECT market_name, time, rain_sum

FROM cleaned_weather_data
ORDER BY rain_sum DESC
LIMIT 10;

-- *******************************************************************************************************************************************************
-- 5. Analyze Monthly Averages (for Surat)
SELECT data_year, data_month, AVG(temperature_mean) as avg_monthly_temp, SUM(rain_sum) as total_monthly_rain
FROM cleaned_weather_data
WHERE market_name = 'Surat' 
GROUP BY data_year, data_month
ORDER BY data_year, data_month;

-- *******************************************************************************************************************************************************
-- 6. Find the Most Common Weather Events
SELECT weather_code, COUNT(*) AS event_count
FROM cleaned_weather_data
GROUP BY weather_code
ORDER BY event_count DESC;

-- *******************************************************************************************************************************************************
-- 6. Find All Days With Rain

SELECT market_name, time, temperature_mean, weather_code
FROM cleaned_weather_data
WHERE weather_code IN (61, 63, 65);

-- *******************************************************************************************************************************************************
-- 7. Abnormally Hot/Cold Days

-- Step 1: Create a CTE to calculate the average for each month
WITH monthly_avg AS (
    SELECT market_name, data_year, data_month, AVG(temperature_mean) AS avg_month_temp
    FROM cleaned_weather_data
    GROUP BY market_name, data_year, data_month)

-- Step 2: Join the daily data with the monthly averages
SELECT c.time, c.market_name, c.temperature_mean AS daily_temp, m.avg_month_temp, (c.temperature_mean - m.avg_month_temp) AS "temp_difference"
FROM cleaned_weather_data AS c
JOIN monthly_avg AS m ON c.market_name = m.market_name AND c.data_year = m.data_year AND c.data_month = m.data_month
WHERE c.market_name = 'surat' 
ORDER BY temp_difference ASC -- (DESC abnormally HOT days first, ASC abnormally COLD days first)
limit 10;

-- *******************************************************************************************************************************************************
-- 8. Temp Drop After Rain
WITH NextDayTemp AS (
    SELECT time, market_name, temperature_mean, rain_sum,
        -- Get the mean temp from the *next* day
        LEAD(temperature_mean, 1) OVER(
            PARTITION BY market_name
            ORDER BY time) AS next_day_temp
    FROM
        cleaned_weather_data
)
-- Now, select only the heavy rain days and show the temperature change
SELECT time, market_name, temperature_mean AS rainy_day_temp, next_day_temp, (next_day_temp - temperature_mean) AS temp_change, rain_sum
FROM NextDayTemp
WHERE rain_sum > 20 -- Define "heavy rain"
ORDER BY temp_change ASC; -- Show the biggest temperature drops first


-- *******************************************************************************************************************************************************
-- 9. Top 3 Hottest Days
WITH RankedTemps AS (
    SELECT market_name, time, temperature_max,
        
        -- Assign a rank to each day *within* its market, from hottest to coldest
        ROW_NUMBER() OVER(
            PARTITION BY market_name
            ORDER BY temperature_max DESC ) AS temp_rank
    FROM cleaned_weather_data )
    
-- Select the top 3 from each market
SELECT market_name, time, temperature_max
FROM RankedTemps
WHERE temp_rank <= 3
ORDER BY market_name, temp_rank;

-- *******************************************************************************************************************************************************
-- 10. Longest Dry Spell

WITH SpellIdentifier AS (
    -- Step 1: Mark "break" days (rainy OR cool) vs. "hot-dry" days
    SELECT time, market_name, (CASE  WHEN rain_sum > 0 OR temperature_max <= 38 THEN 1  ELSE 0  END) AS is_break_day
    FROM
        cleaned_weather_data
),
SpellGroups AS (
    -- Step 2: Create 'islands' by making a group ID for each spell
    SELECT time, market_name, is_break_day, SUM(is_break_day) OVER (PARTITION BY market_name ORDER BY time ) AS spell_group_id
    FROM SpellIdentifier
)
-- Step 3: Count the days in each 'island' of hot-dry days
SELECT market_name, MIN(time) AS spell_start, MAX(time) AS spell_end, COUNT(*) AS consecutive_hot_dry_days
FROM SpellGroups
WHERE is_break_day = 0 
GROUP BY market_name, spell_group_id
ORDER BY consecutive_hot_dry_days DESC
LIMIT 10;