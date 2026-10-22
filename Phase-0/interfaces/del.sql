SELECT 
    agent_id, 
    agent_name, 
    total_points, 
    CAST(ranking AS UNSIGNED) AS ranking
FROM (
    SELECT 
        t.agent_id,
        a.agent_name,
        t.total_points,
        @row := @row + 1,
        @rank := IF(@prev = t.total_points, @rank, @row) AS ranking,
        @prev := t.total_points
    FROM (
        SELECT 
            agent_id,
            SUM(
                ABS((LENGTH(d_name) - LENGTH(d_cons)) - (LENGTH(m_name) - LENGTH(m_cons))) +
                ABS(LENGTH(d_cons) - LENGTH(m_cons))
            ) AS total_points
        FROM (
            SELECT 
                agent_id,
                LOWER(DAYNAME(delivery_date)) AS d_name,
                LOWER(MONTHNAME(delivery_date)) AS m_name,
                REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(LOWER(DAYNAME(delivery_date)), 'a', ''), 'e', ''), 'i', ''), 'o', ''), 'u', '') AS d_cons,
                REPLACE(REPLACE(REPLACE(REPLACE(REPLACE(LOWER(MONTHNAME(delivery_date)), 'a', ''), 'e', ''), 'i', ''), 'o', ''), 'u', '') AS m_cons
            FROM deliveries
            WHERE successful = 1
        ) raw_stats
        GROUP BY agent_id
        HAVING total_points > 0
    ) t
    JOIN agents a ON t.agent_id = a.agent_id
    CROSS JOIN (SELECT @rank := 1, @row := 0, @prev := NULL) vars
    ORDER BY t.total_points DESC, t.agent_id ASC
) final_result
ORDER BY ranking ASC, agent_id ASC;
