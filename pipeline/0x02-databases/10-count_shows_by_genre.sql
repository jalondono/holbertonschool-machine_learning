-- lists all shows contained in hbtn_0d_tvshows without a genre linked.
SELECT tv_genres.name as genre,
COUNT(tv_show_genres.show_id) as number_of_shows
FROM tv_genres
LEFT JOIN tv_show_genres
ON tv_genres.id = tv_show_genres.genre_id
GROUP BY tv_genres.name
HAVING number_of_shows > 0
ORDER BY number_of_shows DESC;