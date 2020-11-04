-- ranks country origins of bands, ordered by the number of (non-unique) fans

SELECT band_name,
IF(split IS NULL, (2020-formed), (split - formed)) as lifespan
FROM metal_bands
WHERE style LIKE '%Glam Rock%'
ORDER BY lifespan DESC;