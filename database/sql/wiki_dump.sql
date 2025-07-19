SET @FA_BADGE := 'wikibase-badge-Q17437796';
SET @GA_BADGE := 'wikibase-badge-Q17437798';
SET @SD_PROP := 'wikibase-shortdesc';  -- short-description key

SELECT
        p.page_title                       AS url_title,
        p.page_id,
        CASE pp_badge.pp_propname
             WHEN @FA_BADGE THEN 'FA'
             WHEN @GA_BADGE THEN 'GA'
        END                                AS quality,
        sd.pp_value                        AS short_description
FROM        page_props pp_badge  -- badge rows
INNER JOIN  page       p  ON p.page_id = pp_badge.pp_page
LEFT  JOIN  page_props sd        -- short-desc rows
           ON sd.pp_page     = p.page_id
          AND sd.pp_propname = @SD_PROP
WHERE       pp_badge.pp_propname IN (@FA_BADGE, @GA_BADGE)
  AND       p.page_namespace   = 0        -- main space only
  AND       p.page_is_redirect = 0        -- skip redirects
ORDER BY    quality DESC,                 -- FA first
            url_title
;
