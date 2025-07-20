SET @FA_BADGE := 'wikibase-badge-Q17437796';
SET @GA_BADGE := 'wikibase-badge-Q17437798';

SELECT
      p.page_id
    , p.page_title                                  AS url_title
    , sd.pp_value                                   AS short_description
    , IF(ppb.pp_propname = @FA_BADGE, 'FA', 'GA')   AS quality
FROM
         page               AS p
    JOIN page_props         AS ppb      ON p.page_id = ppb.pp_page AND p.page_namespace = 0 AND p.page_is_redirect = 0
    LEFT JOIN page_props    AS sd       ON p.page_id = sd.pp_page  AND sd.pp_propname = 'wikibase-shortdesc'
WHERE
        ppb.pp_propname IN (@FA_BADGE, @GA_BADGE)
ORDER BY
      quality DESC
    , url_title
;
