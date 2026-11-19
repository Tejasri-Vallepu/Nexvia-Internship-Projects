# Data and API documentation

For every source, document:
- Source name and official documentation URL
- Collection date/time and query scope
- Fields collected and their meaning
- API quota/rate limits and pagination behavior
- Terms, license, retention, and redistribution restrictions
- Missing values and known sampling limitations

## Suggested normalized fields
`source`, `post_id`, `created_at`, `text`, `language`, `author_id_hash`, `community`,
`reply_to_id`, `like_count`, `comment_count`, `url`

Avoid storing direct identifiers unless necessary and permitted. Hashing does not automatically make data anonymous.
