import pytest

from codeconductor.utils.extract import extract_code


def test_sql_extraction_without_fences_basic():
    text = """
We can solve this with a single query. Consider the following tables...

SELECT customer_id, SUM(amount) AS total_amount
FROM orders
GROUP BY customer_id
ORDER BY total_amount DESC
LIMIT 5;

Some explanation follows here.
"""

    code = extract_code(text, lang_hint="sql")
    assert code.lower().startswith("select ")
    assert code.strip().endswith(";")
    assert "group by" in code.lower()
