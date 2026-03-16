import os
import json
import requests
import sqlite3

# OpenAI API key setup
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise ValueError("Please set the OPENAI_API_KEY environment variable.")

# Optional helper copied from notebook
# (In a plain script we just print, no IPython display required.)

def display_markdown(text):
    print(text)

# Database setup
db_file_name = os.path.abspath('local_data/test_db.db')
os.makedirs(os.path.dirname(db_file_name), exist_ok=True)
if os.path.exists(db_file_name):
    os.remove(db_file_name)


def _connect_readonly():
    conn = sqlite3.connect(db_file_name)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA query_only = ON;")
    return conn


def _rows_to_dicts(cur):
    cols = [c[0] for c in cur.description] if cur.description else []
    return [dict(zip(cols, row)) for row in cur.fetchall()]


def list_tables():
    conn = _connect_readonly()
    try:
        cur = conn.execute("""
            SELECT name FROM sqlite_master
            WHERE type='table' AND name NOT LIKE 'sqlite_%'
            ORDER BY name
        """)
        tables = [r[0] for r in cur.fetchall()]
        return {"tables": tables}
    finally:
        conn.close()


def preview_table(table: str, limit: int = 5, offset: int = 0):
    conn = _connect_readonly()
    try:
        q = f"SELECT * FROM {table} LIMIT ? OFFSET ?"
        cur = conn.execute(q, (int(limit), int(offset)))
        return {"table": table, "rows": _rows_to_dicts(cur)}
    finally:
        conn.close()


def run_sql(sql: str, params: dict | None = None, max_rows: int = 1000):
    head = (sql or "").lstrip().split(" ", 1)[0].lower()
    if head not in {"select", "pragma", "with", "explain"}:
        raise ValueError("Only read-only queries are allowed in this demo")
    conn = _connect_readonly()
    try:
        cur = conn.execute(sql, params or {})
        rows = _rows_to_dicts(cur)[: int(max_rows)]
        return {"rowcount": len(rows), "rows": rows}
    finally:
        conn.close()


def call_openai(openai_api_key, model_name, messages, tools):
    url = "https://api.openai.com/v1/chat/completions"
    headers = {"Authorization": f"Bearer {openai_api_key}", "Content-Type": "application/json"}
    payload = {
        "model": model_name,
        "messages": messages,
        "tools": tools,
        "tool_choice": "auto",
        "parallel_tool_calls": True,
    }
    resp = requests.post(url, headers=headers, data=json.dumps(payload), timeout=60)
    resp.raise_for_status()
    return resp.json()


tools = [
    {
        "type": "function",
        "function": {
            "name": "list_tables",
            "description": "List all non-internal SQLite tables available in the database.",
            "parameters": {"type": "object", "properties": {}},
        },
    },
    {
        "type": "function",
        "function": {
            "name": "preview_table",
            "description": "Return a small sample of rows from a given table to understand its contents.",
            "parameters": {
                "type": "object",
                "properties": {
                    "table": {"type": "string", "description": "Target table name."},
                    "limit": {"type": "integer", "minimum": 1, "maximum": 1000},
                    "offset": {"type": "integer", "minimum": 0},
                },
                "required": ["table"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "run_sql",
            "description": "Run a read-only SQL query (SELECT/PRAGMA/WITH/EXPLAIN) and return rows as JSON.",
            "parameters": {
                "type": "object",
                "properties": {
                    "sql": {"type": "string"},
                    "params": {"type": "object", "additionalProperties": True},
                    "max_rows": {"type": "integer", "minimum": 1, "maximum": 50000},
                },
                "required": ["sql"],
            },
        },
    },
]


LOCAL_FUNCS = {
    "list_tables": lambda **kw: list_tables(),
    "preview_table": preview_table,
    "run_sql": run_sql,
}


def main():
    print("=== Get List of models ===")
    url = "https://api.openai.com/v1/models"
    headers = {
        "Authorization": f"Bearer {openai_api_key}",
        "Content-Type": "application/json"
    }

    response = requests.get(url, headers=headers)
    print(response.text)

    try:
        print(pd.DataFrame(response.json().get('data', [])))
    except Exception as e:
        print("Could not parse model list into DataFrame:", e)

    # Get DB schema via image analysis
    print("=== Request LLM to analyze image ===")
    url = "https://api.openai.com/v1/responses"
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {openai_api_key}"
    }
    prompt = """
Inspect the DB schema in the image.
Generate instructions to reproduce these tables in sqlite3 dialect.
Double-check sqlite3 syntax.
DO not include any text, return SQL code only.
"""

    data = {
        "model": "gpt-5-mini",
        "max_output_tokens": 15000,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": prompt},
                    {
                        "type": "input_image",
                        "image_url": "https://www.astera.com/wp-content/uploads/2024/05/Database-schema.png"
                    }
                ]
            }
        ]
    }

    response = requests.post(url, headers=headers, data=json.dumps(data))
    response.raise_for_status()
    print(response.text)

    db_schema = response.json()['output'][-1]['content'][0]['text']
    print("db_schema:\n", db_schema)

    print("=== Generate DB ===")
    with sqlite3.connect(db_file_name) as conn:
        try:
            c = conn.cursor()
            c.executescript(db_schema)
            conn.commit()
            print("Database and tables created successfully.")
        except sqlite3.Error as e:
            conn.rollback()
            raise RuntimeError(f"Database initialization failed: {e}") from e

    # Fill DB with synthetic data
    print("=== Fill DB with synthetic data ===")
    sample_prompt = f"""
## Instructions
Consider the database schema below.
Your task is to generate sqlite script that would fill that DB with sample data.
Double-check sqlite3 syntax.
DO not include any text, return SQL code only.

## Database Schema:

{db_schema}
"""
    data = {
        "model": "gpt-5-mini",
        "max_output_tokens": 25000,
        "input": [
            {
                "role": "user",
                "content": [
                    {"type": "input_text", "text": sample_prompt},
                    {
                        "type": "input_image",
                        "image_url": "https://www.astera.com/wp-content/uploads/2024/05/Database-schema.png"
                    }
                ]
            }
        ]
    }
    response = requests.post(url, headers=headers, data=json.dumps(data))
    response.raise_for_status()

    sample_data_sql = response.json()['output'][-1]['content'][0]['text']
    print("sample_data_sql:\n", sample_data_sql)

    with sqlite3.connect(db_file_name) as conn:
        try:
            c = conn.cursor()
            c.executescript(sample_data_sql)
            conn.commit()
            print("Sample data inserted successfully.")
        except sqlite3.Error as e:
            conn.rollback()
            raise RuntimeError(f"Filling DB failed: {e}") from e

    # LLM Function calling demo
    system_prompt = """
You are a careful data analyst. Use tools to explore, then return a clear Markdown report with sections:
- Overview
- Key Metrics
- Notable Insights
- Samples (optional)
Keep it concise and actionable. Exclude raw tool outputs and intermediate reasoning from the report.
"""
    user_prompt = f"""
Database schema:
{db_schema}
Create the report.
"""

    print(user_prompt)
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]

    print("=== Conversation Start ===")
    iteration = 0
    while True:
        result = call_openai(openai_api_key, model_name="gpt-5-mini", messages=messages, tools=tools)
        msg = result["choices"][0]["message"]
        iteration += 1

        tool_calls = msg.get("tool_calls", [])
        if tool_calls:
            messages.append(msg)
            for call in tool_calls:
                name = call["function"]["name"]
                args = json.loads(call["function"]["arguments"] or "{}")
                print(f"=== Tool call {iteration}: {name}({args}) ===")
                fn = LOCAL_FUNCS[name]
                try:
                    output = fn(**args)
                except Exception as e:
                    output = {"error": str(e)}
                messages.append({
                    "role": "tool",
                    "tool_call_id": call["id"],
                    "content": json.dumps(output),
                })
            continue

        print("=== Final Answer ===")
        display_markdown(msg.get("content", ""))
        break

    print("=== Conversation End ===")


if __name__ == "__main__":
    main()
