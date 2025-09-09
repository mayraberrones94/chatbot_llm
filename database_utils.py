import sqlite3
from flask import g

DATABASE = "dialogues.db"

def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DATABASE)
        g.db.row_factory = sqlite3.Row
    return g.db


def close_db(exception=None):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db(app):
    """Initialize database with schema.sql"""
    with app.app_context():
        db = get_db()
        with app.open_resource("schema.sql") as f:
            print('initialising database')
            db.executescript(f.read().decode("utf8"))
        db.commit()


def insert_dialogue(prompt, initial_response, blocks):
    """
    Insert a dialogue with its blocks.

    prompt: str
    initial_response: str
    blocks: list of dicts [{ "speaker": "A", "text": "..."}, ...]
    """
    db = get_db()
    cur = db.cursor()

    # Insert into dialogue table
    cur.execute(
        "INSERT INTO dialogue (prompt, initial_response) VALUES (?, ?)",
        (prompt, initial_response),
    )
    dialogue_id = cur.lastrowid

    # Insert blocks
    for i, block in enumerate(blocks, start=1):
        cur.execute(
            """
            INSERT INTO dialogue_block (dialogue_id, speaker, block_order, text)
            VALUES (?, ?, ?, ?)
            """,
            (dialogue_id, block["speaker"], i, block["text"]),
        )

    db.commit()
    return dialogue_id
