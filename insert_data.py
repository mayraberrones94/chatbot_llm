import sqlite3
import pandas as pd
import re

DB_PATH = "dialogues.db"
INPUT_FILE = "dialogues.csv"


def connect_db():
    conn = sqlite3.connect(DB_PATH)
    return conn


def parse_exchange_number(exchange):
    """Extract the numeric part (e.g. '1A' -> 1)."""
    match = re.match(r"(\d+)", str(exchange))
    return int(match.group(1)) if match else None


def import_data():
    # Read spreadsheet (detect Excel vs CSV automatically)
    if INPUT_FILE.endswith(".csv"):
        df = pd.read_csv(INPUT_FILE)
    else:
        df = pd.read_excel(INPUT_FILE)

    # Normalize column names
    df = df.rename(columns=lambda x: x.strip().upper())
    required_cols = ["EXCHANGE #", "SPEAKER 1", "SPEAKER 2"]
    for col in required_cols:
        if col not in df.columns:
            raise ValueError(f"Missing required column: {col}")

    conn = connect_db()
    cur = conn.cursor()

    # Group rows by dialogue number
    grouped = df.groupby(df["EXCHANGE #"].apply(parse_exchange_number))

    for dialogue_num, rows in grouped:
        if pd.isna(dialogue_num):
            continue

        # Insert dialogue (blank prompt + initial response)
        cur.execute(
            "INSERT INTO dialogue (prompt, initial_response) VALUES (?, ?)",
            ("", ""),
        )
        dialogue_id = cur.lastrowid

        block_order = 1
        for _, row in rows.iterrows():
            speaker1_text = str(row["SPEAKER 1"]).strip()
            speaker2_text = str(row["SPEAKER 2"]).strip()

            if speaker1_text and speaker1_text.lower() != "nan":
                cur.execute(
                    """
                    INSERT INTO dialogue_block (dialogue_id, speaker, block_order, text)
                    VALUES (?, ?, ?, ?)
                    """,
                    (dialogue_id, "1", block_order, speaker1_text),
                )
                block_order += 1

            if speaker2_text and speaker2_text.lower() != "nan":
                cur.execute(
                    """
                    INSERT INTO dialogue_block (dialogue_id, speaker, block_order, text)
                    VALUES (?, ?, ?, ?)
                    """,
                    (dialogue_id, "2", block_order, speaker2_text),
                )
                block_order += 1

        print(f"Imported Dialogue {dialogue_num} with {block_order-1} blocks")

    conn.commit()
    conn.close()


if __name__ == "__main__":
    import_data()
