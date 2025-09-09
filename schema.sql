CREATE TABLE dialogue (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    prompt TEXT NOT NULL,
    initial_response TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE dialogue_block (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dialogue_id INTEGER NOT NULL,
    speaker TEXT NOT NULL,
    block_order INTEGER NOT NULL,
    text TEXT NOT NULL,
    FOREIGN KEY (dialogue_id) REFERENCES dialogue(id) ON DELETE CASCADE
);
