import sqlite3

# Connect to SQLite database (creates if not exists)
conn = sqlite3.connect("database/attendance.db")

cursor = conn.cursor()

# Create the attendance table
cursor.execute("""
CREATE TABLE IF NOT EXISTS attendance (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    timestamp TEXT
)
""")

conn.commit()
conn.close()

print("Database and table created successfully!")
