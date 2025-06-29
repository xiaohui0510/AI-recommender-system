import sqlite3
import pandas as pd
from sqlalchemy import create_engine


# Connect to SQLite database (it will create the file if it doesn't exist)
conn = sqlite3.connect('shift_reports.db')
cursor = conn.cursor()

# Create table
cursor.execute('''
    CREATE TABLE IF NOT EXISTS shift_reports (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        date TEXT,
        machine_id TEXT,
        issues TEXT,
        resolution TEXT
    )
''')
conn.commit()
conn.close()



# Read your Excel file
df = pd.read_excel('Cement Mill 3 Summary Shift Report 2023.xlsx', sheet_name='C.Mill 3 (visualize)')
df = df.dropna(how='all', axis=1)  # Remove columns with all NaN
df_cleaned = df.dropna(how='any', axis=0)  # Remove rows with any NaN

# Rename columns if needed so that they match your database schema
df_cleaned.rename(columns={
    'Date': 'date',
    'Equipment #': 'machine_id',
    'Issues': 'issues',
    'Resolution': 'resolution'
}, inplace=True)

# Create SQLAlchemy engine for SQLite (for MySQL change the connection string accordingly)
engine = create_engine('sqlite:///shift_reports.db', echo=True)

# Write to SQL table; if the table already exists, you can set if_exists='append'
df_cleaned.to_sql('shift_reports', con=engine, if_exists='replace', index=False)
print("Data migrated to SQL database successfully!")
