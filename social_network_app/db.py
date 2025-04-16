import psycopg2
import os
from contextlib import contextmanager
from dotenv import load_dotenv
import logging

load_dotenv() # Load .env file for local development

# --- Get the Render PostgreSQL Connection URL ---
# Use 'DATABASE_URL' as the standard variable name Render provides/expects
DATABASE_URL = os.getenv('DATABASE_URL')

if not DATABASE_URL:
    logging.error("FATAL: DATABASE_URL environment variable not set.")
    # Handle appropriately - raise error or exit if critical for app start
    # For now, it will fail later during connection attempt

@contextmanager
def get_connection():
    """Provides a transactional database connection context."""
    conn = None
    cursor = None # Define cursor here to ensure it's closed in finally
    try:
        if not DATABASE_URL:
             raise ValueError("Database connection string not configured in environment variables.")
        conn = psycopg2.connect(DATABASE_URL)
        # Using server-side cursors can sometimes be more efficient for large results,
        # but standard cursors are fine for most web app interactions.
        # Using dictionary cursor is very convenient for accessing columns by name
        # You might need to install 'psycopg2-binary[extras]' or 'psycopg[binary,dict]' for DictCursor
        # Or use standard cursor and access by index (row[0], row[1])
        # cursor = conn.cursor(cursor_factory=psycopg2.extras.DictCursor) # Example with DictCursor
        cursor = conn.cursor() # Standard cursor
        logging.debug("Database connection obtained.")
        yield conn, cursor # Yield both connection and cursor
        conn.commit() # Commit transaction if no exceptions occurred
        logging.debug("Transaction committed.")
    except (Exception, psycopg2.DatabaseError) as error:
        logging.error(f"Database Error: {error}")
        if conn:
            conn.rollback() # Rollback transaction on error
            logging.warning("Transaction rolled back due to error.")
        raise # Re-raise the exception for Flask to handle
    finally:
        if cursor:
            cursor.close()
        if conn:
            conn.close()
            logging.debug("Database connection closed.")

# Example Usage in your Flask routes (using standard cursor):
# with get_connection() as (conn, cursor):
#     cursor.execute('SELECT id, username FROM "Users" WHERE id = %s', (user_id,))
#     user_row = cursor.fetchone()
#     if user_row:
#         user_id_val = user_row[0]
#         username_val = user_row[1]
#     # No explicit commit needed for SELECT
#
# with get_connection() as (conn, cursor):
#     cursor.execute('INSERT INTO "Posts" (user_id, content) VALUES (%s, %s) RETURNING id',
#                    (user_id, post_content))
#     new_post_id = cursor.fetchone()[0]
#     # Commit happens automatically when 'with' block exits without error