import os
import sqlite3
import json
from datetime import datetime

# Try importing official Supabase Client
try:
    from supabase import create_client, Client
    SUPABASE_AVAILABLE = True
except ImportError:
    SUPABASE_AVAILABLE = False

# Load local environment variables from .env if present
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

DB_FILE = 'attendance.db'

# Read Supabase settings
SUPABASE_URL = os.environ.get('SUPABASE_URL')
SUPABASE_KEY = os.environ.get('SUPABASE_KEY')

# Determine DB Type
DB_TYPE = 'SQLITE'
if SUPABASE_AVAILABLE and SUPABASE_URL and SUPABASE_KEY:
    try:
        supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)
        DB_TYPE = 'SUPABASE'
    except Exception:
        DB_TYPE = 'SQLITE'

def init_db(db_path=DB_FILE):
    """
    Initializes database schema.
    For Supabase, table schema must be created in the Supabase Dashboard SQL Editor.
    For SQLite, we automatically construct local tables.
    """
    if DB_TYPE == 'SUPABASE':
        # Supabase DDL operations are done via their web interface
        pass
    else:
        # Local SQLite schema
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id TEXT PRIMARY KEY,
                name TEXT NOT NULL,
                face_encoding TEXT NOT NULL,
                photo_path TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                user_id TEXT NOT NULL,
                name TEXT NOT NULL,
                date TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                FOREIGN KEY (user_id) REFERENCES users (id) ON DELETE CASCADE,
                UNIQUE(user_id, date)
            )
        ''')
        
        conn.commit()
        conn.close()

def add_user(user_id, name, face_encoding, photo_path=None, db_path=DB_FILE):
    """Registers or updates a user profile."""
    if hasattr(face_encoding, 'tolist'):
        encoding_list = face_encoding.tolist()
    else:
        encoding_list = list(face_encoding)
        
    encoding_json = json.dumps(encoding_list)
    
    if DB_TYPE == 'SUPABASE':
        data = {
            "id": user_id,
            "name": name,
            "face_encoding": encoding_json,
            "photo_path": photo_path
        }
        supabase.table("users").upsert(data).execute()
    else:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('''
            INSERT OR REPLACE INTO users (id, name, face_encoding, photo_path)
            VALUES (?, ?, ?, ?)
        ''', (user_id, name, encoding_json, photo_path))
        conn.commit()
        conn.close()

def remove_user(user_id, db_path=DB_FILE):
    """Deletes a user from the directory."""
    if DB_TYPE == 'SUPABASE':
        try:
            supabase.table("users").delete().eq("id", user_id).execute()
            return True
        except Exception:
            return False
    else:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute('DELETE FROM users WHERE id = ?', (user_id,))
        deleted = cursor.rowcount > 0
        conn.commit()
        conn.close()
        return deleted

def get_all_users(db_path=DB_FILE):
    """Retrieves all registered users."""
    if DB_TYPE == 'SUPABASE':
        res = supabase.table("users").select("id, name, face_encoding, photo_path, created_at").execute()
        users = []
        for row in res.data:
            user_dict = dict(row)
            user_dict['face_encoding'] = json.loads(user_dict['face_encoding'])
            users.append(user_dict)
        return users
    else:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        cursor.execute('SELECT id, name, face_encoding, photo_path, created_at FROM users')
        rows = cursor.fetchall()
        conn.close()
        
        users = []
        for row in rows:
            user_dict = dict(row)
            user_dict['face_encoding'] = json.loads(user_dict['face_encoding'])
            users.append(user_dict)
        return users

def log_attendance(user_id, name, db_path=DB_FILE):
    """Logs check-in, handling duplicate check-ins on both SQLite and Supabase."""
    now = datetime.now()
    date_str = now.strftime('%Y-%m-%d')
    timestamp_str = now.strftime('%Y-%m-%d %H:%M:%S')
    
    if DB_TYPE == 'SUPABASE':
        data = {
            "user_id": user_id,
            "name": name,
            "date": date_str,
            "timestamp": timestamp_str
        }
        try:
            supabase.table("attendance").insert(data).execute()
            success = True
            message = f"Attendance logged for {name} ({user_id}) at {timestamp_str.split()[1]}."
        except Exception as e:
            err_msg = str(e).lower()
            if "unique" in err_msg or "duplicate" in err_msg or "409" in err_msg:
                success = False
                message = f"{name} ({user_id}) has already logged attendance for today ({date_str})."
            else:
                success = False
                message = f"Supabase Error: {str(e)}"
    else:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        try:
            cursor.execute('''
                INSERT INTO attendance (user_id, name, date, timestamp)
                VALUES (?, ?, ?, ?)
            ''', (user_id, name, date_str, timestamp_str))
            conn.commit()
            success = True
            message = f"Attendance logged for {name} ({user_id}) at {timestamp_str.split()[1]}."
        except sqlite3.IntegrityError:
            success = False
            message = f"{name} ({user_id}) has already logged attendance for today ({date_str})."
        finally:
            conn.close()
            
    return success, message

def get_attendance_logs(start_date=None, end_date=None, db_path=DB_FILE):
    """Retrieves check-in logs with optional date range filtering."""
    if DB_TYPE == 'SUPABASE':
        query = supabase.table("attendance").select("id, user_id, name, date, timestamp")
        if start_date:
            query = query.gte("date", start_date)
        if end_date:
            query = query.lte("date", end_date)
        res = query.order("timestamp", desc=True).execute()
        return res.data
    else:
        conn = sqlite3.connect(db_path)
        conn.row_factory = sqlite3.Row
        cursor = conn.cursor()
        
        query = 'SELECT id, user_id, name, date, timestamp FROM attendance'
        params = []
        conditions = []
        
        if start_date:
            conditions.append('date >= ?')
            params.append(start_date)
        if end_date:
            conditions.append('date <= ?')
            params.append(end_date)
            
        if conditions:
            query += ' WHERE ' + ' AND '.join(conditions)
            
        query += ' ORDER BY timestamp DESC'
        
        cursor.execute(query, params)
        rows = cursor.fetchall()
        conn.close()
        
        return [dict(row) for row in rows]
