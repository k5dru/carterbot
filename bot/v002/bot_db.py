import sqlite3
import sys

class BotDB:
    def __init__(self, db_name='irc_bot_log.db'):
        self.conn = sqlite3.connect(db_name, check_same_thread=False)
        self.conn.commit()

    def _execute(self, query, params=None):
        c = self.conn.cursor()
        if params:
            c.execute(query, params)
        else:
            c.execute(query)
        self.conn.commit()
        c.close()

    def dbcreate(self):
        # Create tables
        self._execute('''
            CREATE TABLE IF NOT EXISTS messages (
                timestamp DATETIME DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                channel TEXT,
                nick TEXT,
                message TEXT,
                direction TEXT
            )
        ''')
        self._execute('''
            CREATE INDEX messages_timestamp_idx ON messages(timestamp DESC)
        ''')
        self._execute('''
            CREATE TABLE IF NOT EXISTS token_usage (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_tokens INTEGER NOT NULL,
                completion_tokens INTEGER NOT NULL,
                cost REAL NOT NULL,
                timestamp DATETIME DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                model TEXT NOT NULL
            )
        ''')
        self._execute('''
            CREATE TABLE IF NOT EXISTS control_settings (
                setting TEXT PRIMARY KEY,
                value TEXT NOT NULL
            )
        ''')
        self._execute('''
            CREATE TABLE IF NOT EXISTS request_response (
                request_time DATETIME DEFAULT(STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')),
                response_time DATETIME,
                channel TEXT,
                response_text TEXT,
                posted_time DATETIME
            )
        ''')
        self._execute('CREATE INDEX IF NOT EXISTS idx_response_time ON request_response (response_time DESC)')

        # Insert default settings
        default_settings = {
            'temperature': '0.7',
            'max_tokens': '50',
            'antiflood_delay': '1.0',
            'max_line_length': '400',
            'cost_per_mtok_large': '0.20',
            'model_large': 'meta-llama/Meta-Llama-3.1-8B-Instruct',
            'posting_enabled': 'True',
            'messages_since_activation': '999',
            'cost_per_mtok_small': '0.10',  # Retained for potential future use
            'model_small': 'meta-llama/Meta-Llama-3.1-8B-Instruct',  # Retained for potential future use
            'superusers': 'k5dru,w0ny',  # Updated superusers
            'memory': '200',  # Added default for memory
            'response_factor': '1.5',  # Added default for response_factor
            'response_factor_window': '120'  # Added default for response_factor_window
        }

        # Insert default settings only if they do not exist
        for setting, value in default_settings.items():
            self._execute('INSERT OR IGNORE INTO control_settings (setting, value) VALUES (?, ?)', (setting, value))

        print("Database created with default values.")

    def load_system_message(self):
        with open('system_message.txt', 'r') as file:
            return file.read()

    def update_system_message(self, new_message):
        with open('system_message.txt', 'w') as file:
            file.write(new_message)

    def load_setting(self, setting):
        c = self.conn.cursor()
        c.execute('SELECT value FROM control_settings WHERE setting = ?', (setting,))
        result = c.fetchone()
        c.close()
        if result:
            return result[0]
        else:
            return None

    def save_setting(self, setting, value):
        self._execute('INSERT OR REPLACE INTO control_settings (setting, value) VALUES (?, ?)', (setting, value))

    def load_all_settings(self):
        c = self.conn.cursor()
        c.execute('SELECT setting, value FROM control_settings')
        all_settings = c.fetchall()
        c.close()
        return dict(all_settings)

    def load_token_usage(self, model):
        c = self.conn.cursor()
        c.execute('SELECT SUM(prompt_tokens), SUM(completion_tokens), SUM(cost) FROM token_usage WHERE model = ?', (model,))
        usage = c.fetchone() or (0, 0, 0)
        c.close()
        return usage

    def log_message(self, channel, nick, message, direction):
        self._execute('''
            INSERT INTO messages (channel, nick, message, direction)
            VALUES (?, ?, ?, ?)
        ''', (channel, nick, message, direction))

    def log_token_usage(self, prompt_tokens, completion_tokens, cost, model):
        self._execute('''
            INSERT INTO token_usage (prompt_tokens, completion_tokens, cost, model)
            VALUES (?, ?, ?, ?)
        ''', (prompt_tokens, completion_tokens, cost, model))

    def get_recent_channel_messages(self, channel, since_timestamp):
        c = self.conn.cursor()
        c.execute('SELECT nick, message FROM messages WHERE channel = ? AND timestamp >= ? ORDER BY timestamp', (channel, since_timestamp))
        messages = c.fetchall()
        c.close()
        return messages

    def get_timestamp_of_nth_message(self, channel, n):
        c = self.conn.cursor()
        c.execute('SELECT timestamp FROM messages WHERE channel = ? ORDER BY timestamp DESC LIMIT ?', (channel, n))
        timestamps = c.fetchall()
        c.close()
        if timestamps and len(timestamps) >= n:
            return timestamps[n-1][0]
        else:
            return '1970-01-01 00:00:00.000000'  # Return a very old timestamp if there are less than n messages

    def request_response(self, channel):
        self._execute('INSERT INTO request_response (channel) VALUES (?)', (channel,))

    def get_pending_requests(self, channel):
        c = self.conn.cursor()
        if channel == None:
            c.execute('SELECT request_time, channel FROM request_response WHERE response_time IS NULL')
        else:
            c.execute('SELECT request_time, channel FROM request_response WHERE channel = ? AND response_time IS NULL', (channel,))
        requests = c.fetchall()
        c.close()
        return requests

    def has_pending_request(self, channel):
        c = self.conn.cursor()
        c.execute('SELECT request_time FROM request_response WHERE channel = ? AND response_time IS NULL ORDER BY request_time ASC LIMIT 1', (channel,))
        request = c.fetchone()
        c.close()
        return request is not None

    def store_response(self, channel, response):
        self._execute('''
            UPDATE request_response
            SET response_time = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW'), response_text = ?
            WHERE channel = ? AND response_time IS NULL
            ORDER BY request_time ASC
            LIMIT 1
        ''', (response, channel))

    def get_response(self, channel):
        c = self.conn.cursor()
        c.execute('SELECT response_text FROM request_response WHERE channel = ? AND response_time IS NOT NULL AND posted_time IS NULL ORDER BY request_time ASC LIMIT 1', (channel,))
        response = c.fetchone()
        c.close()
        if response:
            return response[0]
        else:
            return None

    def mark_response_posted(self, channel):
        self._execute('''
            UPDATE request_response
            SET posted_time = STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW')
            WHERE channel = ? AND response_time IS NOT NULL AND posted_time IS NULL
            ORDER BY request_time ASC
            LIMIT 1
        ''', (channel,))

    def pending_responses_count(self, channel):
        c = self.conn.cursor()
        c.execute('SELECT COUNT(*) FROM request_response WHERE channel = ? AND response_time IS NULL', (channel,))
        count = c.fetchone()[0]
        c.close()
        return count

    def responses_in_last_window(self, channel, window):
        c = self.conn.cursor()
        c.execute(f'''
            SELECT COUNT(*)
            FROM request_response
            WHERE channel = ? AND response_time IS NOT NULL AND response_time >= STRFTIME('%Y-%m-%d %H:%M:%f', 'NOW', '-{window} seconds')
        ''', (channel,))
        count = c.fetchone()[0]
        c.close()
        return count

# Check if the script is being run directly
if __name__ == '__main__':
    db = BotDB()
    db.dbcreate()  # only ever create database on manual invocation
