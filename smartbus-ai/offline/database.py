"""
database.py — SQLite Database Layer
Lightweight persistent storage for crowd data, alerts, signals, fleet.

Why SQLite?
  - Zero-config, serverless — no postgres/mysql process eating RAM
  - Perfect for single-Pi local data
  - WAL mode = concurrent reads while writing (Flask threads safe)
  - Stores ~1 year of data in <50MB
"""

import sqlite3
import json
import logging
import threading
from datetime import datetime, timedelta
from pathlib import Path

log = logging.getLogger('Database')
DB_PATH = 'smartbus.db'

# Thread-local connections (SQLite is not thread-safe with shared connections)
_local = threading.local()


def get_conn():
    if not hasattr(_local, 'conn') or _local.conn is None:
        _local.conn = sqlite3.connect(DB_PATH, check_same_thread=False)
        _local.conn.row_factory = sqlite3.Row
        _local.conn.execute('PRAGMA journal_mode=WAL')
        _local.conn.execute('PRAGMA synchronous=NORMAL')
    return _local.conn


# ── Default stops (Bengaluru routes)
DEFAULT_STOPS = [
    {'id': 'stop_majestic',    'name': 'Majestic',      'route': '201', 'lat': 12.977, 'lng': 77.572},
    {'id': 'stop_shivajinagar','name': 'Shivajinagar',  'route': '201', 'lat': 12.985, 'lng': 77.600},
    {'id': 'stop_mgroad',      'name': 'MG Road',       'route': '201', 'lat': 12.975, 'lng': 77.607},
    {'id': 'stop_indiranagar', 'name': 'Indiranagar',   'route': '219', 'lat': 12.979, 'lng': 77.639},
    {'id': 'stop_koramangala', 'name': 'Koramangala',   'route': '252', 'lat': 12.936, 'lng': 77.626},
    {'id': 'stop_btm',         'name': 'BTM Layout',    'route': '252', 'lat': 12.916, 'lng': 77.610},
    {'id': 'stop_hsr',         'name': 'HSR Layout',    'route': '219', 'lat': 12.912, 'lng': 77.641},
]

DEFAULT_BUSES = [
    {'id': 'KA-01-F-2287', 'route': '201', 'capacity': 48, 'status': 'active'},
    {'id': 'KA-01-F-2291', 'route': '201', 'capacity': 48, 'status': 'active'},
    {'id': 'KA-01-F-2295', 'route': '219', 'capacity': 54, 'status': 'active'},
    {'id': 'KA-01-F-2301', 'route': '252', 'capacity': 60, 'status': 'active'},
    {'id': 'KA-01-F-2315', 'route': '202', 'capacity': 54, 'status': 'active'},
]


class Database:
    def init(self):
        conn = get_conn()
        c    = conn.cursor()

        c.executescript("""
        CREATE TABLE IF NOT EXISTS stops (
            id          TEXT PRIMARY KEY,
            name        TEXT,
            route       TEXT,
            lat         REAL,
            lng         REAL,
            crowd_count INTEGER DEFAULT 0,
            crowd_level TEXT    DEFAULT 'low',
            updated_at  TEXT
        );

        CREATE TABLE IF NOT EXISTS crowd_history (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            stop_id     TEXT,
            count       INTEGER,
            level       TEXT,
            ts          TEXT
        );
        CREATE INDEX IF NOT EXISTS idx_ch_stop ON crowd_history(stop_id, ts);

        CREATE TABLE IF NOT EXISTS buses (
            id          TEXT PRIMARY KEY,
            route       TEXT,
            capacity    INTEGER,
            status      TEXT DEFAULT 'active',
            next_stop   TEXT,
            eta_min     INTEGER DEFAULT 0,
            onboard     INTEGER DEFAULT 0
        );

        CREATE TABLE IF NOT EXISTS alerts (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            type        TEXT,
            stop_id     TEXT,
            message     TEXT,
            count       INTEGER,
            resolved    INTEGER DEFAULT 0,
            ts          TEXT
        );

        CREATE TABLE IF NOT EXISTS signals (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            stop_id     TEXT,
            ts          TEXT
        );

        CREATE TABLE IF NOT EXISTS dispatches (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            stop_id     TEXT,
            bus_id      TEXT,
            reason      TEXT,
            ts          TEXT
        );
        """)
        conn.commit()

        # Seed stops if empty
        row = c.execute('SELECT COUNT(*) FROM stops').fetchone()[0]
        if row == 0:
            c.executemany(
                'INSERT OR IGNORE INTO stops(id,name,route,lat,lng) VALUES(:id,:name,:route,:lat,:lng)',
                DEFAULT_STOPS
            )
        row2 = c.execute('SELECT COUNT(*) FROM buses').fetchone()[0]
        if row2 == 0:
            c.executemany(
                'INSERT OR IGNORE INTO buses(id,route,capacity) VALUES(:id,:route,:capacity)',
                DEFAULT_BUSES
            )
        conn.commit()
        log.info('Database initialised')

    # ── Live snapshot for dashboard
    def get_live_snapshot(self) -> dict:
        conn = get_conn()
        stops = [dict(r) for r in conn.execute('SELECT * FROM stops ORDER BY name').fetchall()]
        buses = [dict(r) for r in conn.execute('SELECT * FROM buses').fetchall()]
        alerts = [dict(r) for r in conn.execute(
            "SELECT * FROM alerts WHERE resolved=0 ORDER BY ts DESC LIMIT 10"
        ).fetchall()]
        total_crowd = sum(s['crowd_count'] for s in stops)
        signals_today = conn.execute(
            "SELECT COUNT(*) FROM signals WHERE ts >= date('now')"
        ).fetchone()[0]
        return {
            'stops':        stops,
            'buses':        buses,
            'alerts':       alerts,
            'total_crowd':  total_crowd,
            'signals_today': signals_today,
            'ts':           datetime.now().isoformat()
        }

    def get_stop_detail(self, stop_id: str) -> dict:
        conn = get_conn()
        row  = conn.execute('SELECT * FROM stops WHERE id=?', (stop_id,)).fetchone()
        return dict(row) if row else {}

    def update_stop_crowd(self, stop_id: str, count: int, level: str, ts: str):
        conn = get_conn()
        conn.execute(
            'UPDATE stops SET crowd_count=?, crowd_level=?, updated_at=? WHERE id=?',
            (count, level, ts, stop_id)
        )
        conn.execute(
            'INSERT INTO crowd_history(stop_id, count, level, ts) VALUES(?,?,?,?)',
            (stop_id, count, level, ts)
        )
        conn.commit()

    def add_alert(self, alert: dict):
        conn = get_conn()
        conn.execute(
            'INSERT INTO alerts(type,stop_id,message,count,ts) VALUES(?,?,?,?,?)',
            (alert['type'], alert['stop_id'], alert['message'], alert.get('count',0), alert['ts'])
        )
        conn.commit()

    def get_active_alerts(self) -> list:
        conn = get_conn()
        return [dict(r) for r in conn.execute(
            "SELECT * FROM alerts WHERE resolved=0 ORDER BY ts DESC LIMIT 20"
        ).fetchall()]

    def log_passenger_signal(self, stop_id: str):
        conn = get_conn()
        conn.execute('INSERT INTO signals(stop_id,ts) VALUES(?,?)',
                     (stop_id, datetime.now().isoformat()))
        conn.commit()

    def get_signal_count(self, stop_id: str) -> int:
        conn = get_conn()
        return conn.execute(
            "SELECT COUNT(*) FROM signals WHERE stop_id=? AND ts >= datetime('now','-15 minutes')",
            (stop_id,)
        ).fetchone()[0]

    def log_dispatch(self, stop_id: str, result: dict):
        conn = get_conn()
        conn.execute(
            'INSERT INTO dispatches(stop_id,bus_id,reason,ts) VALUES(?,?,?,?)',
            (stop_id, result.get('bus_id',''), result.get('reason',''), datetime.now().isoformat())
        )
        conn.commit()

    def get_heatmap_data(self) -> dict:
        """Returns hourly crowd averages per stop for last 24h."""
        conn  = get_conn()
        stops = [dict(r) for r in conn.execute('SELECT id,name FROM stops').fetchall()]
        result = {}
        for stop in stops:
            rows = conn.execute("""
                SELECT strftime('%H', ts) as hour, AVG(count) as avg_count
                FROM crowd_history
                WHERE stop_id=? AND ts >= datetime('now','-24 hours')
                GROUP BY hour ORDER BY hour
            """, (stop['id'],)).fetchall()
            result[stop['id']] = {
                'name': stop['name'],
                'hourly': {r['hour']: round(r['avg_count']) for r in rows}
            }
        return result

    def get_stats_summary(self) -> dict:
        conn = get_conn()
        total_crowd = conn.execute('SELECT SUM(crowd_count) FROM stops').fetchone()[0] or 0
        dispatches  = conn.execute("SELECT COUNT(*) FROM dispatches WHERE ts >= date('now')").fetchone()[0]
        alerts_today = conn.execute("SELECT COUNT(*) FROM alerts WHERE ts >= date('now')").fetchone()[0]
        signals_today = conn.execute("SELECT COUNT(*) FROM signals WHERE ts >= date('now')").fetchone()[0]
        return {
            'total_crowd': total_crowd,
            'dispatches_today': dispatches,
            'alerts_today': alerts_today,
            'signals_today': signals_today,
        }
