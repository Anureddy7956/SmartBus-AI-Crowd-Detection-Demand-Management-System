"""
bus_manager.py — Bus Fleet Manager
Handles dispatch decisions, fleet status simulation, signal threshold logic.
"""

import logging
import random
import time
from datetime import datetime

log = logging.getLogger('BusManager')

# Auto-dispatch if passenger signals exceed this in 15 min
SIGNAL_THRESHOLD = 8

# Simulated ETA range (minutes) for dispatched buses
ETA_MIN, ETA_MAX = 5, 15


class BusManager:
    def __init__(self, db):
        self.db         = db
        self._fleet     = {}   # runtime fleet state overlay
        self._sim_tick  = 0

    def dispatch_bus(self, stop_id: str, reason: str = 'manual') -> dict:
        """Assign nearest available bus to stop. Returns dispatch result."""
        buses  = self.db.get_live_snapshot()['buses']
        avail  = [b for b in buses if b['status'] == 'active']
        if not avail:
            log.warning('No buses available for dispatch!')
            return {'ok': False, 'message': 'No buses available'}

        bus   = random.choice(avail)
        eta   = random.randint(ETA_MIN, ETA_MAX)
        result = {
            'ok':      True,
            'bus_id':  bus['id'],
            'route':   bus['route'],
            'eta_min': eta,
            'reason':  reason,
            'stop_id': stop_id,
            'ts':      datetime.now().isoformat()
        }
        # Update runtime overlay
        self._fleet[bus['id']] = {**bus, 'next_stop': stop_id, 'eta_min': eta, 'status': 'dispatched'}
        log.info(f"Bus {bus['id']} dispatched to {stop_id} (ETA {eta}min, reason: {reason})")
        return result

    def auto_dispatch(self, stop_id: str, reason: str):
        """Called by detector on overcrowding — dispatches + notifies DB."""
        result = self.dispatch_bus(stop_id, reason)
        self.db.log_dispatch(stop_id, result)
        self.db.add_alert({
            'type':    'dispatch',
            'stop_id': stop_id,
            'message': f"Auto-dispatch: {result.get('bus_id')} → {stop_id} (ETA {result.get('eta_min')}min)",
            'count':   0,
            'ts':      datetime.now().isoformat()
        })

    def check_signal_threshold(self, stop_id: str, count: int):
        """If passenger signals exceed threshold, auto-dispatch."""
        if count >= SIGNAL_THRESHOLD:
            log.info(f'{stop_id}: {count} signals — triggering auto-dispatch')
            self.auto_dispatch(stop_id, 'passenger_signals')

    def get_fleet_status(self) -> list:
        """Returns current fleet list with ETA simulation."""
        buses = self.db.get_live_snapshot()['buses']
        result = []
        for bus in buses:
            overlay = self._fleet.get(bus['id'], {})
            # Simulate ETA countdown
            eta = overlay.get('eta_min', random.randint(2, 15))
            seats_free = random.randint(5, bus['capacity'] - 10)
            result.append({
                **bus,
                'eta_min':    eta,
                'seats_free': seats_free,
                'onboard':    bus['capacity'] - seats_free,
                'status':     overlay.get('status', 'active'),
                'next_stop':  overlay.get('next_stop', '—'),
            })
        return result
