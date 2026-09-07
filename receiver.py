import numpy as np
import random

class VirtualReceiver:
    def __init__(self, num_bands=10):
        self.num_bands = num_bands
        self.current_band = 0
        self.hits = 0
        self.history = []

    def scan(self, t, band, env=None):
        """
        Scans specified band at time slot t.
        Accepts env as 3rd positional or keyword argument.
        """
        self.current_band = band
        is_active = 0
        
        if env is not None:
            if hasattr(env, 'is_active'):
                is_active = env.is_active(t, band)
            elif hasattr(env, 'grid'):
                is_active = int(env.grid[t, band])
        
        if is_active:
            self.hits += 1

        log_entry = {
            'time': t,
            'band': band,
            'result': is_active
        }
        self.history.append(log_entry)
        return is_active

    def scan_band(self, env, band, t):
        """Wrapper method for backward compatibility."""
        return self.scan(t, band, env)

    def reset(self):
        """Resets receiver metrics for a new benchmark run."""
        self.current_band = 0
        self.hits = 0
        self.history = []


def run_sequential_scan(env, num_bands=None, num_time_slots=None):
    if num_bands is None:
        num_bands = getattr(env, 'num_bands', 10)
    if num_time_slots is None:
        num_time_slots = getattr(env, 'num_time_slots', 200)

    receiver = VirtualReceiver(num_bands=num_bands)
    for t in range(num_time_slots):
        band_to_scan = t % num_bands
        receiver.scan(t, band_to_scan, env)
    return receiver


def run_random_scan(env, num_bands=None, num_time_slots=None):
    if num_bands is None:
        num_bands = getattr(env, 'num_bands', 10)
    if num_time_slots is None:
        num_time_slots = getattr(env, 'num_time_slots', 200)

    receiver = VirtualReceiver(num_bands=num_bands)
    for t in range(num_time_slots):
        band_to_scan = random.randint(0, num_bands - 1)
        receiver.scan(t, band_to_scan, env)
    return receiver