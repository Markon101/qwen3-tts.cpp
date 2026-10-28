"""Read-only board energy counters; missing counters stay null, never zero.

Battery integration estimates whole-device power, not isolated HTP energy.
"""
import pathlib
import time
BATTERY = pathlib.Path('/sys/class/power_supply/battery')

def battery_sample():
    row = {'monotonic_s': time.monotonic()}
    for key in ('current_now', 'voltage_now', 'power_now', 'charge_counter', 'energy_now', 'status', 'capacity'):
        try:
            value = (BATTERY / key).read_text().strip()
            row[key] = value if key == 'status' else int(value)
        except (OSError, ValueError):
            row[key] = None
    return row

def energy_summary(rows):
    # Linux power_supply current/voltage units are uA/uV; record raw values too.
    # Never absolutize charging current and call it model consumption.
    points = []
    for r in rows:
        if (r.get('status') or '').lower() != 'discharging':
            continue
        current, voltage = r.get('current_now'), r.get('voltage_now')
        if current is not None and voltage is not None:
            points.append((r['monotonic_s'], abs(current * voltage) / 1e12))
    intervals = [(a,b) for a,b in zip(points,points[1:]) if 0 < b[0]-a[0] < 1]
    joules = sum((b[0]-a[0])*(a[1]+b[1])/2 for a,b in intervals)
    duration = sum(b[0]-a[0] for a,b in intervals)
    return {'scope': 'whole_device_battery_estimate', 'joules': joules if intervals else None, 'integrated_duration_s': duration,
            'samples': len(points), 'charging_excluded': True,
            'limitation': 'Battery sensor cadence and background activity prevent component energy attribution.'}
