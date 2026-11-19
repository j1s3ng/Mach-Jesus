"""Frozen metric AST extracts; source IDs and hashes are in provenance.json."""
import numpy as np
from scipy.signal import butter, sosfiltfilt, welch
from types import SimpleNamespace

def percentile(x):
    return {str(q): float(np.percentile(x, q)) for q in (50, 90, 95, 99, 100)} if len(x) else None

def direction_changes(t, delta, excursion=0.005):
    """Alternating directional changes after >=excursion rad; cadence independent."""
    if len(delta) < 2:
        return []
    low = high = float(delta[0])
    li = hi = 0
    direction = 0
    events = []
    for i, x in enumerate(delta[1:], 1):
        if direction >= 0:
            if x > high:
                high = float(x)
                hi = i
            if high - x >= excursion:
                events.append({'extremum_index': hi, 'confirmed_index': i, 'direction_after': -1})
                direction = -1
                low = float(x)
                li = i
        if direction <= 0:
            if x < low:
                low = float(x)
                li = i
            if x - low >= excursion:
                events.append({'extremum_index': li, 'confirmed_index': i, 'direction_after': 1})
                direction = 1
                high = float(x)
                hi = i
    for e in events:
        e['extremum_race_s'] = float(t[e.pop('extremum_index')])
        e['confirmed_race_s'] = float(t[e.pop('confirmed_index')])
    return events

def metric(a, mask):
    t = a['time']
    delta = a['steer']
    dt = np.diff(t)
    pair = mask[1:] & mask[:-1] & (dt > 0.5 * a['median_dt']) & (dt < 1.5 * a['median_dt'])
    indices = np.flatnonzero(mask)
    duration = float(dt[pair].sum())
    changes = np.diff(delta)
    rate = changes[pair] / dt[pair]
    triple = pair[1:] & pair[:-1]
    allrate = changes / dt
    acceleration = np.diff(allrate)[triple] / ((dt[1:] + dt[:-1])[triple] / 2.0)
    events = []
    if len(indices):
        groups = np.split(indices, np.flatnonzero(np.diff(indices) > 1) + 1)
        for g in groups:
            events.extend(direction_changes(t[g], delta[g]))
    rapid = sum((0.0 < b['extremum_race_s'] - c['extremum_race_s'] <= 1.0 for c, b in zip(events[:-1], events[1:])))
    result = {'sample_count': int(mask.sum()), 'continuous_interval_duration_s': duration, 'command_total_variation_rad': float(np.abs(changes[pair]).sum()), 'total_variation_rad_per_s': float(np.abs(changes[pair]).sum() / duration) if duration else None, 'absolute_steering_rate_radps': percentile(np.abs(rate)), 'steering_rate_rms_radps': float(np.sqrt(np.mean(rate ** 2))) if len(rate) else None, 'absolute_steering_acceleration_radps2': percentile(np.abs(acceleration)), 'steering_acceleration_rms_radps2': float(np.sqrt(np.mean(acceleration ** 2))) if len(acceleration) else None, 'direction_changes_with_0point005rad_excursion': len(events), 'rapid_successive_direction_changes_within1s': rapid, 'highpass_above1hz_command_rms_rad': float(np.sqrt(np.mean(a['highpass'][mask] ** 2))) if mask.any() else None, 'mean_speed_mps': float(a['speed'][mask].mean()) if mask.any() else None, 'max_abs_path_cross_error_m': float(np.abs(a['cross'][mask]).max()) if mask.any() else None, 'nominal_capture_fraction': float((a['geometry'][mask] == 'variable_nominal_path').mean()) if mask.any() else None, 'lateral_proxy_limit_active_fraction': float(a['limited'][mask].mean()) if mask.any() else None, 'physics_slew_limit_exceeded_intervals': int((np.abs(rate) > 1.0 + 1e-06).sum()), 'method_switches': int((a['method'][1:] != a['method'][:-1])[pair].sum()), 'geometry_switches': int((a['geometry'][1:] != a['geometry'][:-1])[pair].sum())}
    return result

def rms(values):
    values = np.asarray(values, float)
    return float(np.sqrt(np.mean(values * values))) if len(values) else None

def highpass(values, fs):
    values = np.asarray(values, float)
    if len(values) < 32:
        return np.zeros_like(values)
    return sosfiltfilt(butter(3, 1.0, fs=fs, btype='highpass', output='sos'), values)

def band_power(values, fs, low=1.0, high=6.0):
    if len(values) < 16:
        return None
    frequency, power = welch(values, fs=fs, nperseg=min(len(values), int(8 * fs)))
    mask = (frequency >= low) & (frequency <= high)
    return float(np.trapz(power[mask], frequency[mask])) if mask.sum() >= 2 else 0.0

def motion_metrics(a, mask):
    base = steering.metric(a, mask)
    ids = np.flatnonzero(mask)
    if not len(ids):
        return {'steering': base, 'body': None}
    body = {'measured_yaw_rate_rms_radps': rms(a['yaw_rate'][mask]), 'measured_yaw_acceleration_rms_radps2': rms(a['yaw_accel'][mask]), 'yaw_rate_above1hz_rms_radps': rms(a['yaw_rate_highpass'][mask]), 'XY_difference_body_lateral_acceleration_rms_mps2': rms(a['body_lateral_accel'][mask]), 'XY_difference_body_lateral_jerk_rms_mps3': rms(a['body_lateral_jerk'][mask]), 'body_lateral_acceleration_above1hz_rms_mps2': rms(a['body_lat_highpass'][mask]), 'yaw_rate_band_1to6hz_power_rad2ps2': None, 'steering_band_1to6hz_power_rad2': None}
    dt = np.diff(a['time'])
    pair = mask[:-1] & mask[1:]
    triple = pair[:-1] & pair[1:]
    forward_rate = np.diff(a['yaw']) / dt
    forward_acceleration = np.diff(forward_rate) / (0.5 * (dt[1:] + dt[:-1]))
    body['forward_difference_yaw_rate_rms_radps'] = rms(forward_rate[pair])
    body['forward_difference_yaw_acceleration_rms_radps2'] = rms(forward_acceleration[triple])
    driven = float(np.linalg.norm(np.diff(a['position'], axis=0), axis=1)[pair].sum())
    body['actual_XY_driven_distance_m'] = driven
    body['steering_variation_rad_per_driven_km'] = 1000.0 * base['command_total_variation_rad'] / driven if driven else None
    if np.all(np.diff(ids) == 1):
        body['yaw_rate_band_1to6hz_power_rad2ps2'] = band_power(a['yaw_rate'][mask], 1.0 / a['median_dt'])
        body['steering_band_1to6hz_power_rad2'] = band_power(a['steer'][mask], 1.0 / a['median_dt'])
    return {'steering': base, 'body': body}

def derive_arrays(a):
    dt = np.diff(a['time'])
    a['median_dt'] = float(np.median(dt))
    if not abs(a['median_dt'] - 0.02) < 1e-07 or not np.all(abs(dt - 0.02) < 1e-07):
        raise ValueError('This evaluator requires genuine uniform50Hz active telemetry; no upsampling')
    fs = 1.0 / a['median_dt']
    a['highpass'] = highpass(a['steer'], fs)
    a['yaw_rate'] = np.gradient(a['yaw'], a['time'])
    a['yaw_accel'] = np.gradient(a['yaw_rate'], a['time'])
    velocity = np.gradient(a['position'], a['time'], axis=0)
    acceleration = np.gradient(velocity, a['time'], axis=0)
    normal = np.column_stack((-np.sin(a['yaw']), np.cos(a['yaw'])))
    a['body_lateral_accel'] = np.sum(acceleration * normal, axis=1)
    a['body_lateral_jerk'] = np.gradient(a['body_lateral_accel'], a['time'])
    a['yaw_rate_highpass'] = highpass(a['yaw_rate'], fs)
    a['body_lat_highpass'] = highpass(a['body_lateral_accel'], fs)
    return a

steering = SimpleNamespace(metric=metric)
