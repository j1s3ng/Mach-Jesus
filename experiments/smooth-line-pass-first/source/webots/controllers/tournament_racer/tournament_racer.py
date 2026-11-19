"""One physically driven race entrant, with its own measured perception stream.

Controller arguments are ``car_id absolute_config_path``. This controller never
looks up opponent nodes, changes a vehicle pose, or terminates the simulation.
The race supervisor owns standings and the separate ground-truth collision audit.
"""
from __future__ import annotations

import hashlib
import json
import math
import sys
import time
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np
from vehicle import Car

from race_lidar.hotlap_driver import RacingReference, PurePursuit, longitudinal_control, wrap_angle
from race_lidar.learned_driver import LearnedDriver
from race_lidar.race_policy import (RacePolicy, SignedProgressTracker, TournamentTraffic,
                                    policy_state, shared_guard)
from race_lidar.remote import remote_detect
from race_lidar.race_control import predictive_guard, parking_plan, combine_observations
from race_lidar.lidar_hazards import LidarHazardChannel
from race_lidar import lidar_hazards as geometry_module
from race_lidar import race_policy as policy_module, race_control as control_module
from race_lidar import hotlap_driver as hotlap_module, learned_driver as learned_module
from race_lidar.rich_driver import RichDriver,MODEL_FORMAT as RICH_FORMAT
from race_lidar import rich_driver as rich_module
from race_lidar import remote as remote_module
from race_lidar import aggressive_hybrid as aggressive_module, demo_chase as chase_module
from race_lidar import smooth_racing_line as line_module


def pose_state(node):
    position = np.asarray(node.getPosition(), float)
    rotation = np.asarray(node.getOrientation(), float).reshape(3, 3)
    velocity = np.asarray(node.getVelocity(), float)
    yaw = math.atan2(rotation[1, 0], rotation[0, 0])
    speed = float(np.linalg.norm(velocity[:2]))
    forward = float(np.dot(velocity[:2], [math.cos(yaw), math.sin(yaw)]))
    if not np.isfinite(np.r_[position, rotation.ravel(), speed, forward]).all():
        raise RuntimeError("Nonfinite physical vehicle state")
    return position, rotation, yaw, speed, forward


def entrant_config(config, car_id):
    entrants = config.get("cars", config.get("entrants", []))
    if isinstance(entrants, dict):
        return dict(entrants[car_id])
    return dict(next(entry for entry in entrants if str(entry.get("car_id", entry.get("id", ""))) == car_id))


def atomic_json(path, payload):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
    temporary.replace(path)


def lidar_points(lidar, position, rotation, yaw):
    raw = np.asarray([(p.x, p.y, p.z) for p in lidar.getPointCloud()], np.float32)
    raw = raw[np.isfinite(raw).all(axis=1)]
    c, s = math.cos(yaw), math.sin(yaw)
    level_rotation = np.array([[c, -s, 0.], [s, c, 0.], [0., 0., 1.]])
    body_to_ground = level_rotation.T @ rotation
    xyz = (raw + [1.6, 0., 1.2]) @ body_to_ground.T + [0., 0., position[2]]
    return np.column_stack((xyz, np.ones(len(xyz), np.float32))).astype(np.float32)


def video_frame(camera, car_id, policy_name, elapsed, speed, progress, length, lane, tracks, completed):
    from PIL import Image, ImageDraw, ImageFont
    rgba = np.frombuffer(camera.getImage(), np.uint8).reshape(camera.getHeight(), camera.getWidth(), 4)
    image = Image.fromarray(rgba[:, :, [2, 1, 0]].copy())
    draw = ImageDraw.Draw(image)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Menlo.ttc", 13)
    except OSError:
        font = ImageFont.load_default()
    draw.rectangle((0, 0, image.width, 65), fill=(8, 15, 25))
    minutes, seconds = divmod(max(0., elapsed), 60.)
    draw.text((10, 5), f"{car_id} | {policy_name} | {int(minutes)}:{seconds:06.3f}", font=font, fill=(245, 247, 250))
    draw.text((10, 26), f"{speed*3.6:5.1f} km/h | {100*max(0., progress)/length:5.1f}% | lane {lane:+.1f}m", font=font, fill=(100, 235, 220))
    draw.text((10, 46), f"{'FINISHED' if completed else 'RACING'} | own LiDAR -> VoxelNeXt | {len(tracks)} tracked cars", font=font, fill=(245, 210, 90))
    return np.asarray(image)


def main():
    if len(sys.argv) != 3:
        raise ValueError("tournament_racer requires controllerArgs [car_id, config_path]")
    car_id, config_path = sys.argv[1], Path(sys.argv[2])
    config = json.loads(config_path.read_text())
    entry = entrant_config(config, car_id)
    output = Path(entry.get("output_dir", Path(config["output_dir"]) / "cars" / car_id))
    output.mkdir(parents=True, exist_ok=True)
    if (output / "control_trace.jsonl").exists():
        raise ValueError(f"Choose a fresh race output directory: {output}")
    driver = Car()
    car = driver.getSelf()
    step_ms = int(driver.getBasicTimeStep())
    dt = step_ms / 1000.
    sensor_ms = int(config.get("sensor_period_ms", 200))
    video_ms = int(config.get("video_period_ms", 100))
    telemetry_ms = int(config.get("telemetry_period_ms", 100))
    if any(period < step_ms or period % step_ms for period in (sensor_ms, video_ms, telemetry_ms)):
        raise ValueError("Sensor, video and telemetry periods must be physics-step multiples")
    sensor_stride, video_stride, telemetry_stride = (period // step_ms for period in (sensor_ms, video_ms, telemetry_ms))
    reference_path = Path(config.get("reference_path", ROOT / ".local/f1-reference/reference.npz"))
    limits = {**config.get("limits", {}), **entry.get("limits", {})}
    reference = RacingReference.from_npz(reference_path,
        ignore_source_speed=bool(entry.get("ignore_reference_source_speed", False)),
        max_speed=float(limits.get("max_speed_mps", config.get("max_speed_mps", 60.))),
        lateral_accel=float(limits.get("lateral_accel_mps2", config.get("lateral_accel_mps2", 8.))),
        brake_decel=float(limits.get("brake_decel_mps2", config.get("brake_decel_mps2", 8.))),
        acceleration=float(limits.get("acceleration_mps2", config.get("acceleration_mps2", 5.))))
    racing_line = None
    racing_line_sha = None
    if entry.get('racing_line_plan'):
        line_path = Path(entry['racing_line_plan'])
        racing_line = line_module.SmoothRacingLine.from_npz(reference, line_path,
            expected_sha256=entry['racing_line_plan_sha256'])
        racing_line_sha = hashlib.sha256(line_path.read_bytes()).hexdigest()
        atomic_json(output / 'racing_line_report.json', racing_line.report())
    start_progress = float(entry.get("start_progress_m", 0.))
    lane_offset = float(entry.get("start_lane_m", 0.))
    start_time = float(config.get("start_time_s", 1.))
    laps = int(config.get("laps", 1))
    maximum_seconds = float(config.get("max_sim_seconds", 480.))
    track_limit = float(config.get("track_error_limit_m", 6.))
    target_progress = reference.length * laps
    predictive = config.get('control_version') == 'predictive_v2'
    park_after_finish = config.get('post_finish_mode') == 'park'
    parked, parking_side, parking_station = False, None, None
    if laps < 1 or maximum_seconds <= start_time or not np.isfinite([start_progress, lane_offset, start_time]).all():
        raise ValueError("Invalid race distance, start or time limit")
    policy_spec = entry.get("policy", "reference")
    if isinstance(policy_spec, dict):
        policy_name = str(policy_spec.get("name", policy_spec.get("type", "neural")))
        model_path = policy_spec.get("model_path", entry.get("model_path"))
    else:
        policy_name, model_path = str(policy_spec), entry.get("model_path")
    learned = None
    if model_path:
        with np.load(model_path,allow_pickle=False) as model_data:
            rich = json.loads(str(model_data['metadata_json'].item())).get('format') == RICH_FORMAT
        learned = RichDriver.from_npz(model_path) if rich else LearnedDriver.from_npz(model_path)
    if learned is None and ("neural" in policy_name or "learned" in policy_name):
        raise ValueError(f"Learned entrant {car_id} must specify model_path")
    policy = RacePolicy(reference, model=learned, name=policy_name, seed=int(entry.get("seed", config.get("seed", 0))))
    aggressive = aggressive_module.AggressiveHybrid(entry['aggressive_hybrid'], racing_line=racing_line) if entry.get('aggressive_hybrid') else None
    flat_out = bool(entry.get('flat_out_no_lift', False))
    actuator_override = {'active': False}
    traffic = TournamentTraffic(reference,rapid_hazard_confidence=float(config.get('rapid_hazard_confidence',.35))) if predictive else TournamentTraffic(reference)
    geometry = LidarHazardChannel(reference) if predictive and config.get('geometry_hazards_enabled',True) else None
    geometry_observation = {'geometry_hazards':[]}
    tracker = SignedProgressTracker(reference, start_progress)
    pursuit = PurePursuit(wheelbase=float(driver.getWheelbase()), max_steer=.50)
    wheel_radius = float(driver.getRearWheelRadius())
    endpoint = str(entry.get("vision_url", config.get("vision_url", "")))
    perception_enabled = bool(config.get("perception_enabled", bool(endpoint)))
    if perception_enabled and not endpoint:
        raise ValueError("perception_enabled requires a vision_url")
    detector_sha,detector_config_sha=None,None
    if perception_enabled and predictive:
        with urllib.request.urlopen(endpoint.rstrip('/')+'/health',timeout=30.) as response:
            health=json.load(response)
        detector_sha=health.get('model',{}).get('checkpoint_sha256')
        detector_config_sha=health.get('model',{}).get('config_sha256')
        expected=config.get('expected_detector_checkpoint_sha256')
        if expected and detector_sha!=expected:
            raise RuntimeError('Detector service does not match this frozen experiment checkpoint')
    record_video = bool(entry.get("record_video", config.get("record_video", False)))
    lidar, camera, video = None, None, None
    if perception_enabled:
        lidar = driver.getDevice("race_lidar")
        lidar.enable(sensor_ms)
        lidar.enablePointCloud()
    if record_video:
        import cv2
        camera = driver.getDevice("race_camera")
        camera.enable(video_ms)
        video = cv2.VideoWriter(str(output / "onboard.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                                1000. / video_ms, (camera.getWidth(), camera.getHeight()))
        if not video.isOpened():
            raise RuntimeError("Could not open onboard MP4 writer")
    wall_started = time.perf_counter()
    runtime_paths=[Path(__file__).resolve(),*[Path(m.__file__).resolve() for m in
                   (policy_module,control_module,hotlap_module,learned_module,geometry_module,rich_module,remote_module,aggressive_module,chase_module,line_module)]]
    runtime_sources={str(p.relative_to(ROOT)):hashlib.sha256(p.read_bytes()).hexdigest() for p in runtime_paths}
    driver.setGear(1)
    driver.setThrottle(0.)
    driver.setBrakeIntensity(1.)
    predictions = (output / "predictions.jsonl").open("x")
    actions = (output / "policy_actions.jsonl").open("x")
    trace = (output / "control_trace.jsonl").open("x")
    position, rotation, yaw, speed, forward = pose_state(car)
    progress = tracker.update(position, speed, dt)
    previous_position = position.copy()
    tick, frames, video_frames, decisions = 0, 0, 0, 0
    vision_changed, gap_guard_steps, lane_guard_steps, violations = 0, 0, 0, 0
    distance_driven, maximum_error, maximum_roll, maximum_speed = 0., 0., 0., 0.
    latencies = []
    finish_time, finish_position, failure = None, None, None
    result_written = False
    latest_frame = None
    point_stride = int(entry.get('pointcloud_stride_frames',config.get('pointcloud_stride_frames',0)))
    point_manifest = None
    if point_stride > 0:
        (output/'points').mkdir(exist_ok=True)
        point_manifest = (output/'points_manifest.jsonl').open('x')
    raw_action = {"target_speed_mps": 0., "target_offset_m": lane_offset, "model_used": learned is not None}
    observation = traffic.observation(progress.progress_m, speed, driver.getTime())
    target_speed, throttle, brake, steering = 0., 0., 1., 0.
    guard_feasible_speed, guard_profile = 0., {}

    def result(status, reason):
        return {"car_id": car_id, "label": entry.get("label", car_id), "policy": policy_name,
            "status": status, "reason": reason, "completed": finish_time is not None,
            "finish_time_s": finish_time, "lap_time_s": finish_time,
            "valid_lap": finish_time is not None and violations == 0 and start_progress <= 0.,
            "progress_m": progress.progress_m, "ordered_progress_m": progress.progress_m,
            "start_progress_m": start_progress, "start_lane_m": entry.get("start_lane_m", 0.),
            "finish_progress_m": target_progress, "reference_length_m": reference.length, "laps": laps,
            "start_time_s": start_time, "simulation_time_s": driver.getTime(),
            "elapsed_simulation_s": max(0., driver.getTime() - start_time),
            "wall_seconds": time.perf_counter() - wall_started,
            "track_violation_steps": violations, "track_error_limit_m": track_limit,
            "maximum_reference_error_m": maximum_error, "maximum_roll_rad": maximum_roll,
            "maximum_speed_mps": maximum_speed, "physical_distance_driven_m": distance_driven,
            "finish_position_m": finish_position, "position_resets_after_spawn": 0,
            "collision_samples": None, "collision_audit": "separate supervisor ground-truth audit",
            "physics_steps": tick, "physics_step_ms": step_ms, "sensor_period_ms": sensor_ms,
            "sensor_frames": frames, "video_frames": video_frames, "policy_decisions": decisions,
            "vision_changed_actions": vision_changed, "traffic_gap_guard_steps": gap_guard_steps,
            "lane_occupancy_guard_steps": lane_guard_steps,
            "inference_median_ms": float(np.median(latencies)) if latencies else None,
            "inference_p95_ms": float(np.percentile(latencies, 95)) if latencies else None,
            "neural_driver_used": learned is not None,
            "aggressive_hybrid_enabled": aggressive is not None, "flat_out_no_lift": flat_out,
            "racing_line_plan_sha256": racing_line_sha, "pass_first_both_sides": aggressive is not None and racing_line is not None,
            "learned_checkpoint_only_result": aggressive is None and not flat_out,
            "model_sha256": learned.checkpoint_sha256 if learned else None,
            "reference_sha256": hashlib.sha256(reference_path.read_bytes()).hexdigest(),
            "perception_enabled": perception_enabled, "vision_url": endpoint,
            "perception_control_source": ("independent measured LiDAR -> VoxelNeXt + raw geometry -> temporal hazards" if geometry is not None else "independent measured LiDAR -> detector boxes -> temporal tracks") if perception_enabled else "none; physics capacity pilot",
            "opponent_ground_truth_used_for_control": False,
            "control_version":config.get('control_version','legacy'),
            "runtime_source_root":str(ROOT),"runtime_source_sha256":runtime_sources,
            "runtime_detector_checkpoint_sha256":detector_sha,
            "runtime_detector_config_sha256":detector_config_sha,
            "runtime_policy_parameters":{'visible_distance_m':float(config.get('visible_distance_m',45.))},
            "post_finish_mode":config.get('post_finish_mode','cooldown'),
            "physical_limits": {"max_speed_mps": reference.max_speed, "lateral_accel_mps2": reference.lateral_accel,
                                "brake_decel_mps2": reference.brake_decel, "acceleration_mps2": reference.acceleration},
            "failure": failure}

    atomic_json(output / "result.json", result("running", "awaiting_common_start"))
    print(f"{car_id}: {policy_name}, signed grid {start_progress:.1f}m, LiDAR={perception_enabled}", flush=True)
    try:
        while True:
            now = driver.getTime()
            race_time = now - start_time
            reference_yaw = reference.sample(progress.progress_m)[1]
            reference_feasible_speed = reference.minimum_speed(progress.progress_m, 3. + speed)
            feasible_speed = (racing_line.minimum_speed(progress.progress_m, 3. + speed)
                if racing_line is not None and finish_time is None
                else reference_feasible_speed)
            if now >= start_time and not parked and not (result_written and finish_time is None):
                if tick % sensor_stride == 0:
                    if perception_enabled:
                        points = lidar_points(lidar, position, rotation, yaw)
                        if point_manifest is not None and (frames < 5 or frames % point_stride == 0):
                            point_file=f'points/{frames:06d}.npy'
                            np.save(output/point_file,points,allow_pickle=False)
                            point_manifest.write(json.dumps({'car_id':car_id,'frame_id':f'{frames:06d}',
                                'timestamp_ns':round(now*1e9),'points_file':point_file,
                                'coordinate_frame':'ego_x_forward_y_left_z_up'})+'\n')
                            point_manifest.flush()
                        before = time.perf_counter()
                        boxes = remote_detect(points, endpoint, timeout=float(config.get("inference_timeout_s", 60.)))
                        latency = 1000. * (time.perf_counter() - before)
                        latencies.append(latency)
                        traffic.update(boxes, position, yaw, progress.progress_m, now, ego_speed=speed)
                        if geometry is not None:
                            geometry_observation=geometry.update(points,position,yaw,progress.progress_m,now)
                        latest_frame = f"{frames:06d}"
                        predictions.write(json.dumps({"car_id": car_id, "frame_id": latest_frame,
                            "phase": "parking" if finish_time is not None else "racing",
                            "simulation_time_s": now, "timestamp_ns": round(now * 1e9), "inference_ms": latency,
                            "points": len(points), "boxes": boxes}, allow_nan=False) + "\n")
                        frames += 1
                    observation = traffic.observation(progress.progress_m, speed, now)
                    if geometry is not None:observation=combine_observations(observation,geometry_observation)
                    if finish_time is None or not park_after_finish:
                        state = policy_state(reference, progress.progress_m, progress.cross_track_m, yaw, speed, lane_offset, observation)
                        if isinstance(learned,RichDriver):state['hazard_tracks']=observation.get('hazard_tracks',[])
                        decision = policy.predict(state, observation, feasible_speed)
                        if aggressive is not None and finish_time is None:
                            decision = aggressive.apply(decision, state, observation, feasible_speed,
                                timestamp=now, dt=dt, brake_decel=reference.brake_decel,
                                visible_distance=float(config.get('visible_distance_m',70.)), progress_m=progress.progress_m)
                        raw_action = decision["raw_action"]
                        decisions += 1
                        vision_changed += int(decision["vision_changed_action"])
                        actions.write(json.dumps({"car_id": car_id, "simulation_time_s": now,
                            "timestamp_ns": round(now * 1e9), "vision_frame_id": latest_frame,
                            "inputs": state, **decision, "confirmed_tracks": observation["confirmed_tracks"],
                            "hazard_tracks": observation.get("hazard_tracks", observation["confirmed_tracks"]),
                            "geometry_hazards": observation.get("geometry_hazards",[]),
                            "phase": "racing" if finish_time is None else "cooldown",
                            "feasible_speed_cap_mps": feasible_speed, "racing_line_plan_sha256": racing_line_sha,
                            "selected_path_speed_cap_mps": raw_action.get("path_speed_cap_mps",feasible_speed),
                            "reference_speed_cap_mps":reference_feasible_speed}, allow_nan=False) + "\n")
                if finish_time is not None:
                    if park_after_finish:
                        traffic.max_lateral_m = 16.
                        parking_side = parking_side if parking_side is not None else (1. if progress.cross_track_m >= 0. else -1.)
                        plan = parking_plan(progress.progress_m, target_progress, entry.get('grid_index',0),
                                            progress.cross_track_m, speed, parked=parked, side=parking_side)
                        parked, parking_station = plan['parked'], plan['station_m']
                        raw_action = plan
                    else:
                        raw_action = {"target_speed_mps": min(20., feasible_speed),
                                      "target_offset_m": math.copysign(3.2, lane_offset if abs(lane_offset) > .5 else 1.)}
                guard_feasible_speed = feasible_speed
                guard_profile = {}
                if racing_line is not None and aggressive is not None and finish_time is None:
                    guard_feasible_speed, guard_profile = aggressive.guard_profile_cap(
                        progress.progress_m, 3. + speed, raw_action, progress.cross_track_m, feasible_speed)
                if predictive:
                    guarded = predictive_guard(raw_action, observation, body_speed=speed, current_lane=lane_offset,
                        actual_lateral=progress.cross_track_m, feasible_speed=guard_feasible_speed, dt=dt,
                        maximum_lane=15. if finish_time is not None and park_after_finish else float(entry.get("maximum_lane_m",3.2)),
                        brake_decel=reference.brake_decel, visible_distance=float(config.get('visible_distance_m',45.)),
                        enforce_visibility_stopping_budget=not (aggressive is not None and aggressive.relax_visibility),
                        racing_parameters=entry.get("racing_guard_parameters"))
                else:
                    guarded = shared_guard(raw_action, observation, body_speed=speed, current_lane=lane_offset,
                        feasible_speed=feasible_speed, dt=dt, brake_decel=reference.brake_decel)
                lane_offset, target_speed = guarded["target_offset_m"], guarded["target_speed_mps"]
                gap_guard_steps += int(any(x in guarded['guards'] for x in ('measured_vehicle_gap','predictive_stopping_gap')))
                lane_guard_steps += int(any(x in guarded['guards'] for x in ('alongside_lane_occupied','predicted_lane_conflict')))
                target_speed *= float(np.clip(1. - .06 * abs(progress.cross_track_m - lane_offset), .55, 1.))
                if abs(wrap_angle(reference_yaw - yaw)) > .4:
                    target_speed = min(target_speed, 12.)
                steering, _ = pursuit.control(reference, progress.progress_m, position, yaw, speed, dt, lateral_offset=lane_offset)
                throttle, brake = longitudinal_control(target_speed, max(0., forward),
                    max(0., float(driver.getCurrentSpeed()) / 3.6), mass=880., wheel_radius=wheel_radius,
                    brake_coefficient=700., max_accel=reference.acceleration, max_decel=reference.brake_decel)
                if guarded.get('minimum_decel_mps2',0.) > 0.:
                    throttle = 0.
                    brake = max(brake, min(1., guarded['minimum_decel_mps2']*880.*wheel_radius/(4.*700.)))
            else:
                throttle, brake = 0., 1.
            commanded, actuator_override = chase_module.flat_out_actuators(flat_out,
                now >= start_time and finish_time is None and not result_written and not parked, throttle, brake)
            throttle, brake = commanded['throttle'], commanded['brake']
            if parked:
                throttle, brake = 0., 1.
                if lidar is not None:
                    lidar.disable();lidar = None
                if video is not None:
                    camera.disable();video.release();video = None
            if result_written and finish_time is None:
                throttle, brake = 0., 1.
            driver.setSteeringAngle(-steering)
            driver.setThrottle(throttle)
            driver.setBrakeIntensity(brake)
            old_progress, old_time = progress.progress_m, driver.getTime()
            terminal_sample = False
            if driver.step() == -1:
                break
            tick += 1
            position, rotation, yaw, speed, forward = pose_state(car)
            progress = tracker.update(position, speed, dt)
            distance_driven += float(np.linalg.norm(position[:2] - previous_position[:2]))
            previous_position = position.copy()
            roll = math.atan2(rotation[2, 1], rotation[2, 2])
            if driver.getTime() >= start_time and not result_written:
                maximum_error = max(maximum_error, abs(progress.cross_track_m))
                maximum_roll, maximum_speed = max(maximum_roll, abs(roll)), max(maximum_speed, speed)
                violations += int(abs(progress.cross_track_m) > track_limit)
                if progress.distance_m > 25. or abs(roll) > 1. or position[2] > 3.:
                    atomic_json(output / "result.json", result("failed", "vehicle_left_reference_corridor_or_rolled"))
                    result_written = True
                    terminal_sample = True
                elif progress.progress_m >= target_progress and driver.getTime() - start_time > 10.:
                    fraction = float(np.clip((target_progress - old_progress) / max(progress.progress_m - old_progress, 1e-9), 0., 1.))
                    finish_time = old_time + fraction * dt - start_time
                    finish_position = position.tolist()
                    atomic_json(output / "result.json", result("completed" if violations == 0 else "completed_invalid",
                        "crossed_common_finish_after_full_ordered_path"))
                    result_written = True
                    terminal_sample = True
                    print(f"{car_id} FINISH {finish_time:.3f}s, violations={violations}", flush=True)
                elif driver.getTime() >= maximum_seconds:
                    atomic_json(output / "result.json", result("timeout", "race_time_limit"))
                    result_written = True
                    terminal_sample = True
            if tick % telemetry_stride == 0 or terminal_sample:
                trace.write(json.dumps({"car_id": car_id, "time_s": max(0., driver.getTime() - start_time),
                    "simulation_time_s": driver.getTime(), "timestamp_ns": round(driver.getTime() * 1e9),
                    "position_m": position.tolist(), "yaw_rad": yaw, "speed_mps": speed,
                    "forward_speed_mps": forward, "progress_m": progress.progress_m,
                    "cross_track_error_m": progress.cross_track_m, "reference_distance_m": progress.distance_m,
                    "target_speed_mps": target_speed, "target_lane_offset_m": lane_offset,
                    "steering_left_rad": steering, "throttle": throttle, "brake": brake, "roll_rad": roll,
                    "confirmed_vision_tracks": len(observation["confirmed_tracks"]),
                    "vision_frame_id": latest_frame, "completed": finish_time is not None,
                    "terminal_sample":terminal_sample, "actuator_override": actuator_override,
                    "path_mode":raw_action.get("path_mode"), "selected_path_profile_lane_m":raw_action.get("path_profile_lane_m"),
                    "nominal_racing_line_offset_m":raw_action.get("nominal_racing_line_offset_m"),
                    "guard_feasible_speed_cap_mps":guard_feasible_speed if now>=start_time else feasible_speed,
                    "racing_line_plan_sha256":racing_line_sha, "guard_path_profile":guard_profile,
                    "reference_speed_cap_mps":reference_feasible_speed, "nominal_speed_cap_mps":feasible_speed}, allow_nan=False) + "\n")
                trace.flush()
                predictions.flush()
                actions.flush()
            if video is not None and tick % video_stride == 0:
                import cv2
                frame = video_frame(camera, car_id, policy_name, finish_time if finish_time is not None else race_time,
                    speed, progress.progress_m, target_progress, lane_offset, observation["confirmed_tracks"], finish_time is not None)
                video.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))
                if video_frames % 100 == 0:
                    from PIL import Image
                    Image.fromarray(frame).save(output / f"onboard_{video_frames:06d}.jpg", quality=88)
                video_frames += 1
            if tick % max(1, int(10. / dt)) == 0:
                print(f"{car_id}: {driver.getTime()-start_time:.1f}s {progress.progress_m:.0f}/{target_progress:.0f}m {speed*3.6:.1f}km/h tracks={len(observation['confirmed_tracks'])}", flush=True)
    except Exception as error:
        failure = f"{type(error).__name__}: {error}"
        print(f"{car_id}: {failure}", flush=True)
        if not result_written:
            atomic_json(output / "result.json", result("failed", "controller_exception"))
            result_written = True
        # Leave this physical car braking until the supervisor closes the heat.
        driver.setThrottle(0.)
        driver.setBrakeIntensity(1.)
        while driver.step() != -1:
            pass
    finally:
        if not result_written:
            atomic_json(output / "result.json", result("incomplete", "simulation_closed_before_finish"))
        atomic_json(output / "controller_summary.json", {**result("completed" if finish_time is not None else "stopped", "controller_closed"),
                    'parked':parked,'parking_station_m':parking_station,'control_version':config.get('control_version','legacy')})
        trace.close()
        predictions.close()
        actions.close()
        if point_manifest is not None:point_manifest.close()
        if video is not None:
            video.release()


if __name__ == "__main__":
    main()
