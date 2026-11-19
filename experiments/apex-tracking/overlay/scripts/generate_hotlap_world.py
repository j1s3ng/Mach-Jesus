"""Build an offline Webots circuit and a generic physical race car.

Reference XY coordinates are unchanged. Boundaries are a visual corridor, not
the surveyed F1 circuit. The car is driven through Webots wheel physics by the
separate controller; this generator never scripts vehicle motion.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from race_lidar.hotlap_geometry import load_geometry
from race_lidar.hotlap_driver import RacingReference


def box(size, color, translation=(0, 0, 0)):
    return f'''Pose {{ translation {' '.join(map(str, translation))} children [ Shape {{
      appearance PBRAppearance {{ baseColor {color} roughness .7 metalness 0 }}
      geometry Box {{ size {' '.join(map(str, size))} }} }} ] }}'''


def ribbon(left, right, color, z=.01, mask=None):
    vertices = []
    for a, b in zip(left, right):
        vertices.extend([f"{a[0]:.4f} {a[1]:.4f} {z}", f"{b[0]:.4f} {b[1]:.4f} {z}"])
    n = len(left)
    # A wide approximate corridor can fold its inner edge at a tight corner.
    # Emit individually upward-facing triangles rather than relying on VRML's
    # quad triangulation, keeping these small overlap regions visible.
    xy = np.stack([left,right],axis=1).reshape(-1,2)
    faces = []
    for i in range(n):
        if mask is not None and not mask[i]:
            continue
        a,b,c,d = 2*i,2*i+1,2*((i+1)%n)+1,2*((i+1)%n)
        for triangle in [(a,b,c),(a,c,d)]:
            p,q,r = xy[list(triangle)]
            area = (q[0]-p[0])*(r[1]-p[1])-(q[1]-p[1])*(r[0]-p[0])
            if area < 0:
                triangle = (triangle[0],triangle[2],triangle[1])
            faces.append(' '.join(map(str,triangle))+' -1')
    return f'''Shape {{ appearance PBRAppearance {{ baseColor {color} roughness 1 metalness 0 }}
      geometry IndexedFaceSet {{ coord Coordinate {{ point [ {', '.join(vertices)} ] }}
      coordIndex [ {', '.join(faces)} ] }} }}'''


def wheel(name, left=False):
    return f'''VehicleWheel {{ name "{name}" tireRadius .35 thickness .32 rimRadius .22
      subdivision 24 edgeSubdivision 2 wheelSide {'TRUE' if left else 'FALSE'}
      contactMaterial "race_tire" physics Physics {{ density -1 mass 20 }} }}'''


def reference_spawn(reference, progress_m, lane_offset_m=0.):
    point, yaw, _, _ = reference.sample(progress_m)
    normal = np.array([-math.sin(yaw), math.cos(yaw)])
    return point + lane_offset_m * normal, yaw


def backmarker_specs(count):
    if not isinstance(count, int) or not 0 <= count <= 12:
        raise ValueError("backmarkers must be an integer between 0 and 12")
    initial = [70.,260.,650.]
    return [{"name":f"backmarker_{i}","def":f"BACKMARKER_{i}",
             "start_progress_m":initial[i] if i < 3 else 650.+450.*(i-2),
             "lane_offset_m":-2.5,"speed_scale":.55,
             "color":[".95 .20 .10",".96 .78 .13",".62 .28 .88"][i%3]}
            for i in range(count)]


def physical_car(name, def_name, point, yaw, color, controller, controller_args=(), sensors=False):
    body = [box((4.8, 1.6, .45), color, (1.6, 0, .12)),
            box((2.1, .8, .45), ".035 .045 .065", (1.2, 0, .5)),
            box((.35, 2.0, .12), color, (3.9, 0, -.1)),
            box((.45, 2.0, .12), color, (-.65, 0, .6))]
    sensor_nodes = '''DEF HOTLAP_LIDAR Lidar { translation 1.6 0 1.2 name "race_lidar" horizontalResolution 1024
          fieldOfView 6.2831853 verticalFieldOfView .44 numberOfLayers 32 near .1 minRange 1 maxRange 100 noise .0001 }
        DEF HOTLAP_CAMERA Camera { translation 3.2 0 .85 name "race_camera" width 640 height 360
          fieldOfView 1.4 near .1 far 180 }''' if sensors else ''
    args = ' '.join(json.dumps(str(value)) for value in controller_args)
    return f'''DEF {def_name} Car {{ translation {point[0]:.5f} {point[1]:.5f} .35
      rotation 0 0 1 {yaw:.8f} name "{name}" controller "{controller}" controllerArgs [ {args} ]
      supervisor TRUE window "<none>" model "generic physical race car"
      type "propulsion" engineType "electric" engineSound "" gearRatio [ -1 1 ]
      wheelbase 3.2 trackFront 1.7 trackRear 1.7 wheelsDampingConstant .2
      engineMaxTorque 1500 engineMaxPower 400000 maxVelocity 100 brakeCoefficient 700
      minSteeringAngle -.6 maxSteeringAngle .6
      physics Physics {{ density -1 mass 800 centerOfMass [ 1.6 0 -.05 ] inertiaMatrix [ 220 1700 1800 0 0 0 ] }}
      boundingObject Pose {{ translation 1.6 0 .12 children [ Box {{ size 4.8 1.6 .45 }} ] }}
      wheelFrontRight {wheel('front right wheel')}
      wheelFrontLeft {wheel('front left wheel', True)}
      wheelRearRight {wheel('rear right wheel')}
      wheelRearLeft {wheel('rear left wheel', True)}
      extensionSlot [ {' '.join(body)} {sensor_nodes} ] }}'''


def build(reference, output, spacing=3., half_width=10., backmarkers=0, *,
          decorative_trackside_posts_enabled=True):
    if type(decorative_trackside_posts_enabled) is not bool:
        raise ValueError("decorative_trackside_posts_enabled must be a boolean")
    reference, output = Path(reference).resolve(), Path(output).resolve()
    geometry = load_geometry(reference, spacing, half_width)
    center, normal, widths = geometry["center_xy"], geometry["normal_xy"], geometry["half_width_m"]
    left, right = center + normal * widths[:, None], center - normal * widths[:, None]
    lo, hi = np.minimum(left.min(0), right.min(0)), np.maximum(left.max(0), right.max(0))
    midpoint, extent = (lo + hi) / 2, hi - lo
    start = geometry["reference_xy"][0]
    yaw = math.atan2(*(geometry["reference_xy"][1] - start)[::-1])
    # Native physics Car assets are bundled locally; no network PROTO fetches.
    car_url = Path(__import__('os').path.relpath(ROOT / "webots/protos/vehicle/Car.proto", output.parent)).as_posix()
    wheel_url = Path(__import__('os').path.relpath(ROOT / "webots/protos/vehicle/VehicleWheel.proto", output.parent)).as_posix()
    world = [f'''#VRML_SIM R2025b utf8
# Approximate road corridor derived from public reference XY; not surveyed F1 boundaries.
EXTERNPROTO "{car_url}"
EXTERNPROTO "{wheel_url}"
WorldInfo {{ title "Blue Galaxy | Full hot lap" basicTimeStep 20 coordinateSystem "ENU"
  contactProperties [ ContactProperties {{ material1 "race_tire" material2 "race_track"
    coulombFriction [ 1.5 ] bounce 0 softCFM .00001 }} ] }}
Viewpoint {{ orientation .29 .54 .79 2.45 position {start[0]+25:.3f} {start[1]-25:.3f} 18
  follow "ego" followSmoothness .1 }}
Background {{ skyColor [ .55 .69 .84 ] }}
DirectionalLight {{ direction -.4 -.3 -1 intensity 1.6 ambientIntensity .6 castShadows FALSE }}
Solid {{ name "ground" translation {midpoint[0]:.3f} {midpoint[1]:.3f} 0 contactMaterial "race_track"
  children [ Shape {{ appearance PBRAppearance {{ baseColor .48 .43 .31 roughness 1 metalness 0 }}
    geometry Plane {{ size {extent[0]+300:.3f} {extent[1]+300:.3f} }} }} ]
  boundingObject Plane {{ size {extent[0]+300:.3f} {extent[1]+300:.3f} }} }}
''', ribbon(left + normal * 5, right - normal * 5, ".10 .32 .42", .004),
             ribbon(left, right, ".14 .15 .17", .008),
             ribbon(left, left - normal * .15, ".95 .95 .90", .012),
             ribbon(right + normal * .15, right, ".95 .95 .90", .012)]
    # Alternating curbs use two meshes per side rather than thousands of nodes.
    alternate = (np.floor(geometry["s_m"] / 5).astype(int) % 2) == 0
    for mask, color in [(alternate, ".88 .12 .10"), (~alternate, ".96 .95 .90")]:
        world += [ribbon(left + normal * .9, left + normal * .18, color, .025, mask),
                  ribbon(right - normal * .18, right - normal * .9, color, .025, mask)]
    ref_tangent = np.roll(geometry["reference_xy"], -1, 0) - np.roll(geometry["reference_xy"], 1, 0)
    ref_tangent /= np.linalg.norm(ref_tangent, axis=1, keepdims=True)
    ref_normal = np.column_stack([-ref_tangent[:, 1], ref_tangent[:, 0]])
    world.append(ribbon(geometry["reference_xy"] + .09 * ref_normal,
                        geometry["reference_xy"] - .09 * ref_normal, ".25 .78 .33", .018))
    # Start/finish checkerboard, centered on the reference start.
    checker = []
    for row in range(2):
        for col in range(20):
            checker.append(box((.55, .75, .012), ".96 .96 .96" if (row+col)%2 else ".03 .03 .03",
                               ((row-.5)*.55, (col-9.5)*.75, .026)))
    world.append(f'''Pose {{ translation {start[0]:.4f} {start[1]:.4f} 0 rotation 0 0 1 {yaw:.7f}
      children [ {' '.join(checker)} ] }}''')
    # Discrete trackside marker posts provide range/flow cues without unmodeled obstacles.
    if decorative_trackside_posts_enabled:
        stride = max(1, round(50. / (geometry["length_m"] / len(center))))
        for i in range(0, len(center), stride):
            for side in [-1, 1]:
                xy = center[i] + normal[i] * side * (widths[i] + 2)
                world.append(f'''Pose {{ translation {xy[0]:.3f} {xy[1]:.3f} .55
              children [ Shape {{ appearance PBRAppearance {{ baseColor .92 .92 .85 roughness 1 metalness 0 }}
                geometry Box {{ size .25 .25 1.1 }} }} ] }}''')
    world.append(physical_car('ego','EGO',start,yaw,'.10 .34 .90','hotlap_recorder',sensors=True))
    traffic = backmarker_specs(backmarkers)
    if traffic:
        driving_reference = RacingReference.from_npz(reference)
        for spec in traffic:
            spawn, heading = reference_spawn(driving_reference,spec['start_progress_m'],spec['lane_offset_m'])
            spec['spawn_xy_m'] = spawn.tolist()
            spec['spawn_yaw_rad'] = heading
            world.append(physical_car(spec['name'],spec['def'],spawn,heading,spec['color'],'backmarker_driver',
                                      [spec['start_progress_m'],spec['lane_offset_m'],spec['speed_scale']]))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(world) + "\n")
    serial = {key: value for key, value in geometry.items() if isinstance(value, np.ndarray)}
    np.savez(output.with_suffix(".geometry.npz"), **serial)
    metadata = {"reference": str(reference), "reference_xy_unchanged": True,
                "road_geometry": "approximate corridor, not surveyed F1 track boundaries",
                "length_m": geometry["length_m"], "source_endpoint_gap_m": geometry["source_endpoint_gap_m"],
                "road_half_width_m": half_width, "minimum_road_half_width_m": float(widths.min()),
                "road_center_periodic_smoothing_sigma_m": 4.,
                "maximum_reference_to_road_center_offset_m": float(np.linalg.norm(geometry["reference_xy"]-center,axis=1).max()),
                "samples": len(center), "vehicle": "generic Webots Car with physical tires",
                "body_mass_kg": 800, "wheel_mass_kg_each": 20, "wheel_radius_m": .35,
                "brake_coefficient_nm_per_wheel": 700.,
                "wheelbase_m": 3.2, "tire_road_friction": 1.5, "physics_step_ms": 20,
                "lidar_local_translation_m": [1.6, 0, 1.2], "camera_local_translation_m": [3.2, 0, .85],
                "spawn_axle_z_m": .35, "vehicle_position_convention": "rear axle wheel-center height"}
    metadata['backmarkers'] = traffic
    output.with_suffix(".geometry.json").write_text(json.dumps(metadata, indent=2) + "\n")
    return output


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, default=ROOT / ".local/f1-reference/reference.npz")
    parser.add_argument("--output", type=Path, default=ROOT / "webots/worlds/hotlap.wbt")
    parser.add_argument("--spacing", type=float, default=3.)
    parser.add_argument("--half-width", type=float, default=10.)
    parser.add_argument("--backmarkers", type=int, default=0, help="Add 0–12 physical slower cars; default 0")
    parser.add_argument("--no-decorative-trackside-posts", dest="decorative_trackside_posts_enabled",
                        action="store_false", help="Omit non-collidable decorative marker posts; default keeps them")
    args = parser.parse_args()
    print(build(args.reference, args.output, args.spacing, args.half_width,args.backmarkers,
                decorative_trackside_posts_enabled=args.decorative_trackside_posts_enabled))


if __name__ == "__main__":
    main()
