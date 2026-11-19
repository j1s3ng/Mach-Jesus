"""Generate a self-contained Webots circuit using only built-in geometry.

Motion is scripted kinematics for repeatable perception tests, not a validated
vehicle dynamics model. Webots itself produces occluded depth and RGB images.
"""
from __future__ import annotations

import math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def shape_box(size, color, translation=(0, 0, 0)):
    return f'''Pose {{ translation {' '.join(map(str, translation))} children [ Shape {{
      appearance PBRAppearance {{ baseColor {color} roughness 0.65 metalness 0 }}
      geometry Box {{ size {' '.join(map(str, size))} }} }} ] }}'''


def car_body(color):
    parts = [shape_box((5.0, 1.6, .45), color, (0, 0, .47)),
             shape_box((2.1, .8, .45), "0.06 0.08 0.12", (-.4, 0, .85)),
             shape_box((.35, 2.0, .12), color, (2.3, 0, .25)),
             shape_box((.45, 2.0, .12), color, (-2.25, 0, .95))]
    for x in [-1.65, 1.6]:
        for y in [-.95, .95]:
            parts.append(f'''Pose {{ translation {x} {y} .32 rotation 1 0 0 1.5707963 children [
              Shape {{ appearance PBRAppearance {{ baseColor .035 .035 .04 roughness 1 metalness 0 }}
              geometry Cylinder {{ radius .32 height .30 subdivision 12 }} }} ] }}''')
    return "\n".join(parts)


def ring(inner, outer, color, z=.006, segments=144):
    vertices = []
    for i in range(segments):
        a = i * 2 * math.pi / segments
        for r in [inner, outer]:
            vertices.append(f"{r * math.cos(a):.5f} {r * math.sin(a):.5f} {z}")
    faces = [f"{2*i} {2*i+1} {2*((i+1)%segments)+1} {2*((i+1)%segments)} -1" for i in range(segments)]
    return f'''Shape {{ appearance PBRAppearance {{ baseColor {color} roughness 1 metalness 0 }}
      geometry IndexedFaceSet {{ coord Coordinate {{ point [ {', '.join(vertices)} ] }}
      coordIndex [ {', '.join(faces)} ] }} }}'''


def generate():
    world = ['''#VRML_SIM R2025b utf8
WorldInfo { title "Blue Galaxy | LiDAR opponent lab" basicTimeStep 50 coordinateSystem "ENU" }
Viewpoint { orientation .29 .54 .79 2.45 position 113 -105 105 follow "ego" }
Background { skyColor [ .55 .68 .82 ] }
DirectionalLight { direction -.5 -.3 -1 intensity 2 ambientIntensity .55 castShadows FALSE }
Solid { name "ground" children [ Shape { appearance PBRAppearance { baseColor .13 .20 .15 roughness 1 metalness 0 }
  geometry Plane { size 300 300 } } ] boundingObject Plane { size 300 300 } }
''', ring(57, 73, ".15 .16 .18"), ring(56.8, 57.1, ".95 .95 .91", .01),
             ring(72.9, 73.2, ".95 .95 .91", .01), ring(64.95, 65.05, ".62 .64 .65", .011)]
    for i in range(48):
        a = i * math.pi / 24
        for r in [56, 74]:
            world.append(f'''Solid {{ name "marker_{i}_{r}" translation {r*math.cos(a):.4f} {r*math.sin(a):.4f} .4
              children [ Shape {{ appearance PBRAppearance {{ baseColor .8 .22 .12 roughness 1 metalness 0 }}
              geometry Box {{ size .5 .5 .8 }} }} ] boundingObject Box {{ size .5 .5 .8 }} }}''')
    world.append(f'''DEF EGO Robot {{ translation 65 0 0 rotation 0 0 1 1.5707963
      name "ego" controller "race_recorder" supervisor TRUE window "<none>"
      children [ {car_body('.12 .42 .95')}
        Lidar {{ translation 0 0 1.55 name "race_lidar" horizontalResolution 1024
          fieldOfView 6.2831853 verticalFieldOfView .44 numberOfLayers 32
          near .1 minRange 1 maxRange 100 noise .0001 }}
        Camera {{ translation 1.6 0 1.2 name "race_camera" width 640 height 360
          fieldOfView 1.4 near .1 far 130 }}
      ] }}''')
    for i, (phase, color) in enumerate([(.28, ".95 .28 .13"), (.66, ".93 .76 .16"), (1.12, ".62 .3 .94")]):
        world.append(f'''DEF OPPONENT_{i} Solid {{ translation {65*math.cos(phase):.5f} {65*math.sin(phase):.5f} 0
          rotation 0 0 1 {phase+math.pi/2:.6f} name "opponent_{i}"
          children [ {car_body(color)} ] }}''')
    out = ROOT / "webots/worlds/race_loop.wbt"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(world) + "\n")
    return out


if __name__ == "__main__":
    print(generate())
