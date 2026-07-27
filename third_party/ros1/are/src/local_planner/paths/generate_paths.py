#!/usr/bin/env python3
"""
Python equivalent of path_generator.m
Generates startPaths.ply, paths.ply, and pathList.ply for localPlanner.
"""

import numpy as np
from scipy.interpolate import CubicSpline
import os

dis = 1.0
angle = 27.0
deltaAngle = angle / 3.0  # 9.0
scale = 0.65


def matlab_range(start, stop, step):
    n = round((stop - start) / step)
    return [start + i * step for i in range(n + 1)]


pathStartAll = []
pathAll = []
pathList = []
pathID = 0
groupID = 0

pathStartR = np.linspace(0, dis, 101)  # MATLAB: 0:0.01:dis

for shift1 in matlab_range(-angle, angle, deltaAngle):
    # 2 data points → linear interpolation: shift from 0 to shift1
    pathStartShift = pathStartR * shift1 / dis

    pathStartX = pathStartR * np.cos(pathStartShift * np.pi / 180)
    pathStartY = pathStartR * np.sin(pathStartShift * np.pi / 180)
    pathStartZ = np.zeros_like(pathStartX)

    for i in range(len(pathStartX)):
        pathStartAll.append((pathStartX[i], pathStartY[i], pathStartZ[i], groupID))

    for shift2 in matlab_range(-angle * scale + shift1, angle * scale + shift1, deltaAngle * scale):
        for shift3 in matlab_range(-angle * scale**2 + shift2, angle * scale**2 + shift2, deltaAngle * scale**2):
            waypts_x = np.concatenate([pathStartR, [2 * dis, 3 * dis - 0.001, 3 * dis]])
            waypts_y = np.concatenate([pathStartShift, [shift2, shift3, shift3]])

            pathR = np.linspace(0, 3 * dis, 301)  # MATLAB: 0:0.01:3*dis

            cs = CubicSpline(waypts_x, waypts_y, bc_type='not-a-knot')
            pathShift = cs(pathR)

            pathX = pathR * np.cos(pathShift * np.pi / 180)
            pathY = pathR * np.sin(pathShift * np.pi / 180)
            pathZ = np.zeros_like(pathX)

            for i in range(len(pathX)):
                pathAll.append((pathX[i], pathY[i], pathZ[i], pathID, groupID))

            pathList.append((pathX[-1], pathY[-1], pathZ[-1], pathID, groupID))
            pathID += 1

    groupID += 1

print(f"Groups: {groupID}, Paths: {pathID}")
print(f"Start path points: {len(pathStartAll)}")
print(f"Path points: {len(pathAll)}")
print(f"Path list entries: {len(pathList)}")

script_dir = os.path.dirname(os.path.abspath(__file__))

with open(os.path.join(script_dir, 'startPaths.ply'), 'w') as f:
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write(f'element vertex {len(pathStartAll)}\n')
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    f.write('property int group_id\n')
    f.write('end_header\n')
    for x, y, z, gid in pathStartAll:
        f.write(f'{x:.6f} {y:.6f} {z:.6f} {gid}\n')

with open(os.path.join(script_dir, 'paths.ply'), 'w') as f:
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write(f'element vertex {len(pathAll)}\n')
    f.write('property float x\n')
    f.write('property float y\n')
    f.write('property float z\n')
    f.write('property int path_id\n')
    f.write('property int group_id\n')
    f.write('end_header\n')
    for x, y, z, pid, gid in pathAll:
        f.write(f'{x:.6f} {y:.6f} {z:.6f} {pid} {gid}\n')

with open(os.path.join(script_dir, 'pathList.ply'), 'w') as f:
    f.write('ply\n')
    f.write('format ascii 1.0\n')
    f.write(f'element vertex {len(pathList)}\n')
    f.write('property float end_x\n')
    f.write('property float end_y\n')
    f.write('property float end_z\n')
    f.write('property int path_id\n')
    f.write('property int group_id\n')
    f.write('end_header\n')
    for x, y, z, pid, gid in pathList:
        f.write(f'{x:.6f} {y:.6f} {z:.6f} {pid} {gid}\n')

print("All path files generated successfully!")
