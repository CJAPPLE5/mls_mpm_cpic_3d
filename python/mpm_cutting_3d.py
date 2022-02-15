from importlib.resources import path
from turtle import color
from matplotlib.pyplot import grid
from pkg_resources import parse_version
import taichi as ti
import numpy as np
import os
import sys
import math

export_file = ""
obj_file = "./python/spatula2.obj"


def read_obj(path):
    f = open(path)

    v_num = 0
    f_num = 0

    while f.readable():
        s = f.readline()
        if "Vertices:" in s:
            v_num = int(s.split()[-1])
        if "Faces:" in s:
            f_num = int(s.split()[-1])
        if "v" in s:
            break

    x_r = ti.Vector.field(3, dtype=float, shape=(f_num,3))
    x_rp = ti.Vector.field(3, dtype=float, shape=f_num)

    points = []
    faces = []
    maxx = -1
    minn = 9999

    for i in range(v_num):
        ps = [float(j) for j in  s.split()[1:]]
        maxx = max(maxx, max(ps))
        minn = min(minn, min(ps))
        points.append(ps)
        s = f.readline()

    for i in range(v_num):
        points[i] = [j/(maxx-minn) for j in points[i]]

    while "f" not in s:
        s = f.readline()

    for i in range(f_num):
        fs = [int(j) for j in  s.split()[1:]]
        faces.append(fs)
        s = f.readline()

    for i in range(f_num):
        ps = [points[j-1] for j in faces[i]]
        for j in range(3):
            x_r[i,j] = ps[j]
        x_rp[i] = (x_r[i,0] + x_r[i,1] + x_r[i,2]) / 3
    f.close()
    return x_rp, x_r

ti.init(arch=ti.gpu,advanced_optimization=False)
w = 1.0
dim = 3
quality = 1
bound = 3
n_grid = 64
n_particles = n_grid**dim // 2**(dim - 1)
dx, inv_dx = w / n_grid, float(n_grid) / w
dt = 1e-4 / quality
p_vol, p_rho = (dx*0.5)**3, 1
p_mass = p_vol * p_rho
E, nu = 0.1e4, 0.2
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu))
x = ti.Vector.field(dim, dtype=float, shape=n_particles)
v = ti.Vector.field(dim, dtype=float, shape=n_particles)
C = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles)
F = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles)
material = ti.field(dtype=int, shape=n_particles)
Jp = ti.field(dtype=float, shape=n_particles)
grid_v = ti.Vector.field(dim, dtype=float, shape=(n_grid, )*dim)
grid_m = ti.field(dtype=float, shape=(n_grid,) * dim)
energy = ti.field(dtype=float, shape=())
dirr = ti.Vector([1.0,1.0,1.0])


n_rseg = 100
r_circle = 0.1*w
r_seg = 0.01*w

x_rp, x_r = read_obj(obj_file)
cen_seg = [0.8*w, 0.8*w, 0.8*w]

grid_d = ti.field(dtype=float, shape=(n_grid,) * dim)
grid_A = ti.field(dtype=int, shape=(n_grid, )* dim)
grid_T = ti.field(dtype=int, shape=(n_grid, )*dim)

p_d = ti.field(dtype=float, shape=n_particles)
p_A = ti.field(dtype=int, shape=n_particles)
p_T = ti.field(dtype=int, shape=n_particles)
p_n = ti.Vector.field(dim, dtype=float, shape=n_particles)
particles_radius = 0.001
particles_radius1 = 0.01
gravity = 20
r_v = 100

colors = ti.Vector([0.0,1.0,0.0,1.0])
colors_spon = ti.Vector([1.0,0.0,0.0,1.0])
@ti.func
def getnormal(i):
    normal = (x_r[i,1]-x_r[i,0]).cross(x_r[i,2]-x_r[i,1])
    return normal


@ti.func
def isinplane(points, i):
    flag = -1
    distance = -1
    T = -1
    normal = (x_r[i,1]-x_r[i,0]).cross(x_r[i,2]-x_r[i,1])
    # print(i,x_r[i,0],x_r[i,1],x_r[i,2])
    # print(x_r[i,1]-x_r[i,0], x_r[i,2]-x_r[i,1], normal.normalized())
    d = -1*normal.dot(x_r[i,0])
    xp = ((normal[1]**2 + normal[2]**2)*points[0] - normal[0]*(normal[1]*points[1]+normal[2]*points[2]+d)) / (normal[0]**2+normal[1]**2+normal[2]**2)
    yp = ((normal[0]**2 + normal[2]**2)*points[1] - normal[1]*(normal[0]*points[0]+normal[2]*points[2]+d)) / (normal[0]**2+normal[1]**2+normal[2]**2)
    zp = ((normal[0]**2 + normal[1]**2)*points[2] - normal[2]*(normal[0]*points[0]+normal[1]*points[1]+d)) / (normal[0]**2+normal[1]**2+normal[2]**2)
    p_pro = ti.Vector([xp, yp, zp])
    b1 = (p_pro-x_r[i,0]).cross(x_r[i,1]-x_r[i,0]).dot((x_r[i,2]-x_r[i,0]).cross(x_r[i,1]-x_r[i,0]))
    b2 = (p_pro-x_r[i,1]).cross(x_r[i,2]-x_r[i,1]).dot((x_r[i,0]-x_r[i,1]).cross(x_r[i,2]-x_r[i,1]))
    b3 = (p_pro-x_r[i,2]).cross(x_r[i,0]-x_r[i,2]).dot((x_r[i,1]-x_r[i,2]).cross(x_r[i,0]-x_r[i,2]))
    if (b1>=0 and b2>=0 and b3>=0) or(b1<=0 and b2<=0 and b3<+0):
        flag = 1
    distance = (points.dot(normal)+d) / (normal[0]**2+normal[1]**2+normal[2]**2)
    condi = (points-x_rp[i]).dot(normal) > 0
    if condi:
        T = 1
    return flag, distance, T, normal

@ti.kernel
def initialize():
    for i in range(n_particles):
        x[i] = ti.Vector([ti.random()* 0.4 * w + 0.3 * w for i in range(dim)])
        v[i] = ti.Matrix([0, 0, 0])
        F[i] = ti.Matrix.identity(float, dim)
        Jp[i] = 1


@ti.kernel
def substep():
    x_dir = dirr
    # print(x_dir)
    ti.static_print("CDF")
    for p in x_rp:
        x_rp[p] += x_dir * dt * r_v
        x_r[p,0] += x_dir * dt * r_v
        x_r[p,1] += x_dir * dt * r_v
        x_r[p,2] += x_dir * dt * r_v
    
    #CDF
    for i in ti.grouped(grid_A):
        grid_A[i] = 0
        grid_T[i] = 0
        grid_d[i] = 0.0

    for p in x_rp:
        # ba = x_r[p+1] - x_r[p]
        base = (x_rp[p] * inv_dx - 0.5).cast(int)
        p_normal = (x_r[p,1] - x_r[p,0]).outer_product(x_r[p,2]-x_r[p,1])
        for i, j, k in ti.static(ti.ndrange(3,3,3)):
            offset = ti.Vector([i, j, k])
            point = (offset + base).cast(float) * dx 
            condi, distance, T, _ = isinplane(point, p)

            if condi:
                grid_d[base+offset] = distance
                grid_A[base+offset] = 1
                grid_T = T

    for p in x:
        p_A[p] = 0
        p_T[p] = 0.0
        p_d[p] = 0.0

        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        Tpr = 0.0
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i,j,k])
            if grid_A[base+offset] == 1:
                p_A[p] = 1
            
            weight = w[i][0] * w[j][1] * w[k][2]
            Tpr += weight * grid_d[base+offset] * grid_T[base+offset]

        if p_A[p] == 1:
            if Tpr > 0:
                p_T[p] = 1
            else:
                p_T[p] = -1
            p_d[p] = abs(Tpr)


    for I in ti.grouped(grid_m):
        grid_v[I] = ti.zero(grid_v[I])
        grid_m[I] = 0
    ti.block_dim(n_grid)

    ti.static_print("P2G")
    # P2G
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        F[p] = (ti.Matrix.identity(float, dim) + dt*C[p]) @ F[p]
        h = 0.7
        mu, la = mu_0 * h, lambda_0 * h
        U, sig, V = ti.svd(F[p])
        J = 1.0
        for d in ti.static(range(dim)):
            new_sig = sig[d, d]
            J *= new_sig
        stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[p].transpose() + ti.Matrix.identity(float, 3) * la * J * (J - 1)
        stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
        affine = stress + p_mass * C[p]
        for i, j, k in ti.static(ti.ndrange(dim, dim, dim)):
            offset = ti.Vector([i,j,k])
            if p_T[p] * grid_T[base+offset] == -1:
                pass
            else:
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[i][0] * w[j][1] * w[k][2]
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass
      

    for I in ti.grouped(grid_m):
        if grid_m[I] > 0: # No need for epsilon here
            grid_v[I] = (1 / grid_m[I]) * grid_v[I] # Momentum to velocity
            grid_v[I][2] -= dt * gravity # gravity
        cond = I < bound and grid_v[I] < 0 or I > n_grid - bound and grid_v[
            I] > 0
        grid_v[I] = 0 if cond else grid_v[I]
    ti.block_dim(n_grid)

    ti.static_print("G2P")
    # G2P
    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
        new_v = ti.Vector.zero(float, dim)
        new_C = ti.Matrix.zero(float, dim, dim)
        for i, j, k in ti.static(ti.ndrange(3,3,3)):
            g_v = ti.Vector([0.0, 0.0, 0.0])
            offset = ti.Vector([i, j, k])
            if p_T[p] * grid_T[base + ti.Vector([i,j,k])] == -1:
                point = (offset + base).cast(float) * dx 
                condi, distance, T, normal = isinplane(point, p)
                sg = v[p].dot(normal)
                if sg > 0:
                    g_v = v[p]
                else:
                    g_v = v[p] - normal.normalized() * distance

            dpos = ti.Vector([i,j,k]).cast(float) - fx
            weight = w[i][0] * w[j][1] * w[k][2]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C
        x[p] += dt * v[p]




paused = False

initialize()
res = (512, 512)
window = ti.ui.Window("MLS-MPM-3D", res, vsync=True)

frame_id = 0
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(0.5, 1.0, 1.95)
camera.lookat(0.5, 0.3, 0.5)
camera.fov(55)

def render():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.ambient_light((0, 0, 0))

    scene.particles(x, color=(0.5, 0, 0), radius=particles_radius)
    scene.particles(x_rp, color=(0, 0, 0.5), radius=particles_radius1)
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))

    canvas.scene(scene)

frame = 0

while window.running:
    print(dirr)
    if window.is_pressed(ti.ui.LEFT, 'a'): dirr[0] *= 2
    if window.is_pressed(ti.ui.RIGHT, 'd'): dirr[0] /= 2
    if window.is_pressed(ti.ui.UP, 'w'): dirr[1] *= 2
    if window.is_pressed(ti.ui.DOWN, 's'): dirr[1] /= 2
    if window.is_pressed(ti.ui.SHIFT, 'shift'): dirr[2] *= 2
    if window.is_pressed(ti.ui.SPACE, 'space'): dirr[2] /= 2
    if not paused:
        for s in range(int(5e-3 // dt)):
            # print("substep:{}".format(s))
            substep()
    pos = x.to_numpy()
    render()
    window.show()
    # print(frame)
    if export_file:
        writer = ti.PLYWriter(num_vertices=n_particles)
        writer.add_vertex_pos(pos[:, 0], pos[:, 1], pos[:, 2])
        writer.export_frame(frame, export_file)
    
    frame += 1