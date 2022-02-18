import taichi as ti
import numpy as np
import os

export_ply = False
export_video = False

export_file = ""
export_video_path = "./video"

if export_ply:
    dir_num = -1
    for root,dirs,files in os.walk('./tmp'):
        # print(dirs)
        for temp_dir in dirs:
            dir_num = max(dir_num,int(temp_dir[3:]))
            # print(dir_num)
    dir_num += 1

    if not os.path.exists("tmp/ply{}".format(dir_num)):
        os.makedirs("tmp/ply{}".format(dir_num))
    
if export_ply:
    export_file = "tmp/ply{}/cut.ply".format(dir_num)

# obj文件路径
obj_file = "./python/ring1.obj"
# 圆环半径0.1，初始圆心位置[0.7, 0.5, 0.5]
# 方块范围[0.3-0.7, 0.0-0.4, 0-0.4]
# move为圆环平移量，[0.0,0.0,0.2]表示移到方块角上
move = [0.0, 0.0, 0.2]

# 读取obj
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
        for i in range(3):
            ps[i] += move[i]
        points.append(ps)
        s = f.readline()

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
        if i== 0:
            print(x_r[i,0] , x_r[i,1] , x_r[i,2])
            print(x_rp[i])
    f.close()
    return x_rp, x_r

ti.init(arch=ti.gpu,advanced_optimization=False)

if export_video:
    video_manager = ti.VideoManager(output_dir=export_video_path, framerate=24, automatic_build=False)

w = 1.0
c=1
dim = 3
quality = 1
bound = 3
n_grid = 32
n_particles = n_grid**dim // 2**(dim - 1)
dx, inv_dx = w / n_grid, float(n_grid) / w
dt = 1e-4 / quality
p_vol, p_rho = (dx*0.5)**3, 1
p_mass = p_vol * p_rho
E, nu = 0.1e4, 0.2
kh = 0.1
mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1+nu) * (1 - 2 * nu))
x = ti.Vector.field(dim, dtype=float, shape=n_particles)
v = ti.Vector.field(dim, dtype=float, shape=n_particles)
C = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles)
F = ti.Matrix.field(dim, dim, dtype=float, shape=n_particles)
color_p = ti.Vector.field(dim, dtype=float, shape=n_particles)
material = ti.field(dtype=int, shape=n_particles)
Jp = ti.field(dtype=float, shape=n_particles)
grid_v = ti.Vector.field(dim, dtype=float, shape=(n_grid, )*dim)
grid_m = ti.field(dtype=float, shape=(n_grid,) * dim)
energy = ti.field(dtype=float, shape=())
x_dir = ti.Vector.field(3, dtype=float, shape=())
center = ti.Vector.field(3, dtype=float, shape=())

# 圆环半径，粒子数
ring_radius = 0.1
n_rseg = 100
r_circle = 0.1*w
r_seg = 0.01*w

x_rp, x_r = read_obj(obj_file)
color_x_rp = ti.Vector.field(3, float, shape=x_rp.shape)
cen_seg = [0.8*w, 0.8*w, 0.8*w]

grid_d = ti.field(dtype=float, shape=(n_grid,) * dim)
grid_A = ti.field(dtype=int, shape=(n_grid, )* dim)
grid_T = ti.field(dtype=int, shape=(n_grid, )*dim)
grid_r = ti.field(dtype=ti.i32, shape=(n_grid,)*dim)

p_d = ti.field(dtype=float, shape=n_particles)
p_A = ti.field(dtype=int, shape=n_particles)
p_T = ti.field(dtype=int, shape=n_particles)
p_n = ti.Vector.field(dim, dtype=float, shape=n_particles)
particles_radius = 0.01
particles_radius1 = 0.005
gravity = 20
r_v = 0.5

colors = ti.Vector([0.0,1.0,0.0,1.0])
colors_spon = ti.Vector([1.0,0.0,0.0,1.0])
@ti.func
def getnormal_i(i):
    vec_p_cen = x_rp[i] - center[None]
    vec_p_cen[1] = 0.0
    vec_p_cen = vec_p_cen.normalized()

    # normal = vec_p_cen
    vec_p_cen *= 0.1
    p_ring = center[None]+vec_p_cen
    normal = (x_rp[i] - p_ring).normalized()
    
    normal[1] = normal[1] / 20
    normal = normal.normalized()
    # if x_rp[i][0]**2 + x_rp[i][2]**2 < ring_radius**2:
    #     normal *= -1 
    return normal

@ti.func
def getnormal_p(i):
    vec_p_cen = x[i] - center[None]
    vec_p_cen[1] = 0.0
    vec_p_cen = vec_p_cen.normalized()
    # normal = vec_p_cen

    vec_p_cen *= 0.1
    p_ring = center[None]+vec_p_cen
    normal = (x[i] - p_ring).normalized()
    # normal[1] = 0.0
    normal[1] = normal[1] / 20
    normal = normal.normalized()
    # if x[i][0]**2 + x[i][2]**2 < ring_radius**2:
    #     normal *= -1

    return normal

@ti.func 
def getdistance_i(i):
    vec_p_cen = x_rp[i] - center[None]
    vec_p_cen[1] = 0.0
    vec_p_cen = vec_p_cen.normalized()
    # normal = vec_p_cen

    vec_p_cen *= 0.1
    p_ring = center[None]+vec_p_cen
    vec_d = p_ring - x_rp[i]
    distance = (vec_d.dot(vec_d))**0.5 - 0.005
    return distance

@ti.func 
def getdistance_p(i):
    vec_p_cen = x[i] - center[None]
    vec_p_cen[1] = 0.0
    vec_p_cen = vec_p_cen.normalized()
    # normal = vec_p_cen

    vec_p_cen *= 0.1
    p_ring = center[None]+vec_p_cen
    vec_d = p_ring - x[i]
    distance = (vec_d.dot(vec_d))**0.5 - 0.005
    return distance

@ti.func
def getT_i(i):
    vec_d = x_rp[i] - center[None]
    T = -1
    if vec_d[0]**2 + vec_d[2]**2 > ring_radius**2:
        T = 1
    return T
    

@ti.func
def getT_p(i):
    vec_d = x[i] - center[None]
    T = -1
    if vec_d[0]**2 + vec_d[2]**2 > ring_radius**2:
        T = 1
    return T


@ti.func
def isoutside(points):
    flag = points[0] > w or points[1] > w or points[2] > w
    return flag


@ti.func
def isinplane(points, i):
    flag = False
    normal = (x_r[i,1]-x_r[i,0]).cross(x_r[i,2]-x_r[i,1])
    normal = normal.normalized()
    d = -1*normal.dot(x_r[i,0])
    xp = ((normal[1]**2 + normal[2]**2)*points[0] - normal[0]*(normal[1]*points[1]+normal[2]*points[2]+d)) / (normal[0]**2+normal[1]**2+normal[2]**2)
    yp = ((normal[2]**2 + normal[0]**2)*points[1] - normal[1]*(normal[0]*points[0]+normal[2]*points[2]+d)) / (normal[0]**2+normal[1]**2+normal[2]**2)
    zp = ((normal[0]**2 + normal[1]**2)*points[2] - normal[2]*(normal[0]*points[0]+normal[1]*points[1]+d)) / (normal[0]**2+normal[1]**2+normal[2]**2)
    p_pro = ti.Vector([xp, yp, zp])
    b1 = (x_r[i,0]-p_pro).cross(x_r[i,1]-p_pro)
    b2 = (x_r[i,1]-p_pro).cross(x_r[i,2]-p_pro)
    b3 = (x_r[i,2]-p_pro).cross(x_r[i,0]-p_pro)
    c1 = b1.dot(b2)
    c2 = b2.dot(b3)
    if c1 > 0 and c2 > 0:
        flag = True

    return flag

@ti.kernel
def initialize():
    center[None] = [0.7, 0.5, 0.5]
    for i in ti.static(range(3)):
        center[None][i] += move[i]
    for i in range(n_particles):
        x[i] = ti.Vector([ti.random()* 0.4 * w + 0.3 * w, ti.random()* 0.4 *w, ti.random()* 0.4 * w + 0.3 * w])
        v[i] = ti.Matrix([0, 0, 0])
        F[i] = ti.Matrix.identity(float, dim)
        Jp[i] = 1

@ti.kernel
def CDF():
    # print(x_dir)
    ti.static_print("CDF")
    center[None] += x_dir[None] * dt * r_v
    for p in x_rp:
        x_rp[p] += x_dir[None] * dt * r_v
        x_r[p,0] += x_dir[None] * dt * r_v
        x_r[p,1] += x_dir[None] * dt * r_v
        x_r[p,2] += x_dir[None] * dt * r_v
    #CDF
    for i in ti.grouped(grid_A):
        grid_A[i] = 0
        grid_T[i] = 0
        grid_d[i] = 0.0
        grid_r[i] = -1

    for p in x_rp:
        color_x_rp[p] = [0.0, 1.0, 0.0]
        base = int(x_rp[p] * inv_dx - 0.5)
        for i, j, k in ti.static(ti.ndrange(3,3,3)):
            offset = ti.Vector([i, j, k])
            point = (offset + base).cast(float) * dx 
            if not isoutside(point):
                condi = isinplane(point, p)
                distance = getdistance_i(i)
                T = getT_i(i)
                normal = getnormal_i(i)
                # sg = normal.dot(x_dir[None])
                if condi and normal.dot(x_dir[None]) > 0:
                    # print(base, offset, point)
                    if T == 1:
                        color_x_rp[p] = [1.0, 0.0, 0.0]
                    else:
                        color_x_rp[p] = [0.0, 0.0, 1.0]
                    if grid_r[base+offset] == -1 or distance < grid_d[base+offset]:
                        grid_d[base+offset] = distance
                        grid_r[base+offset] = p
                        grid_A[base+offset] = 1
                        grid_T[base+offset] = T

    for p in x:
        p_A[p] = 0
        p_T[p] = 0.0
        p_d[p] = 0.0

        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        Tpr = 0.0

        # d_vecs = ti.Vector([0,]*27).cast(float)
        # diag = ti.Matrix.identity(float, 27)
        # temp = ti.Vector([0,0,0,0]).cast(float)
        # Q = ti.Matrix.rows([temp,]*27) 
      
        
        for i, j, k in ti.static(ti.ndrange(3, 3, 3)):
            offset = ti.Vector([i,j,k])
            if grid_A[base+offset] == 1:
                p_A[p] = 1

            # nn = (i+1)*(j+1)*(k+1)-1
            # d_sign = grid_d[base + offset] * grid_T[base + offset] 
            weight = w[i][0] * w[j][1] * w[k][2]
            dpos = (offset.cast(float) - fx) * dx

            # for mn in ti.static(range(27)):
            #     if mn == nn:
            #         d_vecs[mn] = d_sign
            #         diag[mn,mn] = weight
            #         Q[mn,0] = 1
            #         Q[mn,1] = dpos[0]
            #         Q[mn,2] = dpos[1]
            #         Q[mn,3] = dpos[2]
            # Tpr += weight * d_sign
        T = getT_p(p)
        if p_A[p] == 1:
            if p_T[p] == 0:
                p_T[p] = T
                if T == 1:
                    color_p[p] = [0.0, 1.0, 0.0]
                else:
                    color_p[p] = [1.0, 0.0, 0.0]
                # print("nega".format(x[p]))
            p_d[p] = getdistance_p(p)
            # M = Q.transpose() @ diag @ Q
            # M = M.inverse()
            # dist_p = M @ Q.transpose() @ diag @ d_vecs
            # p_d[p] = dist_p[0]
            # p_n[p] = ti.Vector([dist_p[1],dist_p[2],dist_p[3]]).normalized()
            p_n[p] = getnormal_p(p)
        else:
            p_T[p] = 0
            if color_p[p][0] == 0.0 and color_p[p][1] == 0.0:
                color_p[p] = [0.0, 0.0, 1.0]

@ti.kernel
def P2G():
    for I in ti.grouped(grid_m):
        grid_v[I] = ti.zero(grid_v[I])
        grid_m[I] = 0

    for p in x:
        base = (x[p] * inv_dx - 0.5).cast(int)
        fx = x[p] * inv_dx - base.cast(float)
        w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
        F[p] = (ti.Matrix.identity(float, dim) + dt*C[p]) @ F[p]
        h = 0.5
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
            # if False:
                pass
            else:
                offset = ti.Vector([i, j, k])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[i][0] * w[j][1] * w[k][2]
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass
      

    for i, j, k in grid_m:
        if grid_m[i, j, k] > 0: # No need for epsilon here
            grid_v[i, j, k] = (1 / grid_m[i, j, k]) * grid_v[i, j, k] # Momentum to velocity
            grid_v[i, j, k][1] -= dt * gravity # gravity
        
        # seperate boundary      
        if i < 3 and grid_v[i, j, k][0] < 0:          grid_v[i, j, k][0] = 0 # Boundary conditions
        if i > n_grid - 3 and grid_v[i, j, k][0] > 0: grid_v[i, j, k][0] = 0
        # if j < 3 and grid_v[i, j, k][1] < 0:          grid_v[i, j, k][1] = 0
        # if j > n_grid - 3 and grid_v[i, j, k][1] > 0: grid_v[i, j, k][1] = 0
        if j < 3: grid_v[i,j,k] = [0.0, 0.0, 0.0]
        if k < 3 and grid_v[i, j, k][2] < 0:          grid_v[i, j, k][2] = 0
        if k > n_grid - 3 and grid_v[i, j, k][2] > 0: grid_v[i, j, k][2] = 0


@ti.kernel
def G2P():
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
                # condi, distance, T, normal = isinplane(point, p)
                # normal = getnormal(grid_r[base+offset])
                d_v = v[p] - grid_v[base+offset]
                # print(center[None], x[p], p_T[p], p_n[p])
                sg = d_v.dot(p_n[p])
                if sg > 0:
                    g_v = v[p]
                    # print("gv1:{}".format(g_v))
                else:
                    g_v = grid_v[base+offset] + d_v-d_v.dot(p_n[p])*p_n[p]
                    # print("gv2:{}".format(g_v))
                # print(g_v, dt * c * p_n[p])
                g_v += dt * c * p_n[p]
            else:
                g_v = grid_v[base + ti.Vector([i, j, k])]
                # print("gv3:{}".format(g_v))
            dpos = ti.Vector([i,j,k]).cast(float) - fx
            weight = w[i][0] * w[j][1] * w[k][2]
            new_v += weight * g_v
            new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
        v[p], C[p] = new_v, new_C

        if p_T[p] * p_d[p] < 0:
            f_penalty = kh * p_d[p] * p_n[p]
            dv = dt * f_penalty / p_mass
            v[p] += dv
        # print(v[p])
        x[p] += dt * v[p]


paused = False

initialize()
res = (512, 512)
window = ti.ui.Window("MLS-MPM-3D", res, vsync=True)

frame_id = 0
canvas = window.get_canvas()
scene = ti.ui.Scene()
camera = ti.ui.make_camera()
camera.position(1.2, 1.2, 1.2)
camera.lookat(0.5, 0.3, 0.5)

# camera.position(0.5, 1.5, 0.5)
# camera.lookat(0.5, 0.5, 0.5)

camera.fov(55)

def render():
    camera.track_user_inputs(window, movement_speed=0.03, hold_key=ti.ui.RMB)
    scene.set_camera(camera)

    scene.ambient_light((0, 0, 0))

    scene.particles(x, per_vertex_color=color_p, radius=particles_radius)
    scene.particles(x_rp, per_vertex_color=color_x_rp, radius=particles_radius1)
    scene.point_light(pos=(0.5, 1.5, 0.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.5, 1.5), color=(0.5, 0.5, 0.5))
    scene.point_light(pos=(0.5, 1.0, 0.6), color=(1.0, 1.0, 1.0))
    canvas.scene(scene)

frame = 0

while window.running:
    if window.get_event(ti.ui.PRESS):
        if window.event.key == 'r': initialize()
        elif window.event.key in [ti.ui.ESCAPE]: break
    if window.is_pressed(ti.ui.LEFT, 'a'): x_dir[None][0] = 2
    if window.is_pressed(ti.ui.RIGHT, 'd'): x_dir[None][0] = -2
    if window.is_pressed(ti.ui.UP, 'w'): x_dir[None][1] = 2
    if window.is_pressed(ti.ui.DOWN, 's'): x_dir[None][1] = -2
    if window.is_pressed(ti.ui.SHIFT, "shift"):
        x_dir[None][0], x_dir[None][1], x_dir[None][2] = 0, 0, 0
    if window.is_pressed(ti.ui.ALT, "alt"):
        pause = not pause
    if not paused:
        for s in range(int(5e-3 // dt)):
            # print("substep:{}".format(s))
            CDF()
            P2G()
            G2P()
            pass
    # ti.print_kernel_profile_info('count')
    render()
    window.show()
    pos = x.to_numpy()
    if export_video:
        video_manager.write_frame(pos)

    if export_file:
        writer_p = ti.PLYWriter(num_vertices=pos.shape[0])
        writer_p.add_vertex_pos(pos[:, 0], pos[:, 1], pos[:, 2])
        writer_p.export_frame(frame, export_file)

    
    frame += 1
if export_video:
    video_manager.make_video(gif=True, mp4=True)