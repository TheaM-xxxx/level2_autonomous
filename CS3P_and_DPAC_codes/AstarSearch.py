# -*- coding: utf-8 -*-
# @Author   : Linsen Zhang
import heapq
import copy
from multiprocessing import Process

import numpy as np
import datetime

import main
from main import *
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtkmodules.vtkFiltersModeling
import math

from sklearn.cluster import DBSCAN
from sklearn.cluster import MeanShift, estimate_bandwidth
from sklearn.mixture import GaussianMixture
Open = []
global ske_point
global ves_poly
'''
# Go diagonal or straight in the horizontal direction, only straight in vertical
Operations = [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], \
              [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0]]
'''

fix_stepsize = 1.0

Operations = [[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], \
              [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0], [1, 0, 1], [0, 1, 1], \
              [-1, 0, -1], [0, -1, -1], [1, 0, -1], [-1, 0, 1], [0, 1, -1], [0, -1, 1], \
              [1, 1, 1], [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1], [1, 1, -1], \
              [1, -1, 1], [-1, 1, 1]]

Operations = np.array(Operations)
Operations = Operations * fix_stepsize
Operations = Operations.tolist()

NodeNum = 0


# Class of Node State
class Node(object):
    def __init__(self, state, parent, hashval, gn, hn):
        self.state = state
        self.parent = parent
        self.child = []
        self.hashval = hashval
        self.gn = gn
        self.hn = hn
        self.fn = self.gn + self.hn

    def __lt__(self, another):
        return self.fn < another.fn  # A*

    def __eq__(self, another):
        return self.hashval == another.hashval


# Class of Node State
class Node_Blind(object):
    def __init__(self, state, parent, hashval, gn):
        self.state = state
        self.parent = parent
        self.child = []
        self.hashval = hashval
        self.gn = gn

    def __lt__(self, another):
        return self.gn < another.gn

    def __eq__(self, another):
        return self.hashval == another.hashval


## Manhanttan distance
# def cal_hn(c_state, g_state):
#     hn = 0
#     N = len(c_state)
#     for i in range(N):
#         hn += 0.1*abs(c_state[i]-g_state[i])
#     return hn


def cal_hn(c_state, g_state):
    hn = 0
    h_dia = min(abs(c_state[0] - g_state[0]), \
                abs(c_state[1] - g_state[1]))
    h_man = abs(c_state[0] - g_state[0]) + abs(c_state[1] - g_state[1])
    h_ver = abs(c_state[2] - g_state[2])
    hn = np.sqrt(2) * 0.1 * h_dia + 0.1 * (h_man - 2 * h_dia) + h_ver

    # 新方法
    global ske_point
    if len(ske_point) != 0:
        # guide_point = guideNode_select(c_state, 2)
        # ske_hn = cal_point2point_distance(guide_point, c_state)

        ske_hn = cal_ske_hn(c_state, g_state, 2)
        return ske_hn
    else:
        return hn


def cal_ske_hn(c_state, g_state, forward_step):
    current_index = find_nearestPoint(c_state, ske_point)
    guideNode_index = 0
    guideNode = []
    try:
        guideNode_index = current_index + forward_step
        for i in range(guideNode_index, len(ske_point)):
            guideNode.append(ske_point[i])
    except:
        guideNode_index = len(ske_point) - 1
        guideNode.append(ske_point[guideNode_index])

    if len(guideNode) > 1:
        last_node = guideNode[0].copy()
        hn = 0
        for i in range(1, len(guideNode)):
            res = cal_point2point_distance(last_node, guideNode[i])
            hn += res
            last_node = guideNode[i].copy()
        return hn
    else:
        try:
            hn = cal_point2point_distance(c_state, guideNode[0])
        except:
            try:
                hn = cal_point2point_distance(c_state, g_state)
            except:
                g_state_list = []
                g_state_list = g_state[0].copy()
                hn = cal_point2point_distance(c_state, g_state_list)
        return hn


def guideNode_select(c_state, forward_step):
    current_index = find_nearestPoint(c_state, ske_point)
    guideNode_index = 0
    guideNode = []
    try:
        guideNode_index = current_index + 2
        guideNode.append(ske_point[guideNode_index])
    except:
        guideNode_index = len(ske_point) - 1
        guideNode.append(ske_point[guideNode_index])
    return guideNode


def find_nearestPoint(c_state, pointlist):
    minDistance = 99999
    minDistance_index = -1
    for i in range(len(pointlist) - 1):
        res = cal_point2point_distance(c_state, pointlist[i])
        if res < minDistance:
            minDistance = res
            minDistance_index = i
    return minDistance_index


def cal_point2point_distance(point1, point2):
    try:
        distance = math.sqrt(pow(abs(point1[0] - point2[0]), 2) + pow(abs(point1[1] - point2[1]),2) + pow(abs(point1[2] - point2[2]), 2))
    except:
        point1_list = []
        point1_list = point1[0].copy()
        distance = math.sqrt(pow(abs(point1_list[0] - point2[0]), 2) + pow(abs(point1_list[1] - point2[1]),2) + pow(abs(point1_list[2] - point2[2]), 2))
    return distance


lastt = a.datetime.datetime.now()
startt = a.datetime.datetime.now()
def gen_child(current, goal, hashtable, openlist, col_hash):

    points = vtk.vtkPoints()
    for op in Operations:
        childstate = [i + j for i, j in zip(op, current.state)]
        h = hash(str(childstate))
        if h in hashtable or h in col_hash:
            continue
        points.InsertNextPoint(childstate)

    pointsPolydata = vtk.vtkPolyData()
    pointsPolydata.SetPoints(points)
    checkInside = vtkmodules.vtkFiltersModeling.vtkSelectEnclosedPoints()
    checkInside.SetInputData(pointsPolydata)
    checkInside.SetSurfaceData(ves_poly)
    checkInside.SetTolerance(0.001)
    checkInside.Update()

    op_count = 0  # op序号
    for op in Operations:
        childstate = [i + j for i, j in zip(op, current.state)]
        h = hash(str(childstate))
        if h in hashtable or h in col_hash:  # add by zhanglinsen Determine if it's in the table first
            continue

        if bool(1 - checkInside.IsInside(op_count)):
            col_hash.add(hash(str(childstate)))
            continue

        op_count += 1

        # if collision_detect(childstate):
        #     col_hash.add(hash(str(childstate)))
        #     continue

        # h = hash(str(childstate))
        # if h in hashtable:
        # continue
        hashtable.add(h)

        if Operations.index(op) < 6:
            gn = current.gn + 1
        elif Operations.index(op) < 18:
            gn = current.gn + 1.414
        else:
            gn = current.gn + 1.732



        # longitude constraint
        res = []
        try:
            res = curvature_cal(current, 5)
            # print(res)
            if res > 0.7:  # Curvature (the inverse of the radius of curvature), the larger the value, the smoother it is.
                gn = current.gn + 60
        except:
            pass

        hn = cal_hn(childstate, goal.state)
        childnode = Node(childstate, current, h, gn, hn)
        current.child.append(childnode)
        heapq.heappush(openlist, childnode)


        global NodeNum
        NodeNum += 1
        if NodeNum % 100 == 0:
            currentt = a.datetime.datetime.now()
            global lastt, startt
            print("\r{} {}\t itv:{} s\t tt:{} s\t hn:{}".format(NodeNum, childstate, (currentt - lastt).total_seconds(), (currentt - startt).total_seconds(), hn), end="")
            lastt = a.datetime.datetime.now()
        # end = time.time()
        # print(end - start)

def gen_child_Blind(current, goal, hashtable, openlist, col_hash):  # Generate Children Node

    points = vtk.vtkPoints()
    for op in Operations:
        childstate = [i + j for i, j in zip(op, current.state)]
        h = hash(str(childstate))
        if h in hashtable or h in col_hash:
            continue
        points.InsertNextPoint(childstate)

    pointsPolydata = vtk.vtkPolyData()
    pointsPolydata.SetPoints(points)
    checkInside = vtkmodules.vtkFiltersModeling.vtkSelectEnclosedPoints()
    checkInside.SetInputData(pointsPolydata)
    checkInside.SetSurfaceData(ves_poly)
    checkInside.SetTolerance(0.001)
    checkInside.Update()

    op_count = 0
    for op in Operations:
        childstate = [i + j for i, j in zip(op, current.state)]
        h = hash(str(childstate))
        if h in hashtable or h in col_hash:  # add by zhanglinsen Determine if it's in the table first
            continue

        if bool(1 - checkInside.IsInside(op_count)):
            col_hash.add(hash(str(childstate)))
            continue

        op_count += 1

        # # 碰撞检测
        # if collision_detect(childstate):
        #     col_hash.add(hash(str(childstate)))
        #     continue

        # h = hash(str(childstate))
        # if h in hashtable:
        # continue
        hashtable.add(h)

        if Operations.index(op) < 6:
            gn = current.gn + 1 * fix_stepsize
        elif Operations.index(op) < 18:
            gn = current.gn + 1.414 * fix_stepsize
        else:
            gn = current.gn + 1.732 * fix_stepsize


        res = []
        try:
            res = curvature_cal(current, 5)
            # print(res)
            if res > 0.7:  # Curvature (the reciprocal of the radius of curvature), the larger the value the smoother it is
                gn = current.gn + 60
        except:
            pass

        childnode = Node_Blind(childstate, current, h, gn)
        current.child.append(childnode)
        heapq.heappush(openlist, childnode)


def collision_detect(point):
    # startt = time.time()
    points = vtk.vtkPoints()
    points.InsertNextPoint(point)
    pointsPolydata = vtk.vtkPolyData()
    pointsPolydata.SetPoints(points)
    checkInside = vtkmodules.vtkFiltersModeling.vtkSelectEnclosedPoints()

    checkInside.SetInputData(pointsPolydata)
    checkInside.SetSurfaceData(ves_poly)
    checkInside.SetTolerance(0.001)

    checkInside.Update()
    # endt = time.time()
    # print(endt - startt)
    is_inside = bool(1 - checkInside.IsInside(0))

    del checkInside
    del points
    del pointsPolydata

    return is_inside


def cal_vel_diameter(skeleton_pointlist, ves_poly, cal_step=0.1):
    print("Diameter calculation in progress...")
    diameter_list = []
    global_min_diameter = 999999
    try:
        for point in skeleton_pointlist:
            start_time = datetime.datetime.now()
            directions = [[1, 0, 0], [0, 1, 0], [0, 0, 1], [1, 1, 0], \
                          [-1, 1, 0], [1, 0, 1], [0, 1, 1], [1, 0, -1], \
                          [0, 1, -1], [1, 1, 1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1]]
            min_diameter = 999999
            for i in directions:
                search_rate = cal_step
                dir = [oper * search_rate for oper in i]
                np_point = np.array(point)
                np_dir = np.array(dir)
                step = 1
                positve_len = 0
                negetive_len = 0
                while 1:
                    current_pos = np_point + np_dir * step
                    points = vtk.vtkPoints()
                    points.InsertNextPoint(current_pos)
                    pointsPolydata = vtk.vtkPolyData()
                    pointsPolydata.SetPoints(points)
                    checkInside = vtkmodules.vtkFiltersModeling.vtkSelectEnclosedPoints()
                    checkInside.SetInputData(pointsPolydata)
                    checkInside.SetSurfaceData(ves_poly)
                    checkInside.SetTolerance(0.01)
                    checkInside.Update()
                    is_inside = bool(1 - checkInside.IsInside(0))
                    if is_inside:
                        positve_len = search_rate * step
                        del checkInside
                        del points
                        del pointsPolydata
                        break
                    step += 1
                step = 1
                while 1:
                    current_pos = np_point - np_dir * step
                    points = vtk.vtkPoints()
                    points.InsertNextPoint(current_pos)
                    pointsPolydata = vtk.vtkPolyData()
                    pointsPolydata.SetPoints(points)
                    checkInside = vtkmodules.vtkFiltersModeling.vtkSelectEnclosedPoints()
                    checkInside.SetInputData(pointsPolydata)
                    checkInside.SetSurfaceData(ves_poly)
                    checkInside.SetTolerance(0.01)
                    checkInside.Update()
                    is_inside = bool(1 - checkInside.IsInside(0))
                    if is_inside:
                        negetive_len = search_rate * step
                        del checkInside
                        del points
                        del pointsPolydata
                        break
                    step += 1
                current_diameter = positve_len + negetive_len
                if current_diameter < min_diameter:
                    min_diameter = current_diameter
            end_time = datetime.datetime.now()
            itv_time = (end_time - start_time).total_seconds()
            print(skeleton_pointlist.index(point) + 1, "/", len(skeleton_pointlist), "\t", "calibre:", round(min_diameter, 3), "\t", "itv:", itv_time, "s", "\t","Estimated time:", round((len(skeleton_pointlist) - skeleton_pointlist.index(point) + 1)*itv_time, 1), "s")
            diameter_list.append(min_diameter)
            if min_diameter < global_min_diameter and len(diameter_list) > 2:
                global_min_diameter = min_diameter
        if global_min_diameter == 999999:
            global_min_diameter = -1
        return diameter_list, global_min_diameter
    except:
        return -1


def op_cal(_current):
    last_op = np.array(_current.state) - np.array(_current.parent.state)
    return last_op


def vector_angle_cal(_op0, _op1):
    x = np.array(_op0)
    y = np.array(_op1)

    if (x == y).all():
        return 0


    l_x = np.sqrt(x.dot(x))
    l_y = np.sqrt(y.dot(y))



    dian = x.dot(y)



    cos_ = dian / (l_x * l_y)

    angle_hu = np.arccos(cos_)

    angle_d = angle_hu * 180 / np.pi

    return angle_d

def curvature_cal(point, gap):
    current_point_array = np.array(point.state)
    temp_point = point
    for i in range(gap):
        temp_point = temp_point.parent
    last_point_array = np.array(temp_point.state)
    for i in range(gap):
        temp_point = temp_point.parent
    prelast_point_array = np.array(temp_point.state)
    pointlist_ = (current_point_array - last_point_array) / gap
    last_pointlist_ = (last_point_array - prelast_point_array) / gap
    mea_pointlist_ = (pointlist_ + last_pointlist_) / 2
    pointlist__ = (pointlist_ - last_pointlist_) / gap
    result = math.sqrt(
        pow((mea_pointlist_[1] * pointlist__[2] - mea_pointlist_[2] * pointlist__[1]), 2) + pow(
            (mea_pointlist_[2] * pointlist__[0] - mea_pointlist_[0] * pointlist__[2]), 2) +
        pow((mea_pointlist_[0] * pointlist__[1] - pointlist__[0] * mea_pointlist_[1]), 2)) / pow(
        (math.sqrt(pow(mea_pointlist_[0], 2) + pow(mea_pointlist_[1], 2) + pow(mea_pointlist_[2], 2))), 3)
    return result

def a_star(initial, goal, skeleton_point_list, Reader):

    vascular_polydata = Reader.GetOutput()  #
    Reader.Update()
    global ves_poly
    ves_poly = vascular_polydata

    global ske_point
    ske_point = skeleton_point_list
    inithn = cal_hn(initial, goal)
    None_node = Node(0, 0, 0, 0, 0)  # Set a head node
    init_node = Node(initial, None_node, hash(str(initial)), 0, inithn)
    goal_node = Node(goal, 0, hash(str(goal)), 0, 0)
    if init_node == goal_node:
        print("Initial=Goal! Done!")

    Open = []

    Open.append(init_node)
    heapq.heapify(Open)

    hashset = set()
    hashset.add(init_node.hashval)

    col_hashset = set()
    col_hashset.add(init_node.hashval)

    while (len(Open) != 0):
        minfn_node = heapq.heappop(Open)
        # if minfn_node == goal_node:
        if is_end(minfn_node, goal_node, 1.8):  #
            return minfn_node
        gen_child(minfn_node, goal_node, hashset, Open, col_hashset)

    return None_node


def Blind_Delivery(initial, goal, displacement, hn_func, genchild_func, Open, hashset, col_hashset, process_data, skeleton_point_list, poly_filePath):
    if process_data[4] == 1 or process_data[5] == 1:
        process_data[4] == -1
        return
    if len(Open) == 0:
        startpos = 0
    else:
        startpos = int(Open[0].gn)
    print("\033[0;30;44m[Process_Start]\033[0m\033[1;35m[guide_advance]\033[0m \033[1;36m{}->{}\033[0m".format(startpos, displacement))
    process_data[4] = 1
    startt = datetime.datetime.now()
    # 读取模型文件
    if poly_filePath.endswith(".STL") or poly_filePath.endswith(".stl"):
        Reader = vtk.vtkSTLReader()  # stl file
        Reader.SetFileName(poly_filePath)
        Reader.Update()
    elif poly_filePath.endswith(".OBJ") or poly_filePath.endswith(".obj"):
        Reader = vtk.vtkOBJReader()  # obj file
        Reader.SetFileName(poly_filePath)
        Reader.Update()
    vascular_polydata = Reader.GetOutput()
    global ves_poly
    ves_poly = vascular_polydata
    global ske_point
    ske_point = skeleton_point_list

    None_node = Node_Blind(0, 0, 0, 0)
    init_node = Node_Blind(initial, None_node, hash(str(initial)), 0)
    goal_node = Node_Blind(goal, 0, hash(str(goal)), 0)

    if len(Open) == 0:
        Open.append(init_node)
        heapq.heapify(Open)
        hashset = set()
        col_hashset = set()
        hashset.add(init_node.hashval)
        col_hashset.add(init_node.hashval)

    # while (len(Open) != 0):
    node_list = []
    current_gn = 0
    while (len(Open) != 0) and current_gn < displacement:
        min_gn_node = heapq.heappop(Open)
        current_gn = min_gn_node.gn

        node_list = min_gn_node
        genchild_func(min_gn_node, goal_node, hashset, Open, col_hashset)
        # print_path(min_gn_node)

    myNodes = Blind_print_result(node_list)
    pathdata = a.data_process(myNodes)
    # main.path_display(pathdata)

    # 数据回传 多线程
    process_data[0] = pathdata      # return value
    process_data[1] = Open          # Openlist
    process_data[2] = hashset       # hashset
    process_data[3] = col_hashset   # col_hashset
    process_data[4] = 0  #


    OpenPoint_cal = Process(target=Open_point_cal, args=(Open, process_data, poly_filePath, ))
    OpenPoint_cal.start()

    currentt = datetime.datetime.now()
    print("\033[1;34m[Process_End]\033[0m\033[1;35m[guide_advance]\033[0m \033[1;36m{}->{}\033[0m computation time={}s".format(startpos, round(current_gn), round((currentt - startt).total_seconds())))  # Print Runtime

def auto_rotate(point, vector, angle, unit=1):  # Auto-rotation program with collision detection Returns the point after the maximum possible rotation unit: unit of angular calculation (degrees)
    if angle == 0:
        return point
    # Determining the direction of rotation
    if angle < 0:
        unit = - unit
    current_angle = 0

    while 1:
        current_angle += unit
        if current_angle > angle and angle > 0:
            current_angle -= unit
            break
        if current_angle < angle and angle < 0:
            current_angle -= unit
            break
        new_point = point_rotate_vector(point, vector, current_angle)
        if collision_detect(new_point):  # strike
            current_angle -= unit
            break
    return point_rotate_vector(point, vector, current_angle)

def point_rotate_vector(point, vector, angle, type=1):  # Calculation of the rotation of a space point around a vector The right hand rule is positive against the direction of the fingers type=1 for angles, type=0 for radians
    old_x = point[0]
    old_y = point[1]
    old_z = point[2]
    #
    sum_pow = math.pow(vector[0], 2) + math.pow(vector[1], 2) + math.pow(vector[2], 2)
    sum_sqrt = math.sqrt(sum_pow)

    if sum_sqrt == 0:
        return point

    vx = vector[0] / sum_sqrt
    vy = vector[1] / sum_sqrt
    vz = vector[2] / sum_sqrt

    #
    if type:  #
        angle = angle * math.pi / 180
    s = math.sin(angle)  # sin
    c = math.cos(angle)  # cos
    new_x = (vx * vx * (1 - c) + c) * old_x + (vx * vy * (1 - c) - vz * s) * old_y + (vx * vz * (1 - c) + vy * s) * old_z
    new_y = (vy * vx * (1 - c) + vz * s) * old_x + (vy * vy * (1 - c) + c) * old_y + (vy * vz * (1 - c) - vx * s) * old_z
    new_z = (vx * vz * (1 - c) - vy * s) * old_x + (vy * vz * (1 - c) + vx * s) * old_y + (vz * vz * (1 - c) + c) * old_z
    newpoint = []
    newpoint = [new_x, new_y, new_z]
    return newpoint

def curvature_cal_auto(pointlist, gap):  # pointlist type: list, pointlist direction: current point to start point
    curvature_list = []
    cur_num = 0
    if len(pointlist) < gap * 2 + 1:
        curvature_list.append("error")
        return curvature_list
    while 1:
        try:
            current_point_array = np.array(pointlist[cur_num])
            last_point_array = np.array(pointlist[cur_num + gap])
            prelast_point_array = np.array(pointlist[cur_num + 2 * gap])
            pointlist_ = (current_point_array - last_point_array) / 2
            last_pointlist_ = (last_point_array - prelast_point_array) / 2
            pointlist__ = (pointlist_ - last_pointlist_) / 2

            result = math.sqrt(
                pow((pointlist_[1] * pointlist__[2] - pointlist_[2] * pointlist__[1]), 2) + pow(
                    (pointlist_[2] * pointlist__[0] - pointlist_[0] * pointlist__[2]), 2) +
                pow((pointlist_[0] * pointlist__[1] - pointlist__[0] * pointlist_[1]), 2)) / pow(
                (math.sqrt(pow(pointlist_[0], 2) + pow(pointlist_[1], 2) + pow(pointlist_[2], 2))), 3)
            cur_num += 1
            curvature_list.append(result)
        except:
            for i in range(gap * 2):
                curvature_list.append("null")
            break
    return curvature_list

def pointlist_stat(openlist_):
    point_list = []
    temp_point = openlist_
    while 1:
        try:
            point_list.append(temp_point.state)
            temp_point = temp_point.parent
        except:
            break
    return point_list

def sphere_collision_detect(test_point, vascular_polydata, radius=1, tolerance=0.01):  # Sphere collision detection Inputs: point to be measured, vessel model, test radius, tolerance Outputs: collision 1 no collision 0
    startt = datetime.datetime.now()  # 开始计时
    direction_array = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], \
                  [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0], [1, 0, 1], [0, 1, 1], \
                  [-1, 0, -1], [0, -1, -1], [1, 0, -1], [-1, 0, 1], [0, 1, -1], [0, -1, 1], \
                  [1, 1, 1], [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1], [1, 1, -1], \
                  [1, -1, 1], [-1, 1, 1]])

    # Generate a ball point cloud
    temp_point = np.array(test_point)
    points = vtk.vtkPoints()
    points_array = direction_array * radius + temp_point
    for i in range(len(points_array)):
        points.InsertNextPoint(points_array[i])
    pointsPolydata = vtk.vtkPolyData()
    pointsPolydata.SetPoints(points)

    # 包围集
    checkInside = vtkmodules.vtkFiltersModeling.vtkSelectEnclosedPoints()
    checkInside.SetInputData(pointsPolydata)
    checkInside.SetSurfaceData(vascular_polydata)  #
    checkInside.SetTolerance(tolerance)
    checkInside.Update()

    for i in range(len(points_array)):
        if bool(1 - checkInside.IsInside(i)):
            return 1
    return 0

def Open_point_cal(openlist, process_data, poly_filePath, seg_len=40, gap=5):  # Delivered handler functions
    startt = datetime.datetime.now()  # Start the clock
    # point_set= set()
    point_curvature_list = []
    segPoint_list = []
    temp_list = []
    segment_num = 0
    spaceAngle_list = []
    sum_pointlist = []
    sum_point = []
    short_segPoint_list = []
    for i in range(len(openlist)):
        if openlist[i].gn < openlist[0].gn + 1.8:
            point_list = pointlist_stat(openlist[i])  #
            sum_point.append(point_list[0])
            sum_pointlist.append(point_list)
            current_curvature_list = curvature_cal_auto(point_list, gap)
            # seg_len = 40
            if openlist[i].gn > seg_len - 1:
                segment_num = int(openlist[i].gn / seg_len)
                segPoint_index_list = find_N_Max(current_curvature_list, segment_num)
                segPoint_index_list.sort()
                segPoint_index_list = (np.array(segPoint_index_list) + gap).tolist()
                current_segPoint_list = []
                current_segPoint_list.append(point_list[len(point_list) - 2])
                for j in range(len(segPoint_index_list)):
                    current_segPoint_list.append(point_list[segPoint_index_list[j]])
                    short_segPoint_list.append(point_list[segPoint_index_list[j]])
                current_segPoint_list.append(point_list[0])
                if(len(segPoint_index_list) == segment_num):
                    segPoint_list.append(current_segPoint_list)
            point_curvature_list.append(current_curvature_list)
            # point_set.add(tuple(openlist[i].state))
    # print(segPoint_list)
    for i in range(len(segPoint_list)):
        spaceAngle_list.append(point2Angle(segPoint_list[i]))
    # print(spaceAngle_list)
    bend_energy = bend_Energy_cal(spaceAngle_list)


    collision_list = []
    if len(bend_energy) == 0:

        process_data[9] = []
    else:
        # 读取模型文件
        if poly_filePath.endswith(".STL") or poly_filePath.endswith(".stl"):
            Reader = vtk.vtkSTLReader()  # stl file
            Reader.SetFileName(poly_filePath)
            Reader.Update()
        elif poly_filePath.endswith(".OBJ") or poly_filePath.endswith(".obj"):
            Reader = vtk.vtkOBJReader()  # obj file
            Reader.SetFileName(poly_filePath)
            Reader.Update()
        vascular_polydata = Reader.GetOutput()
        global ves_poly
        ves_poly = vascular_polydata
        for i in range(len(sum_pointlist)):
            res = sphere_collision_detect(sum_pointlist[i][0], ves_poly, radius=1, tolerance=0.01)
            collision_list.append(res)
        process_data[9].append(bend_energy)
        process_data[9].append(collision_list)

    process_data[6]= sum_pointlist.copy()
    process_data[7] = short_segPoint_list.copy()


    if len(bend_energy) > 0:
        threshold = 4.0
        temp_list = copy.deepcopy(sum_point)
        for i in range(len(sum_point)):
            min_bend_energy = min(bend_energy)
            if bend_energy[i] - min_bend_energy > threshold and collision_list[i] == 0:
                temp = copy.deepcopy(sum_point[i])
                temp_list.remove(temp)
        sum_point = temp_list
    # point_list = list(map(list, point_set))
    cluster_list= cluster_DBSCAN(sum_point)
    process_data[8] = cluster_list

    currentt = datetime.datetime.now()
    itv = round((currentt - startt).total_seconds())
    print("OpenPointCalculations completed Number of trajectories:{} computation time={}s".format(len(sum_point), itv))

def bend_Energy_cal(spaceAngle_list):
    temp_list = []
    for i in range(len(spaceAngle_list)):
        sum_temp = 0
        for j in range(len(spaceAngle_list[i])):
            sum_temp += spaceAngle_list[i][j]
        temp_list.append(sum_temp)
    if len(temp_list) == 0:

        return []
    else:

        print("bending potential： sum={} max={} min={} mean={}".format(round(sum(temp_list), 2), round(max(temp_list), 2), round(min(temp_list), 2), round(np.mean(temp_list), 2)))
        return temp_list


def Blind_Twist(initial, goal, angle, hn_func, genchild_func, Open, hashset, col_hashset, process_data, poly_filePath, seg=20, gap=5):
    if process_data[4] == 1 or process_data[5] == 1:
        process_data[5] == -1
        return

    if len(Open) == 0:
        print("Less than minimum interval")
        return

    if Open[0].gn < 40:
        print("Less than minimum interval")
        return

    print("\033[0;30;44m[Process_Start]\033[0m\033[1;33m[guide_twist]\033[0m pos:{} angle->\033[1;36m{}\033[0m".format(int(Open[0].gn), angle))
    process_data[4] = 1
    startt = datetime.datetime.now()


    if poly_filePath.endswith(".STL") or poly_filePath.endswith(".stl"):
        Reader = vtk.vtkSTLReader()  # stl file
        Reader.SetFileName(poly_filePath)
        Reader.Update()
    elif poly_filePath.endswith(".OBJ") or poly_filePath.endswith(".obj"):
        Reader = vtk.vtkOBJReader()  # obj file
        Reader.SetFileName(poly_filePath)
        Reader.Update()
    vascular_polydata = Reader.GetOutput()
    global ves_poly
    ves_poly = vascular_polydata

    segPoint_list = []
    segment_num = 0
    sum_seg_len = []
    sum_point = []
    sum_pointlist = []
    short_segPoint_list = []
    spaceAngle_list = []
    for i in range(len(Open)):
        if Open[i].gn < Open[0].gn + 1.8:
            point_list = pointlist_stat(Open[i])  #
            sum_point.append(point_list[0])
            sum_pointlist.append(point_list)
            current_curvature_list = curvature_cal_auto(point_list, 5)
            # seg_len = 40
            if Open[i].gn > seg - 1:
                segment_num = int(Open[i].gn / seg)  #
                segPoint_index_list = find_N_Max(current_curvature_list, segment_num)
                segPoint_index_list.sort(reverse=True)
                segPoint_index_list = (np.array(segPoint_index_list) + gap).tolist()
                current_segPoint_list = []
                current_segPoint_list_twist = []
                current_segPoint_list_twist.append(point_list[len(point_list) - 2])
                for j in range(segment_num):
                    current_segPoint_list.append(point_list[segPoint_index_list[j]])
                    current_segPoint_list_twist.append(point_list[segPoint_index_list[j]])
                current_segPoint_list_twist.append(point_list[0])
                if (len(segPoint_index_list) == segment_num):
                    segPoint_list.append(current_segPoint_list_twist)

            current_segPoint_list.reverse()
            short_segPoint_list.append(current_segPoint_list)

            current_gn = Open[0].gn
            temp_Node = Open[i]
            seg_len = []
            # for j in range(len(current_segPoint_list)):
            while 1:
                temp_Node = temp_Node.parent
                try:
                    if temp_Node.state == current_segPoint_list[j]:
                        seg_len.append(current_gn - temp_Node.gn)
                        current_gn = temp_Node.gn
                        break
                except:
                    break
            try:
                seg_len.append(temp_Node.gn)
            except:
                seg_len.append(0)

            # print(seg_len)  # 打印各段长度
            sum_seg_len.append(seg_len)

    for i in range(len(segPoint_list)):
        spaceAngle_list.append(point2Angle(segPoint_list[i]))
    # print(spaceAngle_list)
    bend_energy = bend_Energy_cal(spaceAngle_list)

    if len(sum_pointlist) == len(process_data[6]):
        sum_pointlist = process_data[6]
        short_segPoint_list = process_data[7]


    collision_list = []
    if len(bend_energy) == 0:
        #
        process_data[9] = []
    else:
        reader = vtk.vtkOBJReader()  # obj file
        reader.SetFileName("3D.obj")
        vascular_polydata = reader.GetOutput()  #
        reader.Update()
        for i in range(len(sum_pointlist)):
            res = sphere_collision_detect(sum_pointlist[i][0], ves_poly, radius=1, tolerance=0.01)  #
            collision_list.append(res)
        process_data[9].append(bend_energy)
        process_data[9].append(collision_list)

    after_point_list = []
    try:
        for i in range(len(sum_pointlist)):
            gn = Open[0].gn  #
            last_seg_len = sum_seg_len[i][len(sum_seg_len[i]) - 1]
            new_angle = angle * math.sqrt(last_seg_len / (gn - last_seg_len))
            if type(short_segPoint_list[i][0]) == int:

                temp_point1 = np.array(short_segPoint_list[i])  #
                temp_point2 = np.array(sum_pointlist[i][len(sum_pointlist[i]) - 2])
            else:
                temp_point1 = np.array(short_segPoint_list[i][0])
                temp_point2 = np.array(sum_pointlist[i][len(sum_pointlist[i]) - 2])

            after_point = auto_rotate(sum_pointlist[i][0], list(temp_point1 - temp_point2), new_angle)
            after_point_list.append(after_point)
            sum_pointlist[i][0] = after_point
    except:

        print("\033[0;31msum_seg_len Error!\033[0m")


    if len(bend_energy) > 0:
        threshold = 4.0
        temp_list = copy.deepcopy(after_point_list)
        for i in range(len(sum_point)):
            min_bend_energy = min(bend_energy)
            if bend_energy[i] - min_bend_energy > threshold and collision_list[i] == 0:
                temp = copy.deepcopy(after_point_list[i])
                temp_list.remove(temp)
        after_point_list = temp_list

    process_data[6] = sum_pointlist
    process_data[7] = short_segPoint_list
    process_data[4] = 0
    cluster_list = cluster_DBSCAN(after_point_list)
    process_data[8] = cluster_list
    currentt = datetime.datetime.now()
    print("\033[1;34m[Process_End]\033[0m\033[1;33m[guide_twist]\033[0m Number of trajectories:{} angle->\033[1;36m{}\033[0m Calculation time used={}s".format(len(sum_seg_len), angle, round((currentt - startt).total_seconds()))) # Print Runtime

def cluster_DBSCAN(point_list, eps=8, min_samples=10):
    y_pred = DBSCAN(eps=eps, min_samples=min_samples).fit_predict(pointlist)
    # ax = plt.subplot(111, projection='3d')
    # ax.scatter(pointlist[:, 0], pointlist[:, 1], pointlist[:, 2], c=y_pred)
    # plt.show()
    point_cluster_list = []
    for i in range(y_pred.max() + 1):
        temp_list = []
        for j in range(len(y_pred)):
            if y_pred[j] == i:
                temp_list.append(point_list[j])
        point_cluster_list.append(temp_list)
    return point_cluster_list

def cluster_MeanShift(point_list, quantile=0.2):  # Return: list of clustered point sets
    pointlist_array = np.array(point_list)
    #
    bandwidth = estimate_bandwidth(pointlist_array, quantile=quantile)

    # Implement the Mean Shift clustering algorithm
    ms = MeanShift(bandwidth=bandwidth)
    ms.fit(pointlist_array)
    labels = ms.labels_

    point_cluster_list = []
    for i in range(labels.max() + 1):
        temp_list = []
        for j in range(len(labels)):
            if labels[j] == i:
                temp_list.append(pointlist_array[j])
        point_cluster_list.append(temp_list)
    return point_cluster_list

def cluster_GMM(point_list, n_components=3):  # Return: list of clustered point sets
    pointlist_array = np.array(point_list)

    gmm = GaussianMixture(n_components=n_components)
    gmm.fit(point_list)
    labels = gmm.predict(point_list)

    point_cluster_list = []
    for i in range(labels.max() + 1):
        temp_list = []
        for j in range(len(labels)):
            if labels[j] == i:
                temp_list.append(pointlist_array[j])
        point_cluster_list.append(temp_list)
    return point_cluster_list

def twist_Energy_cal():
    pass

def find_N_Max(list_, N):

    i = 0
    for i in range(len(list_)):
        if type(list_[i]) == str:
            list_[i] = -1
            i += 1

    res_index = list(map(list_.index, heapq.nlargest(N, list_)))
    return res_index

def print_path(minfn_node):
    res = []
    current = minfn_node
    while 1:
        try:
            res.append(current.parent.state)
            current = current.parent
        except:
            print(res)
            break


def is_end(current_node, end_node, tolerance):  # tolerance Error radius
    distance = math.sqrt(pow(abs(current_node.state[0] - end_node.state[0]), 2) + pow(abs(current_node.state[1] - end_node.state[1]), 2) + pow(abs(current_node.state[2] - end_node.state[2]), 2))
    if distance <= tolerance:
        return True
    else:
        return False

def point2Angle(pointlist):
    if len(pointlist) > 2:
        angle_list = []
        for i in range(len(pointlist) - 2):
            vector1 = np.array(pointlist[i + 1]) - np.array(pointlist[i])
            vector2 = np.array(pointlist[i + 2]) - np.array(pointlist[i + 1])
            angle = spaceAngle_cal(vector1, vector2)
            angle_list.append(angle)
        return angle_list
        # print(angle_list)

def spaceAngle_cal(vector1, vector2):
    if list(vector1) == [0, 0, 0] or list(vector2) == [0, 0, 0]:
        return 0
    res = np.dot(vector1, vector2)
    norm1 = np.linalg.norm(vector1)
    norm2 = np.linalg.norm(vector2)
    if res / norm1 / norm2 < -1:

        return math.pi
    elif res / norm1 / norm2 > 1:

        return 0
    else:
        re = math.acos(res / norm1 / norm2)
        return re

# Print Searching Result
def Blind_print_result(node):
    None_node = Node(0, 0, 0, 0, 0)
    if node == None_node:
        print("Search failed! No results!")
    deepth = node.gn
    nodes = []
    while (node != None_node):
        nodes.append(node.state)
        node = node.parent

    for i in range(len(nodes) - 1, -1, -1):
        # print(np.array(nodes[i]))
        pass
        # print('-------------------')
    # print("Deepth:", deepth)
    # print("NodeNum:", NodeNum)
    return nodes


# Print Searching Result
def print_result(node):
    None_node = Node(0, 0, 0, 0, 0)
    if node == None_node:
        print("Search failed! No results!")
    deepth = node.gn
    nodes = []
    while (node != None_node):
        nodes.append(node.state)
        node = node.parent

    for i in range(len(nodes) - 1, -1, -1):
        # print(np.array(nodes[i]))
        pass
        # print('-------------------')
    print("Node depth:", deepth)
    print("Number of nodes:", NodeNum)
    return nodes


# Process Searching Result
def data_process(nodes):
    nodes_s2f = nodes[::-1]
    nodes_s2f = [[y for y in x] for x in nodes_s2f]
    pathdata = []
    for i in range(len(nodes_s2f) - 1):
        pathdata.append(tuple(nodes_s2f[i]))
    return pathdata
