# -*- coding: utf-8 -*-
# @Author   : Linsen Zhang
import datetime

import cv2
import os
import heapq
import copy
from multiprocessing import Process
import numpy as np
import math
from sklearn.cluster import DBSCAN
import matplotlib.pyplot as plt
Open = []

Operations = [[0, 1], [0, -1], [-1, 0], [1, 0], [-1, -1], [-1, 1], [1, -1], [1, 1]]
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

    def __lt__(self, another):  # 比较同类对象的不同实例
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

    def __lt__(self, another):  # 比较同类对象的不同实例
        return self.gn < another.gn  # 盲定位

    def __eq__(self, another):
        return self.hashval == another.hashval

# Improvement diagonal distance
def cal_hn(c_state, g_state):
    x = math.pow(c_state[0] - g_state[0], 2)
    y = math.pow(c_state[1] - g_state[1], 2)
    res = math.sqrt(x + y)
    return res


def gen_child(current, goal, hashtable, openlist, hn_func, col_hash, map_list):  # Generate Children Node
    for op in Operations:
        childstate = [i + j for i, j in zip(op, current.state)]
        h = hash(str(childstate))
        if h in hashtable or h in col_hash:  # add by zhanglinsen Determine if it's in the table first
            continue

        # 碰撞检测
        if map_list[childstate[1]][childstate[0]] != 1:
            continue

        # h = hash(str(childstate))
        # if h in hashtable:
        # continue
        hashtable.add(h)

        if Operations.index(op) < 4:
            gn = current.gn + 1  # Step size is 1
        else:
            gn = current.gn + 1.414  # Step size is 1.414


        # 长约束
        try:
            res = curvature_cal(current, 5)
            # print(res)
            if res > 0.7:  # Curvature (the reciprocal of the radius of curvature), the larger the value the smoother it is
                gn = current.gn + 60
        except:
            pass

        hn = hn_func(childstate, goal.state)
        childnode = Node(childstate, current, h, gn, hn)
        current.child.append(childnode)
        heapq.heappush(openlist, childnode)
        global NodeNum
        NodeNum += 1


def gen_child_Blind(current, hashtable, openlist, col_hash, map_list):  # Generate Children Node
    for op in Operations:
        childstate = [i + j for i, j in zip(op, current.state)]
        h = hash(str(childstate))
        if h in hashtable or h in col_hash:  # add by zhanglinsen Determine if it's in the table first
            continue

        # if main.MainWin.cls_collision_detect(childstate):
        if map_list[childstate[1]][childstate[0]] != 1:
            col_hash.add(hash(str(childstate)))
            continue
        # h = hash(str(childstate))
        # if h in hashtable:
        # continue
        hashtable.add(h)

        if Operations.index(op) < 4:
            gn = current.gn + 1
        else:
            gn = current.gn + 1.414


        childnode = Node_Blind(childstate, current, h, gn)
        current.child.append(childnode)
        heapq.heappush(openlist, childnode)


def curvature_cal(point, gap):  # A program to compute curvature by taking points at intervals, with gap being the smoothing radius.
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
    mean_pointlist_ = (pointlist_ + last_pointlist_) / 2
    pointlist__ = (pointlist_ - last_pointlist_) / gap
    result = abs(pointlist__[1]) / pow((1 + pow(mean_pointlist_, 2)), 1.5)
    return result

def curvature_cal_auto(pointlist, gap):
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
            pointlist_ = (current_point_array - last_point_array) / gap
            last_pointlist_ = (last_point_array - prelast_point_array) / gap
            mean_pointlist_ = (pointlist_ + last_pointlist_) / 2
            pointlist__ = (pointlist_ - last_pointlist_) / gap

            result = np.linalg.norm(np.outer(pointlist__, mean_pointlist_)) / pow(np.linalg.norm(mean_pointlist_), 3)
            cur_num += 1
            curvature_list.append(result)
        except:
            for i in range(gap * 2):
                curvature_list.append("null")
            break
    return curvature_list

def find_N_Max(list_, N):

    i = 0
    for i in range(len(list_)):
        if type(list_[i]) == str:
            list_[i] = -1
            i += 1

    res_index = list(map(list_.index, heapq.nlargest(N, list_)))
    return res_index


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
    if list(vector1) == [0, 0] or list(vector2) == [0, 0]:
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

        print("弯曲势能： max={} min={} mean={}".format(round(max(temp_list), 2), round(min(temp_list), 2), round(np.mean(temp_list), 2)))
        return temp_list

def cluster_DBSCAN(point_list, eps=8, min_samples=10):
    pointlist = np.array(point_list)
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

def a_star(initial, goal):

    skeImagePath = "/PNG/label.png"
    uppath = os.path.dirname(os.path.realpath(__file__))
    png_path = uppath + skeImagePath

    picture = cv2.imread(png_path, 0)
    retVal, road_bin = cv2.threshold(picture, 0, 1, cv2.THRESH_BINARY)
    img_list = road_bin.tolist()

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
        if minfn_node == goal_node:
            return minfn_node
        gen_child(minfn_node, goal_node, hashset, Open, cal_hn, col_hashset, img_list)

    return None_node

def Blind_Delivery_2D(initial, goal, displacement, Open, hashset, col_hashset, process_data):
    if process_data[4] == 1 or process_data[5] == 1:
        process_data[4] == -1
        return
    if len(Open) == 0:
        startpos = 0
    else:
        startpos = int(Open[0].gn)
    # print("\033[1;35m[guide_advance]\033[0m \033[1;36m{}->{}\033[0m".format(startpos, displacement))
    process_data[4] = 1
    startt = datetime.datetime.now()


    skeImagePath = "/PNG/label.png"
    uppath = os.path.dirname(os.path.realpath(__file__))
    png_path = uppath + skeImagePath

    picture = cv2.imread(png_path, 0)
    retVal, road_bin = cv2.threshold(picture, 0, 1, cv2.THRESH_BINARY)
    img_list = road_bin.tolist()

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
        gen_child_Blind(min_gn_node, hashset, Open, col_hashset, img_list)
        # print_path(min_gn_node)

    myNodes = Blind_print_result(node_list)
    pathdata = data_process(myNodes)
    # main.path_display(pathdata)

    # 数据回传 多线程
    process_data[0] = pathdata      # Return value
    process_data[1] = Open          # Openlist
    process_data[2] = hashset       # hashset
    process_data[3] = col_hashset   # col_hashset
    process_data[4] = 0             # Run flag (end of program)

    Open_point_cal(Open, process_data)

    currentt = datetime.datetime.now()  # End the clock
    print("\033[1;35m[guide_advance_2D]\033[0m \033[1;36m{}->{}\033[0m computation time={}s".format(startpos, round(current_gn), round((currentt - startt).total_seconds(), 4)))  # Print Runtime

def Open_point_cal(openlist, process_data):
    # startt = datetime.datetime.now()
    # point_set= set()
    point_curvature_list = []
    segPoint_list = []
    temp_list = []
    segment_num = 0
    spaceAngle_list = []
    sum_trace_list = []
    sum_point = []
    sumpoint_list = []
    short_segPoint_list = []
    for i in range(len(openlist)):
        if openlist[i].gn < openlist[0].gn + 1.8:
            point_list = pointlist_stat(openlist[i])
            sum_point.append(point_list[0])
            sum_trace_list.append(point_list)
            sumpoint_list.append(point_list)

    process_data[6] = sumpoint_list.copy()  # 满足要求的openlist转list 包含起点和终点
    # process_data[7] = short_segPoint_list.copy()  # 分段后的点list 不包含起点和终点
    #
    # # 弯曲势能粗删除 过滤
    # if len(bend_energy) > 0:
    #     threshold = 4.0
    #     temp_list = copy.deepcopy(sum_point)
    #     for i in range(len(sum_point)):
    #         min_bend_energy = min(bend_energy)
    #         if bend_energy[i] - min_bend_energy > threshold and collision_list[i] == 0:
    #             temp = copy.deepcopy(sum_point[i])
    #             temp_list.remove(temp)
    #     sum_point = temp_list
    # # point_list = list(map(list, point_set))
    # cluster_list = cluster_DBSCAN(sum_point)
    # process_data[8] = cluster_list

    # currentt = datetime.datetime.now()  # 结束计时
    # itv = round((currentt - startt).total_seconds())
    # print("OpenPoint计算完成 轨迹数:{} 计算用时={}s".format(len(sum_point), itv))

def Blind_Twist(angle, Open, process_data, seg=20, gap=5):
    if process_data[4] == 1 or process_data[5] == 1:
        process_data[5] == -1
        return

    if len(Open) == 0:
        print("Less than minimum interval")
        return

    if Open[0].gn < seg:
        print("Less than minimum interval")
        return

    # print("\033[1;33m[guide_twist]\033[0m pos:{} angle->\033[1;36m{}\033[0m".format(int(Open[0].gn), angle))
    process_data[4] = 1
    startt = datetime.datetime.now()


    skeImagePath = "/PNG/label.png"
    uppath = os.path.dirname(os.path.realpath(__file__))
    png_path = uppath + skeImagePath

    picture = cv2.imread(png_path, 0)
    retVal, road_bin = cv2.threshold(picture, 0, 1, cv2.THRESH_BINARY)
    img_list = road_bin.tolist()

    segPoint_list = []
    segment_num = 0
    sum_seg_len = []
    sum_point = []
    sum_pointlist = []
    short_segPoint_list = []
    spaceAngle_list = []

    for i in range(len(Open)):
        if Open[i].gn < Open[0].gn + 1.8:
            point_list = pointlist_stat(Open[i])
            sum_point.append(point_list[0])
            sum_pointlist.append(point_list)

            current_curvature_list = curvature_cal_auto(point_list, gap)
            seg_len = seg
            if Open[i].gn > seg_len - 1:
                segment_num = int(Open[i].gn / seg_len)
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

            # print(seg_len)
            sum_seg_len.append(seg_len)



    if len(sum_pointlist) == len(process_data[6]):
        sum_pointlist = process_data[6]
        # short_segPoint_list = process_data[7]



    collision_list = []
    if Open[i].gn < seg:

        process_data[9] = []
    else:
        for i in range(len(sum_pointlist)):
            res = circle_collision_detect(sum_pointlist[i][0], img_list, radius=1)
            collision_list.append(res)
        # process_data[9].append(bend_energy)
        process_data[9].append(collision_list)

    after_point_list = []

    for i in range(len(sum_pointlist)):
        gn = Open[0].gn
        last_seg_len = sum_seg_len[i][len(sum_seg_len[i]) - 1]
        new_angle = angle * math.sqrt(last_seg_len / (gn - last_seg_len))
        if type(short_segPoint_list[i][0]) == int:

            temp_point1 = np.array(short_segPoint_list[i])
            temp_point2 = np.array(sum_pointlist[i][len(sum_pointlist[i]) - 2])
        else:
            temp_point1 = np.array(short_segPoint_list[i][0])
            temp_point2 = np.array(sum_pointlist[i][len(sum_pointlist[i]) - 2])

        after_point = auto_rotate(np.array(sum_pointlist[i][0]), np.array(sum_pointlist[i][0]) - temp_point1, new_angle, img_list)
        after_point_list.append(after_point)
        sum_pointlist[i][0] = after_point

    process_data[6] = sum_pointlist
    process_data[7] = short_segPoint_list
    process_data[4] = 0
    cluster_list = cluster_DBSCAN(after_point_list)
    process_data[8] = cluster_list
    currentt = datetime.datetime.now()
    print("\033[1;33m[guide_twist_2D]\033[0m pos:{} Number of trajectories:{} angle->\033[1;36m{}\033[0m Calculation time used={}s".format(int(Open[0].gn), len(sum_seg_len), angle, round((currentt - startt).total_seconds(), 4)))  # Print Runtime

def auto_rotate(point_array, vector, angle, img_list, unit=0.1):
    if angle == 0:
        return vector

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
        new_point = point_rotate_vector(vector, current_angle)

        if round(point_array[1] + new_point[1]) < len(img_list[0]) and round(point_array[1] + new_point[1]) > 0 and round(point_array[0] + new_point[0]) < len(img_list) and round(point_array[0] + new_point[0]) > 0:
            if img_list[round(point_array[1] + new_point[1])][round(point_array[0] + new_point[0])] != 1:  # A collision
                current_angle -= unit
                break
        else:
            break

    if current_angle == 0:
        return point_array.tolist()

    return (point_array + point_rotate_vector(vector, current_angle)).tolist()

def point_rotate_vector(vector, angle, type=1):  # Calculate the rotation of a point in space around a vector The right hand rule is positive against the direction of the fingers type=1 for angles, type=0 for radians
    # Radian conversions
    if type:  # Angle is the angle value
        angle = angle * math.pi / 180
    newpoint = []
    x = vector[0] * math.cos(angle) - vector[1] * math.sin(angle)  # x*cosA-y*sinA
    y = vector[0] * math.sin(angle) + vector[1] * math.cos(angle)  # x*sinA+y*cosA
    newpoint.append(x)
    newpoint.append(y)
    return newpoint

def circle_collision_detect(test_point, img_list, radius=1):  # Circular collision detection Input: points to be tested, set of collision points, test radius Output: collision 1 no collision 0
    startt = datetime.datetime.now()
    direction_array = np.array([[0, -1], [0, 1], [1, -1], [1, 0], [1, 1], [-1, -1], [-1, 0], [-1, 1]])


    temp_point = np.array(test_point)
    points_array = direction_array * radius + temp_point

    for i in range(len(points_array)):
        if round(points_array[i][1]) < len(img_list[0]) and round(points_array[i][1]) > 0 and round(points_array[i][0]) < len(img_list) and round(points_array[i][0]) > 0:
            if img_list[round(points_array[i][1])][round(points_array[i][0])] != 1:
                return 1
        else:
            return 1
    return 0

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

# Print Searching Result
def Blind_print_result(node):
    None_node = Node(0, 0, 0, 0, 0)
    if node == None_node:
        print("Search Fail! No Result!")
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

# Print Searching Result
def print_result(node):
    None_node = Node(0, 0, 0, 0, 0)
    if node == None_node:
        print("Search Fail! No Result!")
    deepth = node.gn
    nodes = []
    while (node != None_node):
        nodes.append(node.state)
        node = node.parent

    for i in range(len(nodes) - 1, -1, -1):
        # print(np.array(nodes[i]))
        pass
        # print('-------------------')
    print("Deepth:", deepth)
    print("NodeNum:", NodeNum)
    return nodes


# Process Searching Result
def data_process(nodes):
    nodes_s2f = nodes[::-1]
    nodes_s2f = [[y for y in x] for x in nodes_s2f]
    pathdata = []
    for i in range(len(nodes_s2f) - 1):
        pathdata.append(tuple(nodes_s2f[i]))
    return pathdata

# startt = datetime.datetime.now()  # A*
# myNode = a_star([179, 504], [139, 125])
# myNodes = print_result(myNode)  # Path data
# A_star_pathdata = data_process(myNodes)
# print(A_star_pathdata)
# currentt = datetime.datetime.now()
# print("Runtime:", (currentt - startt).total_seconds(), 's')  # Print Runtime

