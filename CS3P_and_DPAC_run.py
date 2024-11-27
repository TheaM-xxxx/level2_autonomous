# -*- coding: utf-8 -*-
# @Author   : Linsen Zhang
import math
import copy

import sys
import threading
from threading import Lock
import time

import CS3P_and_DPAC_codes.QT_UI as QT_UI
import pyautogui

from PyQt5.QtChart import QDateTimeAxis,QValueAxis,QSplineSeries,QChart,QChartView

from PyQt5 import QtCore, QtGui, QtWidgets
from PyQt5.QtGui import *
from PyQt5.QtCore import *
from PyQt5.QtWidgets import *
import json

import socket
from socket import SHUT_RDWR
import struct

import cv2
import os, psutil
from skimage import morphology, draw

import numpy as np
from scipy.optimize import leastsq

from PyQt5.QtOpenGL import QGLWidget
import vtk
from vtkmodules.qt.QVTKRenderWindowInteractor import QVTKRenderWindowInteractor
import vtkmodules.vtkFiltersModeling
# import vmtk

import open3d as o3d

import operator
import datetime

from memory_profiler import profile

import CS3P_and_DPAC_codes.AstarSearch as a
import CS3P_and_DPAC_codes.AStar_2D as a_2d
# import AstarSearch_mp as a

import skeletor as sk
import trimesh
import multiprocessing
from multiprocessing import Process, Manager

from stl import mesh

import CS3P_and_DPAC_codes.WM_COPYDATA as WM_COPYDATA

import inspect
import ctypes

import CS3P_and_DPAC_codes.socket2robot as socket2robot

from vtk.util import numpy_support

import pyqtgraph as pg

import array
import random

mapDraw_flag = False
global_polydata = 0

import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import OneHotEncoder
# import DWT
import openpyxl
import keyboard
import re

import CS3P_and_DPAC_codes.torque_period as torque_period

from scipy.interpolate import interp1d
from scipy.spatial.distance import euclidean
from scipy.spatial.distance import cdist

# model_path = "13_.STL"

# model_path = "vein_rot.STL"

# 动物
model_path = "CS3P_and_DPAC_codes/Model-data/238_smooth.stl"


# centerline file
centerline_path = "CS3P_and_DPAC_codes/Point-data/left_center.npy"

# Point file
point_file = "CS3P_and_DPAC_codes/Point-data/left.npy"

# device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class SKE_Node(object):
    def __init__(self, index):
        self.index = index
        self.parent = None

class SKE_Plan(object):
    def __init__(self, startindex, goalindex):
        self.start = SKE_Node(startindex)
        self.end = SKE_Node(goalindex)
        self.nodeList = [self.start]
        self.path_pointlist = []

    def planning(self):
        print("Begin centerline path planning.")
        currentNode = self.start
        flag = 0
        n = 0
        hash_table = []
        h = hash(str(currentNode.index))
        hash_table.append(h)
        while 1:
            flag = 0
            currentNode = self.nodeList[n]
            for i in range(len(main.skeleton_edge.tolist())):
                if currentNode.index in main.skeleton_edge[i].tolist():
                    templist = main.skeleton_edge[i].tolist().copy()
                    templist.remove(currentNode.index)
                    new_node = copy.deepcopy(self.nodeList[0])
                    new_node.index = templist[0]
                    new_node.parent = self.nodeList.index(currentNode)
                    h = hash(str(new_node.index))
                    if h not in hash_table:
                        hash_table.append(h)
                        self.nodeList.append(new_node)
                    flag = 1
                    pass
            if flag == 0:
                print("Track not found!")
                break
            if new_node.index == self.end.index or currentNode.index == self.end.index:
                print("Centerline path planning complete!")
                break
            n += 1

        # Find the end point in the generated point set
        path = []
        end_index = 0
        for i in range(len(self.nodeList)):
            if self.nodeList[i].index == self.end.index:
                end_index = i
                # print("index:", i)
        # Trajectory back
        last_index = end_index
        while self.nodeList[last_index].parent is not None:
            node = self.nodeList[last_index]
            path.append(node.index)
            last_index = node.parent
        if path[len(path) - 1] != self.end.index:
            path.append(self.nodeList[last_index].index)
        # Trajectory printing
        path = path[::-1]
        print("Centerline point trajectory:", path)


        # Convert index in path to point
        for i in range(len(path)):
            self.path_pointlist.append(main.skeleton_point[path[i]].tolist())
        if main.ske_display_actor != 0:
            main.ren.RemoveActor(main.ske_display_actor)

        # main.skeleton_display(self.path_pointlist, linecolor="lightgreen", linewidth=0.1)
        main.skeleton_plan_point = self.path_pointlist.copy()
        skeleton_point_save_array = np.array(main.skeleton_plan_point)
        np.save("skeleton_point", skeleton_point_save_array)

        return path

class interactor(vtk.vtkInteractorStyleTrackballCamera):  # Interactive Classes Mouse Space Point Selection
    def __init__(self, parent=None):
        self.AddObserver("RightButtonPressEvent", self.RightButtonPressEvent)
        self.AddObserver("MiddleButtonPressEvent", self.MiddleButtonPressEvent)
        # self.AddObserver("LeftButtonPressEvent", self.LeftButtonPressEvent)
        self.i = 1
        self.controlpoints = {}
        self.actor = []

        self.modelFilePath = main.modelFilePath

        self.firstpoint = []
        self.lastpoint = []
        self.firstpoint_index = 0
        self.endpoint_index = 0

    def MiddleButtonPressEvent(self, obj, event):
        data = self.GetModelData(main.Reader)

        clickPos = self.GetInteractor().GetEventPosition()  # Get 2D image points

        # Pick from this location
        picker = self.GetInteractor().GetPicker()  # Initialize the picker action
        picker.Pick(clickPos[0], clickPos[1], 0, self.GetDefaultRenderer())  # Self-defined rendering functions

        value_list = []
        for j in range(len(self.controlpoints)):
            value_list.append(self.controlpoints[j + 1])

        # If CellId = -1, nothing was picked
        if value_list.count(data.GetPoint(picker.GetPointId())) == 0:
            point_position = data.GetPoint(picker.GetPointId())

            # Coordinates fitted to the entity
            self.controlpoints[self.i] = point_position

            list_point_position = []
            list_point_position = list(point_position)
            nearest_point = self.find_NearestPoint(list_point_position, main.skeleton_point)


            # Create a sphere
            sphereSource = vtk.vtkSphereSource()
            sphereSource.SetCenter(nearest_point[0])
            # sphereSource.SetRadius(0.2)
            sphereSource.SetRadius(0.5)

            print(nearest_point[0])

            if nearest_point[0][1] > 200:
                NDI_Point = vtk.vtkPoints()
                NDI_Point.InsertNextPoint(nearest_point[0])
                NDI_Point_polydata = vtk.vtkPolyData()
                NDI_Point_polydata.SetPoints(NDI_Point)
                NDI_vertex = vtk.vtkVertexGlyphFilter()
                NDI_vertex.SetInputData(NDI_Point_polydata)
                NDI_mapper = vtk.vtkPolyDataMapper()
                NDI_mapper.SetInputConnection(NDI_vertex.GetOutputPort())
                self.NDI_Pos_actor = vtk.vtkActor()
                self.NDI_Pos_actor.SetMapper(NDI_mapper)
                self.NDI_Pos_actor.GetProperty().SetPointSize(30)  # Change the size of the point
                colors = vtk.vtkNamedColors()
                self.NDI_Pos_actor.GetProperty().SetColor(colors.GetColor3d("black"))  # Dot color
                self.NDI_Pos_actor.GetProperty().SetOpacity(1)
                main.ren.AddActor(self.NDI_Pos_actor)
                main.iren.Initialize()


    def RightButtonPressEvent(self, obj, event):
        data = self.GetModelData(main.Reader)

        clickPos = self.GetInteractor().GetEventPosition()  # Get 2D image points

        # Pick from this location
        picker = self.GetInteractor().GetPicker()  # Initialize the picker action
        picker.Pick(clickPos[0], clickPos[1], 0, self.GetDefaultRenderer())  # Self-defined rendering functions

        value_list = []
        for j in range(len(self.controlpoints)):
            value_list.append(self.controlpoints[j + 1])

        # If CellId = -1, nothing was picked
        if value_list.count(data.GetPoint(picker.GetPointId())) == 0:
            point_position = data.GetPoint(picker.GetPointId())

            # Coordinates fitted to the entity
            self.controlpoints[self.i] = point_position

            list_point_position = []
            list_point_position = list(point_position)
            nearest_point = self.find_NearestPoint(list_point_position, main.skeleton_point)


            # Create a sphere
            sphereSource = vtk.vtkSphereSource()
            sphereSource.SetCenter(nearest_point[0])
            # sphereSource.SetRadius(0.2)
            sphereSource.SetRadius(0.5)

            print(nearest_point[0])

            # Create a mapper and actor
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(sphereSource.GetOutputPort())

            self.actor.append(vtk.vtkActor())
            self.actor[self.i - 1].SetMapper(mapper)
            self.actor[self.i - 1].GetProperty().SetColor(0, 0, 0)
            self.GetDefaultRenderer().AddActor(self.actor[self.i - 1])
            self.i += 1
        self.OnRightButtonDown()

        # print(main.skeleton_edge)
        try:  # Avoid selection points that no longer report errors on the model
            if len(self.firstpoint) != 0 and nearest_point[0].tolist() != self.firstpoint:
                main.ren.RemoveActor(main.text_2d_actor)
                self.lastpoint = nearest_point[0].tolist()
                self.endpoint_index = main.skeleton_point.tolist().index(self.lastpoint)

                print("The End:", [round(x, 2) for x in nearest_point[0]], end=" ")
                print("Terminal serial number:", self.endpoint_index)
                main.path_endpoint = [nearest_point[0][0], nearest_point[0][1], nearest_point[0][2]]  # Global path planning endpoint setting
                main.actor_vascular.GetProperty().SetColor(1, 0, 0)  # Color switching when selecting points Arteries Veins
                main.iren.Initialize()
                skeleton_plan = SKE_Plan(self.firstpoint_index, self.endpoint_index)
                skeleton_plan.planning()
            else:
                self.firstpoint = nearest_point[0].tolist()
                self.firstpoint_index = main.skeleton_point.tolist().index(self.firstpoint)
                for i in range(len(nearest_point[0])):
                    # Starting coordinates rounded Reduced calculations
                    nearest_point[0][i] = round(nearest_point[0][i])
                    pass
                print("Starting point:", nearest_point[0][0], nearest_point[0][1], nearest_point[0][2], end=" ")
                print("Starting point serial number:", self.firstpoint_index)
                main.path_startpoint = [nearest_point[0][0], nearest_point[0][1], nearest_point[0][2]]  # Global route planning starting point setup
                main.text_2d_actor.SetInput("Select EndPoint")
                main.ren.AddActor(main.text_2d_actor)
        except:
            pass
        return


    def GetModelData(self, Reader):
        data = Reader.GetOutput()
        # print("nodes number：", data.GetNumberOfPoints())
        if data.GetNumberOfPoints() == 0:
            raise ValueError("No point data could be loaded from " + self.modelFilePath)
            return None
        return data

    def points_distance(self, point1, point2):  # Calculate the distance between two points Input: two points [list] Return: distance
        return math.sqrt(math.pow(point1[0]-point2[0], 2)+math.pow(point1[1]-point2[1], 2)+math.pow(point1[2]-point2[2], 2))

    def find_NearestPoint(self, test_point, pointlist):
        shortest_distance = 99999
        shortest_point = []
        test_point = list(test_point)
        pointlist = list(pointlist)
        for i in range(len(pointlist)):
            res = self.points_distance(test_point, pointlist[i])
            if res < shortest_distance:
                shortest_distance = res
                shortest_point.clear()
                shortest_point.append(pointlist[i])

        return shortest_point


class MainWin(QtWidgets.QMainWindow):
    # Create the main form
    def __init__(self, parent=None):
        super(MainWin, self).__init__(parent)
        self.x0 = 0
        self.x1 = 0
        self.json_list = []
        self.tcp_socket = 0
        self.connect_flag = False
        cols, rows = 512, 512
        self.bin_img = [[0 for col in range(cols)] for row in range(rows)]
        self.ske_map = []
        self.mapDraw_flag = False
        self.roadDraw_flag = False
        self.dividepoint_list = [0, 0]
        self.divide_flag = False
        self.sorted_ske_map = []
        self.handshake =        [0xAA, 0xAA, 0x01, 0x00, 0x00, 0x00, 0x25, 0xf6, 0x01, 0x00, 0x32, 0x00, 0x00, 0x01, 0x02, 0x52, 0x55, 0x55]
        self.operate =          [0xAA, 0xAA, 0x01, 0x00, 0x00, 0x00, 0xA8, 0xC3, 0x00, 0x00, 0x41, 0x00, 0x00, 0x01, 0x01, 0xAF, 0x55, 0x55]
        self.operate_slave =    [0xAA, 0xAA, 0x01, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x00, 0x41, 0x00, 0x00, 0x01, 0x01, 0x44, 0x55, 0x55]
        self.handshake_flag = False
        self.connectSuccess_flag = False
        self.operateTimer = 0
        self.operateCount = 10
        self.tcp_client = 0
        self.CtrlFlag = 0
        self.BeginFlag = 0
        self.RecCheckSum = 0

        self.FRAME_HEAD = 0xAA
        self.FRAME_TAIL = 0x55
        self.FRAME_CTRL = 0xA5
        self.MaxPackageSize = 40
        self.LastByte = 0
        self.RevOffset = 0
        self.RevOffset = 0
        self.g_LostPackage_Net = 0
        self.PROTOCOL_TRUE = 1
        self.PROTOCOL_FALSE = 0
        self.BeginFlag = self.PROTOCOL_FALSE
        self.g_ValidDataCount_Net = 0
        self.g_RxBuf_Net = [0 for x in range(0,self.MaxPackageSize)]
        self.canMessage_StdId = 0
        self.canMessage_RTR = 0
        self.canMessage_DLC = 0
        self.canMessage_Data = []
        self.g_tMaster_ActualAdvCountOfGw1 = 0
        self.g_tMaster_ActualAdvSpeedOfGw1 = 0
        self.g_tMaster_ActualRotCountOfGw1 = 0
        self.g_tMaster_ActualRotSpeedOfGw1 = 0
        self.g_tMaster_ActualAdvCountOfCath = 0
        self.g_tMaster_ActualAdvSpeedOfCath = 0
        self.g_tMaster_ActualRotCountOfCath = 0
        self.g_tMaster_ActualRotSpeedOfCath = 0
        self.recPosNow = 0
        self.recVelNow = 0
        self.recCurrent = 0
        self.recTorque = 0

        self.centralwidget = 0
        self.vtkWidget = 0
        self.ren = 0
        self.iren = 0

        self.vascular_opacityRate = 100
        self.first_3D = False

        self.skeImagePath = "/PNG/label.png"
        self.pcdFilePath = "3D.pcd"

        self.modelFilePath = model_path

        self.points = []
        self.face = []
        self.edge = []
        self.normal_vector = []

        self.init_touch_list = []
        self.touch_list = []
        self.path_startpoint = [8, 0, 63]
        self.path_endpoint = [-0.64, 277.47, 83.79]

        self.Reader = 0
        self.vascular_polydata = 0

        self.dyn_line_point = 0
        self.dyn_lineSource = 0
        self.dyn_actor_line = 0
        self.dyn_num = 0
        self.pointlist = 0

        self.path_display_flag = False
        self.path_display_actor = 0

        self.skeleton_point = []
        self.skeleton_edge = []
        self.skeleton_display_flag = []
        self.ske_display_actor = 0
        self.skeleton_plan_point = []

        self.actor_vascular = 0

        self.text_3d_actor = 0
        self.text_3d_textSource = 0
        self.text_2d_actor = 0
        self.text_2d_textSource = 0


        self.A_star_pathdata = []

        self.m = Manager()
        # 0             1       2           3               4                   5                   6               7               8               9
        # pathdata, openlist, hashset, col_hashset, guidewire blind localization displacement run flag, guidewire blind localization angle run flag, points in Openlist that meet constraints, segmented point list, clustered point set list, Open that meets requirements, [bending energy, spherical end collision list]
        self.process_data = self.m.list([0, [], 0, 0, 0, 0, [], [], [], []])

        self.Openlist = []
        self.hashset = set()
        self.col_hashset = set()

        # 2D盲定位变量
        self.process_data_2d = self.m.list([0, [], 0, 0, 0, 0, [], [], [], []])
        self.Openlist_2d = []
        self.hashset_2d = set()
        self.col_hashset_2d = set()
        self.guidewire_displacement_2d = 0
        self.guidewire_angle_2d = 0

        self.guidewire_control_msg = []  # Format: [absolute displacement, rotation]
        self.catheter_control_msg = []  # Format: [absolute displacement, rotation]
        self.frame_guidewire_control_msg = 0
        self.frame_catheter_control_msg = 0
        self.guidewire_zero_position = 0  # [Guide wire] 0 position displacement
        self.catheter_zero_position = 0  # [Catheter] 0 position displacement
        self.guidewire_current_position = 0  # [Guide Wire] Current Absolute Position
        self.guidewire_current_angle = 0  # [Guide Wire] Current Absolute Angle
        self.catheter_current_position = 0  # [Conduit] Current absolute position
        self.catheter_current_angle = 0  # [Ducting] Current absolute angle
        self.guidewire_position_factor = 1.0  # [Wire Guide] Displacement vs. Model Scale Relationships
        self.guidewire_angle_factor = 1.0  # [Wire guide] Angle value and actual rotation angle ratio relationship
        self.catheter_position_factor = 1.0  # [Conduit] Displacement and Model Scale Relationships
        self.catheter_angle_factor = 1.0  # [Conduit] Angle value proportional to the actual rotation angle

        self.flag_Blind_Switch = 0
        self.blind_timer = 0

        self.Open_point_list = []
        self.current_display_pointlist = []

        self.blindpoint_actor_list = []
        self.flag_blindpoint_actor = False

        self.thread_lock = Lock()

        self.advance_test = 0
        self.twist_test = 0

        self.blind_process = 0

        self.NDI_listern_thread = 0
        self.NDI_Read_timer = 0
        self.NDI_data = [""]
        self.NDI_data_old = [""]
        self.NDI_data_processed = []
        self.NDI_Pos_Record = []
        self.NDI_Pos_actor = 0

        self.NDI_Pos_ICP = 0

        self.Transform_matrix = 0

        self.x_offset = -27
        self.y_offset = -1300
        self.z_offset = 53

        self.robot_dis = 0
        self.robot_ang = 0

        # self.adv_rate = 34.1426 / 4096
        self.adv_rate = 1 / 113.6534
        self.rot_rate = 22.566

        self.Node_save = []
        self.blind_current_step = 0
        self.blindpoint_actor_list1 = []
        self.Node_list = []

        self.robot_dis_1 = 0
        self.auto_del_flag = 0
        self.voice_play = 0

        self.cam_point = []

        # 目标位置 target_dis
        self.target_dis = 550
        # 力矩检测标志位
        self.torque_flag = 0
        self.list_torque = []
        self.torque_plot = 0
        self.torque_curve = 0
        self.torque_pen = 0

        # Neural network inputs [Displacement, velocity, torque]
        self.NN_input = []
        self.force_state = 0  # Tip-top collision state

        self.ske_dis = []
        self.branch_point = []
        self.ske_path_point = []

        self.block_pos = 0

        self.ske_point_num = 0

        self.workbook = 0
        self.sheet = 0
        self.NN_input_save = []

        self.keyboard_thread = 0
        self.torque_label_input = 0

        self.list_res_dis = []
        self.torqueinit_flag = 0
        self.torqueinit_data = [[], []]
        self.avg_list = []
        self.phase_bias = 0
        self.torque_res_dis = 0
        self.torque_noise = 0

        self.current_avg_pointset = []
        self.sector_avg_branch_list = []

        self.avg_point_list = []
        self.cath_comp_path_actor_list = []

        self.last_DWT_Dis = 0

        self.cath_comp_flag = 0

        self.data_record = []
        self.data_record1 = []


        self.new_ren = 0
        self.new_iren= 0

        self.multpoint_dis_flag = 1
        self.cur_instr_dis = 0


        self.vtkWidget2 = 0


        self.swap_widget_flag = 0


        self.video_capture = 0

        self.prev_frame = None

        self.current_tor = 0

        self.torque_delay = 0

        self.force_2D_actor = 0

        self.current_light = 0
        self.current_camera_pos = [0, 0, 0]
        self.current_camera_focus = [0, 0, 0]

        self.rot_dir = 0

    def closeEvent(self, event):
        # Window exit handlers
        try:
            # Delete individual processes
            if self.blind_timer != 0:
                self.blind_timer.cancel()
            if self.NDI_Read_timer != 0:
                self.NDI_Read_timer.cancel()
            if self.blind_process != 0:
                self.blind_process.terminate()
            # Stop the keyboard listener thread
            self.keyboard_thread.stop()
            # Wait for the keyboard listener thread to finish
            self.keyboard_thread.join()
            print("\033[1;30;41m※※※The system is down.※※※\033[0m")
            sys.stdout = sys.__stdout__
            super().closeEvent(event)
            # Force a shutdown Terminate thread processes that can't be shut down above
            os._exit(0)
        except:
            print("\033[1;30;41mSystem Shutdown Failed!\033[0m")

    def init_video_display(self):
        self.video_capture = cv2.VideoCapture(0)  # Turn on the camera
        if not self.video_capture.isOpened():
            print("Unable to open the camera")
            # sys.exit(1)

        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_frame)
        self.timer.start(50)

        # Create a QLabel to display the image
        self.video_display_label = QLabel(self)
        ui.verticalLayout_7.addWidget(self.video_display_label)

    def update_frame(self):
        ret, frame = self.video_capture.read()

        if ret:
            if self.prev_frame is None:
                self.prev_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                flow = cv2.calcOpticalFlowFarneback(self.prev_frame, gray, None, 0.5, 3, 15, 3, 5, 1.2, 0)

                magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])
                magnitude = cv2.normalize(magnitude, None, 0, 255, cv2.NORM_MINMAX)
                magnitude = magnitude.astype(np.uint8)

                # Raise the threshold for recognition
                threshold = 150

                # Calculate a binary image of the region of change
                change_area = cv2.threshold(magnitude, threshold, 255, cv2.THRESH_BINARY)[1]

                # Finding the contours of areas of change
                contours, _ = cv2.findContours(change_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Fill the change area with red on the original image
                result_frame = frame.copy()
                red_color = (0, 255, 255)  # The order is BGR not RGB!
                for contour in contours:
                    cv2.fillPoly(result_frame, [contour], red_color)  # Fill with red

                image = cv2.cvtColor(result_frame, cv2.COLOR_BGR2RGB)
                h, w, c = image.shape
                q_image = QImage(image.data, w, h, w * c, QImage.Format_RGB888)
                pixmap = QPixmap.fromImage(q_image)

                # Display images on QLabel
                self.video_display_label.setPixmap(pixmap)

                self.prev_frame = gray

    def paintEvent(self, event):  # Drawing program All drawing operations need to be done here.
        painter = QPainter(self)


        painter.setPen(QPen(Qt.red, 1, Qt.DashLine))
        if self.mapDraw_flag is True:
            x = 0
            # Connecting the dots
            while x < len(self.json_list):

                if x > 0:
                    painter.drawLine(int(self.json_list[x - 1][0]), int(self.json_list[x - 1][1]),
                                      int(self.json_list[x][0]), int(self.json_list[x][1]))
                    x = x + 1
                elif x == 0:
                    x = x + 1
            # Sealing
            painter.drawLine(int(self.json_list[0][0]), int(self.json_list[0][1]),
                             int(self.json_list[len(self.json_list) - 1][0]), int(self.json_list[len(self.json_list) - 1][1]))

            painter.setPen(QPen(Qt.yellow, 2, Qt.SolidLine))
            painter.drawLine(int(self.json_list[0][0]),int(self.json_list[0][1])-5,int(self.json_list[len(self.json_list) - 1][0]),int(self.json_list[len(self.json_list) - 1][1])-5)
            self.update()
        else:
            self.update()

        if self.roadDraw_flag is True:
            if self.divide_flag is False:
                self.Line_sort()
                self.divide_flag = True  # Calculate duplicate points and then stop counting
            painter.setPen(QPen(Qt.white, 1, Qt.DashLine))
            for x in range(len(self.sorted_ske_map)):
                if x < 50:
                    painter.setPen(QPen(QColor(255, 255, 255), 1, Qt.DashLine))
                elif x < 100:
                    painter.setPen(QPen(QColor(233, 233, 233), 1, Qt.DashLine))
                elif x < 150:
                    painter.setPen(QPen(QColor(200, 200, 200), 1, Qt.DashLine))
                elif x < 200:
                    painter.setPen(QPen(QColor(160, 160, 160), 1, Qt.DashLine))
                elif x < 250:
                    painter.setPen(QPen(QColor(120, 120, 120), 1, Qt.DashLine))
                elif x < 300:
                    painter.setPen(QPen(QColor(137, 102, 138), 1, Qt.DashLine))
                elif x < 350:
                    painter.setPen(QPen(QColor(90, 83, 157), 1, Qt.DashLine))
                elif x < 400:
                    painter.setPen(QPen(QColor(64, 154, 176), 1, Qt.DashLine))
                elif x < 450:
                    painter.setPen(QPen(QColor(206, 210, 30), 1, Qt.DashLine))
                elif x < 500:
                    painter.setPen(QPen(QColor(224, 125, 16), 1, Qt.DashLine))
                elif x < 550:
                    painter.setPen(QPen(QColor(240, 0, 0), 1, Qt.DashLine))

                painter.drawPoint(self.sorted_ske_map[x][0], self.sorted_ske_map[x][1])  # Drawing a road map The coordinate system is reversed
                self.update()
        else:
            self.update()

        # 2d blind positioning endpoint display program
        if len(self.process_data_2d[6])!= 0:
            painter.setPen(QPen(QColor(255, 0, 0), 2, Qt.DashLine))
            for i in range(len(self.process_data_2d[6])):
                try:
                    painter.drawPoint(round(self.process_data_2d[6][i][0][0]), round(self.process_data_2d[6][i][0][1]))
                except:
                    painter.drawPoint(round(self.process_data_2d[6][i][0]), round(self.process_data_2d[6][i][1]))
            self.update()

    def Line_sort(self):  # Sorting program for line segments after erosion
        # Find the start point, (the point furthest down) the point with the largest Y
        startpoint_Y = 250
        startpoint_X = 0
        lastpoint_X = 0
        lastpoint_Y = 0
        search_range = 1
        # Find the initial point, the bottom point #
        for x in range(len(self.ske_map)):
            if self.ske_map[x][0] > startpoint_Y:
                startpoint_Y = self.ske_map[x][0]
                startpoint_X = self.ske_map[x][1]
        # print("startpoint_Y=", startpoint_Y, "startpoint_X=", startpoint_X)
        # Sorting from the initial point
        self.sorted_ske_map.clear()
        self.sorted_ske_map.append([startpoint_X,startpoint_Y])
        lastpoint_X = startpoint_X
        lastpoint_Y = startpoint_Y
        for x in range(len(self.ske_map)):
            for y in range(len(self.ske_map)):
                if abs(self.ske_map[y][0] - startpoint_Y) <= search_range \
                        and abs(self.ske_map[y][1] - startpoint_X) <= search_range \
                        and (self.ske_map[y][0] != startpoint_Y or self.ske_map[y][1] != startpoint_X) \
                        and (self.ske_map[y][0] != lastpoint_Y or self.ske_map[y][1] != lastpoint_X):
                    self.sorted_ske_map.append([self.ske_map[y][1],self.ske_map[y][0]])
                    lastpoint_X = startpoint_X
                    lastpoint_Y = startpoint_Y
                    startpoint_Y = self.ske_map[y][0]
                    startpoint_X = self.ske_map[y][1]
        # print(self.sorted_ske_map) # Sorted lines

    def mousePressEvent(self, event):  # Read the mouse point
        self.flag = True
        self.x0 = event.x()
        self.y0 = event.y()
        if self.x0 <= 512 and self.y0 <= 512:
            print("select point:", self.x0, self.y0)
        else:
            # print("Not DSA Area")
            pass
        self.point_calibration = True
        self.update()

    def tcp_connect(self):  # TCP connection program
        try:
            print("TCP Connecting")
            ui.pushButton_5.setEnabled(False)
            ui.pushButton_6.setEnabled(True)

            # IP address
            server_ip = ui.lineEdit.text()
            # Port number
            server_port = ui.lineEdit_2.text()

            # Create sockets
            self.tcp_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            # Set up port multiplexing so that ports are released immediately after program exit
            self.tcp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)

            # Server-side connection program
            # Binding ports
            self.tcp_socket.bind(("", int(server_port)))
            # Set up the listener
            self.tcp_socket.listen(128)

            self.connect_flag = True

            self.tcp_client, tcp_client_address = self.tcp_socket.accept()
            # Create multi-threaded objects
            tcp_server_thread = threading.Thread(target=self.tcp_receive_server, args = (self.tcp_client,tcp_client_address)).start()


            print("TCP_Connect Success ")
            main.status.showMessage('TCP_Connect Success', 1000)

        except Exception as e:
            print(e)
            print("TCP Connect Failed")
            main.status.showMessage('TCP Connect Failed', 1000)

    def tcp_disconnect(self):  # TCP disconnection procedure
        try:

            self.tcp_socket.close()
            self.connect_flag = False
            print("TCP Disconnect")
            main.status.showMessage('TCP Disconnect', 1000)
            ui.pushButton_5.setEnabled(True)
            ui.pushButton_6.setEnabled(False)
        except Exception as e:
            print(e)
            print("TCP Disconnect Failed")
            main.status.showMessage('TCP Disconnect Failed', 1000)

    def tcp_send(self, send_buffer):  # TCP sender program
        try:
            print("data length:",len(send_buffer),',',end='')
            data = struct.pack("%dB" % (len(send_buffer)), *send_buffer)
            print("Data:", data)
            # Client mode
            self.tcp_socket.send(data)
            # Server-side model
            self.tcp_client.send(data)
            #print("Data is sent")
            #main.status.showMessage('Data is Sent', 1000)
        except:
            main.status.showMessage('TCP Send Failed', 1000)

    def tcp_receive_server(self, client, clientaddress):  # TCP receiver program ( runs in a separate thread ) server side
        try:
            print("TCP Rec_Thread Created")
            print("Robot_Address:{}".format(clientaddress))
            recBuffer_size = 1024
            while self.connect_flag:
                data = client.recv(recBuffer_size)
                print('new data length:', len(data), ',', 'data:', end="")
                for x in range(len(data)):
                    print(hex(data[x]), ' ', end="")
                print()  # Row line feeds

                # if list(data) == self.handshake:  # Receive a handshake from the slave
                if data[10] == 0x32 and data[11] == 0x00 and data[12] == 0x00 and data[13] == 0x01:  # Receive a handshake from the slave
                    self.handshake_flag = True
                    self.operateTimer = threading.Timer(0.5, self.cycleSend_operate)
                    self.operateTimer.start()
                    print("OPERATE SEND")
                # Successful handshake.
                # elif list(data) == self.operate_slave:  # Receive an operation from the slave
                elif data[10] == 0x42 and data[11] == 0x00 and data[12] == 0x00 and data[13] == 0x01:  # Receive an operation from the slave
                    self.handshake_flag = False
                    self.connectSuccess_flag = 1  # Handshake success sign
                    ui.pushButton_5.setText("HS Succuess")
                    print("Successful handshake.")

                if len(data)!= 0:
                    for index in range(len(data)):
                        if self.ParseByteFromNet(data[index]) == self.PROTOCOL_TRUE:
                            self.canMessage_StdId = self.g_RxBuf_Net[8] + (self.g_RxBuf_Net[9] << 8)
                            self.canMessage_RTR = self.g_RxBuf_Net[10]
                            self.canMessage_DLC = self.g_RxBuf_Net[11]
                            for can_index in range(self.canMessage_DLC):
                                self.canMessage_Data[can_index] = self.g_RxBuf_Net[can_index+12]
                            self.parseSlaveMsg()  # Slave sensor data parsing

                if not data:
                    self.connect_flag = 0
                    print("Slave disconnected")
                    self.client.close()
                    # print("TCP Connect ERROR")
                    # main.status.showMessage('TCP Connect ERROR', 1000)
                    break
        except Exception as e:
            print(e)
            print("TCP_Receive Failed")
            main.status.showMessage('TCP_Receive Failed', 1000)

    def tcp_receive(self):  # TCP receiver program ( runs in a separate thread ) Clients
        try:
            print("TCP Rec_Thread Created")
            recBuffer_size = 1024
            while self.connect_flag:
                data = self.tcp_socket.recv(recBuffer_size)
                print('new data length:', len(data), ',', 'data:', end="")
                for x in range(len(data)):
                    print(hex(data[x]), ' ', end="")
                print()

                if list(data) == self.handshake:  # Receive a handshake from the slave
                    self.handshake_flag = True
                    self.operateTimer = threading.Timer(0.5, self.cycleSend_operate)
                    self.operateTimer.start()
                # A successful handshake
                elif list(data) == self.operate_slave:  # Receive an operation from the slave
                    self.handshake_flag = False
                    self.connectSuccess_flag = 1  # Handshake success sign
                    ui.pushButton_5.setText("HS Succuess")

                if len(data)!= 0:
                    for index in range(len(data)):
                        if self.ParseByteFromNet(data[index]) == self.PROTOCOL_TRUE:
                            self.canMessage_StdId = self.g_RxBuf_Net[8] + (self.g_RxBuf_Net[9] << 8)
                            self.canMessage_RTR = self.g_RxBuf_Net[10]
                            self.canMessage_DLC = self.g_RxBuf_Net[11]
                            for can_index in range(self.canMessage_DLC):
                                self.canMessage_Data[can_index] = self.g_RxBuf_Net[can_index+12]
                            self.parseSlaveMsg()  # Slave sensor data parsing

                if not data:
                    self.connect_flag = 0
                    # print("TCP Connect ERROR")
                    # main.status.showMessage('TCP Connect ERROR', 1000)
                    break
        except:
            print("TCP_Receive Failed")
            main.status.showMessage('TCP_Receive Failed', 1000)

    def parseSlaveMsg(self):  # Slave sensor feedback parsing program
        if self.canMessage_StdId == 0x62:  # Position and velocity feedback S2M_RPL_PosAndVel
            if self.canMessage_RTR == 0x01:  # Guide wire delivery
                self.g_tMaster_ActualAdvCountOfGw1 = (self.canMessage_Data[0]) + (self.canMessage_Data[1] << 8) \
                                                     + (self.canMessage_Data[2] << 16) + (self.canMessage_Data[3] << 24)
                self.g_tMaster_ActualAdvSpeedOfGw1 = (self.canMessage_Data[4]) + (self.canMessage_Data[5] << 8) \
                                                     + (self.canMessage_Data[6] << 16) + (self.canMessage_Data[7] << 24)

                self.guidewire_current_position = self.g_tMaster_ActualAdvCountOfGw1  # [Guide Wire] Current Absolute Displacement
                relative_position = self.guidewire_current_position - self.guidewire_zero_position  # Relative displacement
                if relative_position >= 0:  # Greater than or equal to 0
                    self.guidewire_control_msg.append([relative_position * self.guidewire_position_factor, 0])  # Relative displacement is all positive.
                else:
                    pass
                    # self.guidewire_control_msg.append([0, 0])  # [Guide Wire] Relative Displacement
                print("[Guidewire delivery]__ Absolute displacement:{} ".format(self.guidewire_current_position), "__ Speed:{}".format(self.g_tMaster_ActualAdvSpeedOfGw1))

            elif self.canMessage_RTR == 0x02:  # Guide wire rotation
                self.g_tMaster_ActualRotCountOfGw1 = (self.canMessage_Data[0]) + (self.canMessage_Data[1] << 8) \
                                                     + (self.canMessage_Data[2] << 16) + (self.canMessage_Data[3] << 24)
                self.g_tMaster_ActualRotSpeedOfGw1 = (self.canMessage_Data[4]) + (self.canMessage_Data[5] << 8) \
                                                     + (self.canMessage_Data[6] << 16) + (self.canMessage_Data[7] << 24)

                if self.guidewire_current_angle != 0:
                    relative_angle = self.g_tMaster_ActualRotCountOfGw1 - self.guidewire_current_angle  # [Guide Wire] Angle change value has a positive reading.
                else:  # Getting the angle value for the first time
                    relative_angle = 0
                self.guidewire_current_angle = self.g_tMaster_ActualRotCountOfGw1  # [Guide Wire] Current Absolute Angle
                self.guidewire_control_msg.append([0, relative_angle * self.guidewire_angle_factor ])  # [Guide Wire] Current Relative Angle
                print("[Guidewire rotation]___ Absolute angle:{} ".format(self.guidewire_current_angle), "__Speed:{}".format(self.g_tMaster_ActualRotSpeedOfGw1))

            elif self.canMessage_RTR == 0x03:  # catheter push
                self.g_tMaster_ActualAdvCountOfCath = (self.canMessage_Data[0]) + (self.canMessage_Data[1] << 8) \
                                                      + (self.canMessage_Data[2] << 16) + (self.canMessage_Data[3] << 24)
                self.g_tMaster_ActualAdvSpeedOfCath = (self.canMessage_Data[4]) + (self.canMessage_Data[5] << 8) \
                                                      + (self.canMessage_Data[6] << 16) + (self.canMessage_Data[7] << 24)

                self.catheter_current_position = self.g_tMaster_ActualAdvCountOfCath  # [conduit] Current absolute displacement
                relative_position = self.catheter_current_position - self.catheter_zero_position  # Relative displacement
                if relative_position >= 0:  # Greater than or equal to 0
                    self.catheter_control_msg.append([relative_position * self.catheter_position_factor, 0])  # Relative displacements are all positive.
                else:
                    pass
                    # self.catheter_control_msg.append([0, 0])  # 【导管】 相对位移
                print("[conduit push] __ absolute displacement:{} ".format(self.catheter_current_position), "__Speed:{} ".format(self.g_tMaster_ActualAdvSpeedOfCath))

            elif self.canMessage_RTR == 0x04:  # Catheter rotation
                self.g_tMaster_ActualRotCountOfCath = (self.canMessage_Data[0]) + (self.canMessage_Data[1] << 8) \
                                                      + (self.canMessage_Data[2] << 16) + (self.canMessage_Data[3] << 24)
                self.g_tMaster_ActualRotSpeedOfCath = (self.canMessage_Data[4]) + (self.canMessage_Data[5] << 8) \
                                                      + (self.canMessage_Data[6] << 16) + (self.canMessage_Data[7] << 24)

                if self.catheter_current_angle != 0:
                    relative_angle = self.g_tMaster_ActualRotCountOfCath - self.catheter_current_angle  # [Conduit] Angle change value has a positive reading.
                else:  # Getting the angle value for the first time
                    relative_angle = 0
                self.catheter_current_angle = self.g_tMaster_ActualRotCountOfCath  # [Ducting] Current absolute angle
                self.guidewire_control_msg.append([0, relative_angle * self.catheter_angle_factor])  # [Ducting] Current relative angle

                print("[Catheter rotation]__ Angle:{} ".format(self.catheter_current_angle), "___Speed:{}".format(self.g_tMaster_ActualRotSpeedOfCath))

        elif self.canMessage_StdId == 0xD2:  # Position, speed, current, torque feedback S2M_RPL_ALLDATA
            self.recPosNow = (self.canMessage_Data[2]) + (self.canMessage_Data[3] << 8) \
                             + (self.canMessage_Data[4] << 16) + (self.canMessage_Data[5] << 24)
            self.recVelNow = (self.canMessage_Data[6]) + (self.canMessage_Data[7] << 8) \
                             + (self.canMessage_Data[8] << 16) + (self.canMessage_Data[9] << 24)
            self.recCurrent = (self.canMessage_Data[10]) + (self.canMessage_Data[11] << 8)
            self.recTorque = (self.canMessage_Data[12]) + (self.canMessage_Data[13] << 8)
            print("[Class-wide data]___ Displacement:{} ".format(self.recPosNow),"__Speed:{} ".format(self.recVelNow),"___ Current:{} ".format(self.recCurrent), "__Torque:{} ".format(self.recTorque))

    def ParseByteFromNet(self, data):
        if ((data == self.FRAME_HEAD) and (self.LastByte == self.FRAME_HEAD)) or (self.RevOffset > self.MaxPackageSize):
            if (self.RevOffset < 21 and self.RevOffset > 0):
                self.g_LostPackage_Net = self.g_LostPackage_Net + 1
            self.RevOffset = 0
            self.BeginFlag = self.PROTOCOL_TRUE
            self.LastByte = data
            return self.PROTOCOL_FALSE
        if (data == self.FRAME_TAIL) and (self.LastByte == self.FRAME_TAIL) and self.BeginFlag:
            self.RevOffset = self.RevOffset - 1
            self.g_ValidDataCount_Net = self.RevOffset - 1
            self.RecCheckSum = self.RecCheckSum - self.FRAME_TAIL
            self.RecCheckSum = self.RecCheckSum - self.g_RxBuf_Net[self.g_ValidDataCount_Net]
            self.LastByte = data
            self.BeginFlag = self.PROTOCOL_FALSE
            if self.RecCheckSum == self.g_RxBuf_Net[self.g_ValidDataCount_Net]:
                self.RecCheckSum = 0
                return self.PROTOCOL_TRUE
            self.g_LostPackage_Net = self.g_LostPackage_Net + 1
            self.RecCheckSum = 0
            return self.PROTOCOL_FALSE
        self.LastByte = data
        if self.BeginFlag:
            if self.CtrlFlag:
                self.RevOffset = self.RevOffset + 1
                self.g_RxBuf_Net[self.RevOffset] = data
                self.RecCheckSum = self.RecCheckSum + data
                self.CtrlFlag = self.PROTOCOL_FALSE
                self.LastByte = self.FRAME_CTRL

            elif data == self.FRAME_CTRL:
                self.CtrlFlag = self.PROTOCOL_TRUE
            else:
                self.RevOffset = self.RevOffset + 1
                self.g_RxBuf_Net[self.RevOffset] = data
                self.RecCheckSum = self.RecCheckSum + data
        return self.PROTOCOL_FALSE

    def cycleSend_operate(self):
        tickStart = int(time.time() * 1000)
        if self.handshake_flag and self.operateCount > 0:
            main.tcp_send(self.operate)  # Reply to OPERATE
            self.operateCount = self.operateCount - 1
            timer = threading.Timer(0.5, self.cycleSend_operate)
            timer.start()

        elif self.operateCount == 0:
            self.operateTimer.cancel()
            self.delay_killOperate()
            self.connectSuccess_flag = -1
            ui.pushButton_5.setText("HS Failure")
        else:
            self.operateTimer.cancel()
            self.delay_killOperate()

    def delay_killOperate(self):  # Used to send a postoperational unresponsive handler
        #self.operateTimer.cancel()
        self.handshake_flag = False
        self.operateCount = 10

    def zero_init(self):  # Set the initial position of the [guidewire] [catheter].
        self.guidewire_zero_position = socket2robot.ActualAdvCountOfGw
        self.catheter_zero_position = socket2robot.ActualAdvCountOfGw
        self.guidewire_control_msg.clear()  # 清空【导丝】控制信号list
        self.catheter_control_msg.clear()  # 清空【导管】控制信号list
        self.blind_current_step = 0


    def vessels_skeleton(self, linecolor="white", linewidth=3):  # 3D vascular skeletonization
        # mesh = trimesh.Trimesh(vertices=self.points, faces=self.edge, process=False)
        mesh = trimesh.load_mesh(self.modelFilePath, encoding="GB2312")
        # trimesh.Scene(mesh).show()
        # fixed = sk.pre.fix_mesh(mesh, remove_disconnected=5, inplace=False)
        # skel = sk.skeletonize.by_wavefront(fixed, waves=1, step_size=1)
        skel = sk.skeletonize.by_wavefront(mesh, waves=1, step_size=2)
        # skel.mesh_map
        # skel.swc.head()
        # skel.show(mesh=True)

        skeleton_map = []
        skeleton_map = skel.vertices
        print("Number of mesh skeletonized line segments:", len(skeleton_map))

        self.skeleton_point = skeleton_map

        skenode_order = skel.edges
        self.skeleton_edge = skel.edges
        # print(skenode_order)
        skenode_order.tolist()

        res = self.debranch_filt(skenode_order, switch=0)  # switch is the debranching function switch
        # print(res)
        for x in range(int(len(res))):
            act_temp = self.VTK_linedrew(skeleton_map[res[x][0]], skeleton_map[res[x][1]], color=linecolor, width=linewidth)
            self.ren.AddActor(act_temp)  # Centerline display

        self.iren.Render()

        # Interactive styles
        style = interactor()
        style.SetDefaultRenderer(self.ren)
        self.iren.SetInteractorStyle(style)

        # Mouse picking space points
        picker = vtk.vtkPointPicker()
        picker.SetTolerance(0.005)
        self.iren.SetPicker(picker)
        print("\033[1;30mPlease begin point selection.！\033[0m")
        # print(skeleton_map) # Show coordinates of 3D points on the centerline
        self.text_2d_actor.SetInput("Select StartPoint")
        self.ren.AddActor(self.text_2d_actor)

        self.actor_vascular.GetProperty().SetColor(1, 1, 1)  # Color switching when selecting points
        self.iren.Initialize()

    def debranch_filt(self, inputData, switch=1):  # centerline debranching switch for debranching switch:0 off 1 on

        inputData_list = []
        outputData_list = []
        for x in range(len(inputData)):
            inputData_list.append(inputData[x][0])
            inputData_list.append(inputData[x][1])
        for x in range(len(inputData_list)):
            try:
                if inputData_list.count(inputData_list[x]) < 2 and switch == 1:
                    if x % 2 == 0:
                        del inputData_list[x]
                        del inputData_list[x]
                        x = 0
                    else:
                        del inputData_list[x - 1]
                        del inputData_list[x - 1]
                        x = 0
            except:
                pass
        for x in range(len(inputData_list)):
            if x % 2 == 0:
                outputData_list.append([inputData_list[x], inputData_list[x + 1]])

        return outputData_list

    def branch_cal(self, linecolor="white", linewidth=3):  # Find the centerline branching point and index
        # mesh = trimesh.Trimesh(vertices=self.points, faces=self.edge, process=False)
        mesh = trimesh.load_mesh("CS3P_and_DPAC_codes/Model-data/238.stl", encoding="GB2312")
        # trimesh.Scene(mesh).show()
        # fixed = sk.pre.fix_mesh(mesh, remove_disconnected=5, inplace=False)
        # skel = sk.skeletonize.by_wavefront(fixed, waves=1, step_size=1)
        skel = sk.skeletonize.by_wavefront(mesh, waves=1, step_size=2)
        # skel.mesh_map
        # skel.swc.head()
        # skel.show(mesh=True)

        skeleton_map = []
        skeleton_map = skel.vertices
        print("Number of mesh skeletonized line segments:", len(skeleton_map))

        self.skeleton_point = skeleton_map

        skenode_order = skel.edges
        self.skeleton_edge = skel.edges
        # print(skenode_order)
        skenode_order.tolist()


        ske_p_array = np.load(centerline_path, allow_pickle=True)

        for i in self.skeleton_point:
            i[0] = round(i[0])
            i[1] = round(i[1])
            i[2] = round(i[2])

        path_startpoint = list(ske_p_array[0])
        path_endpoint = list(ske_p_array[len(ske_p_array) - 1])

        firstpoint_index = self.skeleton_point.tolist().index([round(path_startpoint[0]), round(path_startpoint[1]), round(path_startpoint[2])])
        endpoint_index = self.skeleton_point.tolist().index([round(path_endpoint[0]), round(path_endpoint[1]), round(path_endpoint[2])])

        SP = SKE_Plan(firstpoint_index, endpoint_index)
        path = SP.planning()
        self.ske_path_point = path
        # print(path)

        # Branch Acquisition
        branch_index_list = []
        branch_index = []
        for i in range(len(path)):
            for j in range(len(skenode_order)):
                if path[i] == skenode_order[j][0] and skenode_order[j][1] not in path:
                    branch_index_list.append([skenode_order[j][1], path[i]])
                    branch_index.append(skenode_order[j][1])

                if path[i] == skenode_order[j][1] and skenode_order[j][0] not in path:
                    branch_index_list.append([skenode_order[j][0], path[i]])
                    branch_index.append(skenode_order[j][0])


        # Branching out
        for i in range(len(branch_index)):
            for j in range(len(skenode_order)):
                if branch_index[i] == skenode_order[j][0] and skenode_order[j][1] not in path:
                    branch_index_list.append([skenode_order[j][1], branch_index[i]])
                    branch_index.append(skenode_order[j][1])

                if branch_index[i] == skenode_order[j][1] and skenode_order[j][0] not in path:
                    branch_index_list.append([skenode_order[j][0], branch_index[i]])
                    branch_index.append(skenode_order[j][0])



        branch_point = []
        for i in branch_index:
            branch_point.append(skeleton_map[i])

        # print(branch_point)

        filt_len = 4  # Anything less than that is filtered out
        for x in range(len(branch_index_list)):
            if self.points_distance(skeleton_map[branch_index_list[x][0]], skeleton_map[branch_index_list[x][1]]) > filt_len:  # Filter out short branches
                # act_temp = self.VTK_linedrew(skeleton_map[branch_index_list[x][0]], skeleton_map[branch_index_list[x][1]], color=linecolor, width=linewidth)
                # self.ren.AddActor(act_temp)
                pass

        self.iren.Render()

        # Interactive styles
        style = interactor()
        style.SetDefaultRenderer(self.ren)
        self.iren.SetInteractorStyle(style)

        self.iren.Initialize()

        return branch_point

    def mark_display(self, file_path, color_input='pink', opacity=0.5):  # Branch marking
        Reader = vtk.vtkSTLReader()  # stl file
        Reader.SetFileName(file_path)
        Reader.Update()
        vascular_polydata = Reader.GetOutput()
        mapper = vtk.vtkOpenGLPolyDataMapper()
        mapper.SetInputConnection(Reader.GetOutputPort())
        actor_vascular = vtk.vtkOpenGLActor()
        actor_vascular.SetMapper(mapper)  # Set the Mapper that generates the geometry elements. i.e. connect an Actor to the end of the visualization pipeline (the end of the visualization pipeline is the Mapper)
        self.ren.AddActor(actor_vascular)  # Insert actor
        self.new_ren.AddActor(actor_vascular)  # Insert actor
        colors = vtk.vtkNamedColors()
        actor_vascular.GetProperty().SetColor(colors.GetColor3d(color_input))
        actor_vascular.GetProperty().SetOpacity(opacity)  # Percent opacity (range: 0-1)
        self.ren.SetUseDepthPeeling(0)  # Select to use depth stripping (if supported) (initial value 0 (false))
        self.new_ren.SetUseDepthPeeling(0)  # Select to use depth stripping (if supported) (initial value 0 (false))
        self.ren.SetOcclusionRatio(1)  # Set the masking ratio (initial value 0.0, exact image).
        self.new_ren.SetOcclusionRatio(1)  # Set the masking ratio (initial value 0.0, exact image).
        self.ren.SetMaximumNumberOfPeels(100)  # Set the maximum number of rendering channels (initial value is 4)
        self.new_ren.SetMaximumNumberOfPeels(100)  # Set the maximum number of rendering channels (initial value is 4)
        self.vtkWidget.GetRenderWindow().SetMultiSamples(0)  # Force not selected with multisample buffer (because initial value is 8)
        self.vtkWidget2.GetRenderWindow().SetMultiSamples(0)  # Force not selected with multisample buffer (because initial value is 8)
        self.vtkWidget.GetRenderWindow().SetAlphaBitPlanes(1)  # Use rendering window with Alpha bit (initial value 0 (false)).
        self.vtkWidget2.GetRenderWindow().SetAlphaBitPlanes(1)  # Use rendering window with Alpha bit (initial value 0 (false)).



    def model_display(self):  # 3D modeling program
        # Display vtk version number
        # print(vtkmodules.vtkCommonCore.vtkVersion.GetVTKSourceVersion())

        # Setting the stage
        self.centralwidget = QtWidgets.QWidget()
        self.vtkWidget = MyQVTKRenderWindowInteractor(self.centralwidget)  # Setting up the vtk Window Interactor

        # Create vtk timer
        vtkTimer_cb = vtkTimerCallback()
        self.vtkWidget.AddObserver('TimerEvent', vtkTimer_cb.execute)
        timeid = self.vtkWidget.CreateRepeatingTimer(100)  # 100ms

        # Shut down the vtk timer
        # self.vtkWidget.KillTimer()

        # Insert VTK into pyqt
        ui.verticalLayout_3.addWidget(self.vtkWidget)  # Insert vtk form
        self.ren = vtk.vtkRenderer()  # Setting up the renderer
        # self.ren = vtk.vtkOpenGLRenderer()
        self.vtkWidget.GetRenderWindow().AddRenderer(self.ren)
        self.iren = self.vtkWidget.GetRenderWindow().GetInteractor()

        # Interactor style, in which the user controls the camera to rotate, zoom in, zoom out, and so on.
        style = vtk.vtkInteractorStyleTrackballCamera()
        self.iren.SetInteractorStyle(style)

        # Set the background color
        # self.ren.SetBackground(0, 0, 0)
        self.ren.SetBackground2(0.0, 0.0, 0.0)
        self.ren.SetBackground(0.0, 0.0, 0.0)
        self.ren.SetGradientBackground(1)

        # Read the model file
        print("\033[1;32m_Loading Models_\033[0m")
        if self.modelFilePath.endswith(".STL") or self.modelFilePath.endswith(".stl"):
            self.Reader = vtk.vtkSTLReader()  # stl file
            self.Reader.SetFileName(self.modelFilePath)
            self.Reader.Update()

        elif self.modelFilePath.endswith(".OBJ") or self.modelFilePath.endswith(".obj"):
            self.Reader = vtk.vtkOBJReader()  # obj file
            self.Reader.SetFileName(self.modelFilePath)
            self.Reader.Update()
            # self.obj2list(self.modelFilePath)


        self.vascular_polydata = self.Reader.GetOutput()  # Vascular model poly data
        # self.Reader.Update()

        closed_flag = ["Unclosed”, ”Closed"]
        checkInside = vtkmodules.vtkFiltersModeling.vtkSelectEnclosedPoints()
        print("Model file name: {} Closed：{}".format(self.modelFilePath, closed_flag[checkInside.IsSurfaceClosed(self.vascular_polydata)]))

        # Model Size Measurement
        self.modelSize_Measurement()

        # Creating a mapper
        # mapper = vtk.vtkPolyDataMapper()  # Rendering polygonal geometry data
        mapper = vtk.vtkOpenGLPolyDataMapper()
        mapper.SetInputConnection(self.Reader.GetOutputPort())
        # The input data interface of the VTK visualization pipeline , and the corresponding interface for the output data of the visualization pipeline is GetOutputPort();
        # mapper.SetInputConnection(objReader.GetOutputPort())

        # Create actor
        # self.actor_vascular = vtk.vtkActor()
        self.actor_vascular = vtk.vtkOpenGLActor()
        self.actor_vascular.SetMapper(mapper)  # Set the Mapper that generates the geometry elements. i.e. connect an Actor to the end of the visualization pipeline (the end of the visualization pipeline is the Mapper)
        self.ren.AddActor(self.actor_vascular)  # Insert actor

        # Setting the blood vessel color Veins Arteries
        self.actor_vascular.GetProperty().SetColor(1, 0, 0)

        # Opacity Nvidia
        self.actor_vascular.GetProperty().SetOpacity(0.3)  # Opacity percentage (range: 0-1)
        self.ren.SetUseDepthPeeling(0)  # Select to use depth stripping (if supported) (initial value is 0 (false))
        self.ren.SetOcclusionRatio(0.1)  # To set the masking ratio (initial value 0.0, exact image).
        self.ren.SetMaximumNumberOfPeels(100)  # Set the maximum number of rendering channels (initial value is 4)
        self.vtkWidget.GetRenderWindow().SetMultiSamples(0)  # Forced not to select buffer with multisample (because initial value is 8)
        self.vtkWidget.GetRenderWindow().SetAlphaBitPlanes(1)  # Use a rendering window with an Alpha bit (initial value 0 (false)).

        # revolve
        transform = vtk.vtkTransform()
        transform.RotateWXYZ(0, 0, 0, 0)  # perspectives
        transformFilter = vtk.vtkTransformPolyDataFilter()
        transformFilter.SetTransform(transform)
        transformFilter.SetInputConnection(self.Reader.GetOutputPort())
        transformFilter.Update()
        mapper.SetInputConnection(transformFilter.GetOutputPort())


        # Threads in the veins
        points = vtk.vtkPoints()
        # mult op 0.01
        self.pointlist = [0,0,0]

        self.dyn_line_point = vtk.vtkPoints()
        self.dyn_line_point.InsertNextPoint(self.pointlist)
        self.dyn_line_point.InsertNextPoint(self.pointlist)

        self.dyn_lineSource = vtk.vtkLineSource()
        self.dyn_lineSource.SetPoints(self.dyn_line_point)
        self.dyn_lineSource.Update()
        # mapper_line = vtk.vtkPolyDataMapper()
        dyn_mapper_line = vtk.vtkOpenGLPolyDataMapper()
        dyn_mapper_line.SetInputConnection(self.dyn_lineSource.GetOutputPort())
        # actor_line = vtk.vtkActor()
        self.dyn_actor_line = vtk.vtkOpenGLActor()
        self.dyn_actor_line.SetMapper(dyn_mapper_line)
        colors = vtk.vtkNamedColors()
        self.dyn_actor_line.GetProperty().SetColor(colors.GetColor3d("Yellow"))  # Color of the line
        self.dyn_actor_line.GetProperty().SetLineWidth(3)  # Line width
        self.ren.AddActor(self.dyn_actor_line)


        # 2Dtext display
        self.text_2d_actor = vtk.vtkTextActor()
        self.text_2d_actor.SetTextScaleModeToProp()
        self.text_2d_actor.SetDisplayPosition(0, 0)  # placement
        self.text_2d_actor.SetInput("Select StartPoint")
        self.text_2d_actor.GetActualPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
        # self.text_2d_actor.GetPosition2Coordinate().SetValue(0.6, 0.1)

        self.text_2d_actor.GetTextProperty().SetFontSize(1)
        self.text_2d_actor.GetTextProperty().SetFontFamilyToCourier()
        self.text_2d_actor.GetTextProperty().SetJustificationToCentered()
        self.text_2d_actor.GetTextProperty().BoldOff()
        self.text_2d_actor.GetTextProperty().ItalicOff()
        self.text_2d_actor.GetTextProperty().ShadowOff()
        self.text_2d_actor.GetTextProperty().SetColor(1, 1, 0)

        # self.ren.AddActor(self.text_2d_actor)



        self.force_2D_actor = vtk.vtkTextActor()
        self.force_2D_actor.SetTextScaleModeToProp()
        self.force_2D_actor.SetDisplayPosition(512, 512)
        self.force_2D_actor.SetInput("Select StartPoint")
        self.force_2D_actor.GetActualPosition2Coordinate().SetCoordinateSystemToNormalizedViewport()
        # self.force_2D_actor.GetPosition2Coordinate().SetValue(0.6, 0.1)

        self.force_2D_actor.GetTextProperty().SetFontSize(1)
        self.force_2D_actor.GetTextProperty().SetFontFamilyToCourier()
        self.force_2D_actor.GetTextProperty().SetJustificationToCentered()
        self.force_2D_actor.GetTextProperty().BoldOff()
        self.force_2D_actor.GetTextProperty().ItalicOff()
        self.force_2D_actor.GetTextProperty().ShadowOff()
        self.force_2D_actor.GetTextProperty().SetColor(1, 0, 0)


        # Interactive styles
        style = interactor()
        style.SetDefaultRenderer(self.ren)
        self.iren.SetInteractorStyle(style)

        # Mouse picking space points
        picker = vtk.vtkPointPicker()
        picker.SetTolerance(0.005)
        self.iren.SetPicker(picker)

        # keys_pressed = pygame.key.get_pressed()
        #print(keys_pressed)



        # Create a new QVTKRenderWindowInteractor and renderer
        self.vtkWidget2 = MyQVTKRenderWindowInteractor(self.centralwidget)
        self.new_ren = vtk.vtkRenderer()
        self.vtkWidget2.GetRenderWindow().AddRenderer(self.new_ren)
        ui.verticalLayout_5.addWidget(self.vtkWidget2)

        # Create new interactor styles
        self.new_iren = self.vtkWidget2.GetRenderWindow().GetInteractor()
        new_style = vtk.vtkInteractorStyleTrackballCamera()
        self.new_iren.SetInteractorStyle(new_style)

        # create new actor_vascular (same actor)
        new_actor_vascular = vtk.vtkOpenGLActor()

        # Clean up the mesh to remove non-streaming geometry
        clean_filter = vtk.vtkCleanPolyData()
        clean_filter.SetInputConnection(self.actor_vascular.GetMapper().GetInputConnection(0, 0))
        clean_filter.Update()

        # Calculate the normal
        normals = vtk.vtkPolyDataNormals()
        normals.SetInputConnection(clean_filter.GetOutputPort())
        normals.Update()

        # Convert models to triangles
        triangle_filter = vtk.vtkTriangleFilter()
        triangle_filter.SetInputConnection(normals.GetOutputPort())
        triangle_filter.Update()

        # Convert models to triangles
        subdivision_filter = vtk.vtkLoopSubdivisionFilter()
        subdivision_filter.SetInputConnection(triangle_filter.GetOutputPort())
        subdivision_filter.SetNumberOfSubdivisions(2)
        subdivision_filter.Update()

        # Create smoothing filters
        smoother = vtk.vtkSmoothPolyDataFilter()
        smoother.SetInputConnection(subdivision_filter.GetOutputPort())
        smoother.SetNumberOfIterations(50)
        smoother.SetRelaxationFactor(0.1)
        smoother.Update()  #

        # Set the smoothed data to the new mapper
        new_mapper = vtk.vtkPolyDataMapper()
        new_mapper.SetInputConnection(smoother.GetOutputPort())
        new_actor_vascular.SetMapper(new_mapper)

        # Increase model brightness
        new_actor_vascular.GetProperty().EdgeVisibilityOn()
        new_actor_vascular.GetProperty().SetLineWidth(0.3)
        new_actor_vascular.GetProperty().SetAmbient(0.1)
        new_actor_vascular.GetProperty().SetDiffuse(1.5)
        new_actor_vascular.GetProperty().SetSpecular(0.2)
        new_actor_vascular.GetProperty().SetSpecularPower(20)
        new_actor_vascular.GetProperty().EdgeVisibilityOff()

        self.new_ren.AddActor(new_actor_vascular)

        # Set the background color of the new environment
        self.new_ren.SetBackground(0, 0, 0)

        # Render new environments
        self.vtkWidget2.GetRenderWindow().Render()

        # Set the blood vessel color Veins Arteries
        new_actor_vascular.GetProperty().SetColor(1, 0, 0)

        # Opacity
        new_actor_vascular.GetProperty().SetOpacity(1)
        self.new_ren.SetUseDepthPeeling(0)
        self.new_ren.SetOcclusionRatio(0)
        self.new_ren.SetMaximumNumberOfPeels(100)
        self.vtkWidget2.GetRenderWindow().SetMultiSamples(0)
        self.vtkWidget2.GetRenderWindow().SetAlphaBitPlanes(1)



        self.mark_display('CS3P_and_DPAC_codes/Model-data/238_cut2.stl', color_input='yellow', opacity=0.5)

        self.new_iren.Start()

        # First window initialization
        self.ren.ResetCamera()
        self.iren.Initialize()

        # Second window initialization
        self.new_ren.ResetCamera()
        self.new_iren.Initialize()

    def swap_widget(self):
        if self.swap_widget_flag:
            ui.verticalLayout_3.removeWidget(self.vtkWidget2)
            ui.verticalLayout_5.removeWidget(self.vtkWidget)
            ui.verticalLayout_5.addWidget(self.vtkWidget2)
            ui.verticalLayout_3.addWidget(self.vtkWidget)
            self.swap_widget_flag = 0
        else:
            ui.verticalLayout_3.removeWidget(self.vtkWidget)
            ui.verticalLayout_5.removeWidget(self.vtkWidget2)
            ui.verticalLayout_5.addWidget(self.vtkWidget)
            ui.verticalLayout_3.addWidget(self.vtkWidget2)
            self.swap_widget_flag = 1

    def RightButtonPressEvent(self, obj, event):
        clickPos = self.GetInteractor().GetEventPosition()  # Get 2D image points
        print("Picking pixel: ", clickPos)

    def path_display(self, path_list):
        if self.path_display_flag is False:
            lineSource = vtk.vtkLineSource()
            points = vtk.vtkPoints()
            for x in range(len(path_list)):
                points.InsertNextPoint(path_list[x])

            lineSource.SetPoints(points)
            lineSource.Update()
            # mapper_line = vtk.vtkPolyDataMapper()
            mapper_line = vtk.vtkOpenGLPolyDataMapper()
            mapper_line.SetInputConnection(lineSource.GetOutputPort())
            # actor_line = vtk.vtkActor()
            self.path_display_actor = vtk.vtkOpenGLActor()
            self.path_display_actor.SetMapper(mapper_line)
            colors = vtk.vtkNamedColors()
            self.path_display_actor.GetProperty().SetColor(colors.GetColor3d("yellow"))
            self.path_display_actor.GetProperty().SetLineWidth(3)
            self.ren.AddActor(self.path_display_actor)
            # self.ren.ResetCamera()
            self.iren.Initialize()
            self.path_display_flag = True
        else:
            self.ren.RemoveActor(self.path_display_actor)
            # self.ren.ResetCamera()
            self.iren.Initialize()
            self.path_display_flag = False

    def path_advance(self):
        if self.dyn_num < len(self.process_data[0]) - 1:
            temp_line_point = vtk.vtkPoints()
            for i in range(self.dyn_num + 2):
                temp_line_point.InsertNextPoint(self.process_data[0][i])
            self.dyn_lineSource.SetPoints(temp_line_point)
            self.dyn_lineSource.Modified()
            self.iren.Render()
            self.dyn_num = self.dyn_num + 1
        else:
            print("Arrive at the finish line")

    def path_retreat(self):
        if self.dyn_num > 1:
            temp_line_point = vtk.vtkPoints()
            for i in range(self.dyn_num):
                temp_line_point.InsertNextPoint(self.process_data[0][i])
            self.dyn_lineSource.SetPoints(temp_line_point)
            self.dyn_lineSource.Modified()
            self.iren.Render()
            self.dyn_num = self.dyn_num - 1
        else:
            temp_line_point = vtk.vtkPoints()
            temp_line_point.InsertNextPoint(self.process_data[0][0])
            temp_line_point.InsertNextPoint(self.process_data[0][0])
            self.dyn_lineSource.SetPoints(temp_line_point)
            self.dyn_lineSource.Modified()
            self.iren.Render()
            print("Retracement Starting point")

    def points_distance(self, point1, point2):  # Calculate the distance between two points Input: two points [list] Return: distance
        return math.sqrt(math.pow(point1[0]-point2[0], 2)+math.pow(point1[1]-point2[1], 2)+math.pow(point1[2]-point2[2], 2))

    def closed_detect(self, polydata):  # Polyhedron closure detection Input value: polygon model Return value: number of open edge points
        closedDetect = vtk.vtkFeatureEdges()
        closedDetect.SetInputData(self.vascular_polydata)
        closedDetect.BoundaryEdgesOn()
        closedDetect.FeatureEdgesOff()
        closedDetect.ManifoldEdgesOff()
        closedDetect.NonManifoldEdgesOff()
        closedDetect.Update()
        numberOfOpenEdges = closedDetect.GetOutput().GetNumberOfCells()
        return numberOfOpenEdges

    def collision_detect(self, point, polydata):  # Collision detection using ray method Inputs: points[list], polygon data Returns: True collision, False no collision
        if self.closed_detect(polydata) == 0:  # Test the polygon for closure
            points = vtk.vtkPoints()
            points.InsertNextPoint(point)
            pointsPolydata = vtk.vtkPolyData()
            pointsPolydata.SetPoints(points)
            checkInside = vtkmodules.vtkFiltersModeling.vtkSelectEnclosedPoints()
            checkInside.SetInputData(pointsPolydata)
            checkInside.SetSurfaceData(polydata)
            checkInside.SetTolerance(0.01)
            checkInside.Update()
            is_inside = bool(1 - checkInside.IsInside(0))
            del checkInside
            del points
            del pointsPolydata
            return is_inside
        else:
            print("Polyhedron not closed")
            main.status.showMessage('Polyhedron not closed', 1000)

    def is_touch(self, detect_point):  # Collision detection Use point-surface relationships to determine if True is a collision (deprecated)
        if self.init_touch_list != 0:
            self.touch_list.clear()
            self.init_touch_list.clear()
            for x in range(len(self.face)):
                self.init_touch_list.append(self.is_up_down(self.path_startpoint, x))
        if detect_point != self.path_startpoint:  # 不是初始点
            self.touch_list.clear()  # 清空碰撞表
            for x in range(len(self.face)):
                self.touch_list.append(self.is_up_down(detect_point, x))
                if self.touch_list[x] != self.init_touch_list[x]:
                    return True
        return False

    def is_up_down(self, point, face_num):  # Determine the up and down relationship between a point and a plane
        array_point = np.array(point)
        point_A = np.array(self.face[face_num][0])
        nor_vector = np.array(self.normal_vector[face_num])
        distance = ((array_point-point_A)*nor_vector).sum(0)/np.linalg.norm(nor_vector)
        if distance > 0:
            return True
        else:
            return False

    def VTK_linedrew(self, point1, point2, color="white", width=0.001):  # Two dots to draw a line
        points = vtk.vtkPoints()
        points.InsertNextPoint(point1)
        points.InsertNextPoint(point2)
        lineSource = vtk.vtkLineSource()
        lineSource.SetPoints(points)
        # mapper_line = vtk.vtkPolyDataMapper()
        mapper_line = vtk.vtkOpenGLPolyDataMapper()
        mapper_line.SetInputConnection(lineSource.GetOutputPort())
        # actor_line = vtk.vtkActor()
        actor_line = vtk.vtkOpenGLActor()
        actor_line.SetMapper(mapper_line)
        colors = vtk.vtkNamedColors()
        actor_line.GetProperty().SetColor(colors.GetColor3d(color))
        actor_line.GetProperty().SetLineWidth(width)
        return actor_line

    def VTK_mutilinedrew(self, pointlist, color="white", width=5):  # Multi-point line drawing
        points = vtk.vtkPoints()
        for i in range(len(pointlist)):
            points.InsertNextPoint(pointlist[i])
        lineSource = vtk.vtkLineSource()
        lineSource.SetPoints(points)
        # mapper_line = vtk.vtkPolyDataMapper()
        mapper_line = vtk.vtkOpenGLPolyDataMapper()
        mapper_line.SetInputConnection(lineSource.GetOutputPort())
        # actor_line = vtk.vtkActor()
        actor_line = vtk.vtkOpenGLActor()
        actor_line.SetMapper(mapper_line)
        colors = vtk.vtkNamedColors()
        actor_line.GetProperty().SetColor(colors.GetColor3d(color))
        actor_line.GetProperty().SetLineWidth(width)
        return actor_line


    def blindpoint_display(self, cluster_list, color="blue"):  # Display blind locator point set with duplicate judgment
        self.blindpoint_actor_list.clear()
        color_list = ["yellow", "blue", "teal", "grey", "black", "orange", "white", "pink", "purple", "syan", "tan", "green", "aqua", "silver", "violet"]
        hn_sum = []
        aver_point_list = []

        if self.current_display_pointlist != cluster_list and len(cluster_list)!= 0:
            for i in range(len(cluster_list)):
                points = vtk.vtkPoints()
                temp_array = np.array([0, 0, 0])
                for j in range(len(cluster_list[i])):
                    points.InsertNextPoint(cluster_list[i][j])
                    temp_array = temp_array + cluster_list[i][j]
                aver_point_list.append(list(temp_array / len(cluster_list[i])))
                polydata = vtk.vtkPolyData()
                polydata.SetPoints(points)
                vertex = vtk.vtkVertexGlyphFilter()
                vertex.SetInputData(polydata)
                mapper = vtk.vtkPolyDataMapper()
                mapper.SetInputConnection(vertex.GetOutputPort())
                blindpoint_actor = vtk.vtkActor()
                blindpoint_actor.SetMapper(mapper)
                blindpoint_actor.GetProperty().SetPointSize(2)
                colors = vtk.vtkNamedColors()
                try:
                    blindpoint_actor.GetProperty().SetColor(colors.GetColor3d(color_list[i]))
                except:

                    blindpoint_actor.GetProperty().SetColor(colors.GetColor3d("black"))
                self.blindpoint_actor_list.append(blindpoint_actor)

        farthest_point = []
        if len(self.skeleton_plan_point) > 0:
            cluster_sec = []
            for i in range(len(aver_point_list)):
                temp_dis = []
                temp_copy = []
                for j in range(len(self.skeleton_plan_point)):
                    temp_dis.append(self.points_distance(aver_point_list[i], self.skeleton_plan_point[j]))
                temp_copy = temp_dis.copy()
                temp_copy.sort()
                sec_index = [temp_dis.index(temp_copy[0]), temp_dis.index(temp_copy[1])]
                cluster_sec.append(max(sec_index))

            # Multiple point handlers within the same subparagraph
            max_index = max(cluster_sec)
            index_sum = []
            dis_sum = []
            temp_copy = []
            for i in range(len(cluster_sec)):
                if cluster_sec[i] == max_index:
                    index_sum.append(i)
                    dis_sum.append(self.points_distance(aver_point_list[i], self.skeleton_plan_point[max_index]))
            temp_copy = dis_sum.copy()
            temp_copy.sort()
            farthest_point = aver_point_list[index_sum[dis_sum.index(temp_copy[0])]]
            # print(farthest)

        if len(farthest_point) == 0:
            farthest_point = '-No endpoint selected-'
        else:  # Spherical target points exist

            sphereSource = vtk.vtkSphereSource()
            sphereSource.SetCenter(farthest_point)
            sphereSource.SetRadius(2)
            # Create a mapper and actor
            targetPoint_mapper = vtk.vtkPolyDataMapper()
            targetPoint_mapper.SetInputConnection(sphereSource.GetOutputPort())
            targetPoint_actor = vtk.vtkActor()
            targetPoint_actor.SetMapper(targetPoint_mapper)

            colors = vtk.vtkNamedColors()
            targetPoint_actor.GetProperty().SetColor(1, 1, 0)
            self.blindpoint_actor_list.append(targetPoint_actor)

        for i in range(len(self.blindpoint_actor_list)):
            self.ren.AddActor(self.blindpoint_actor_list[i])
        # self.ren.ResetCamera()
        self.iren.Initialize()
        self.current_display_pointlist = cluster_list.copy()
        self.flag_blindpoint_actor = True
        try:
            if self.multpoint_dis_flag:
                print("Number of clusters: {}, mean target point：{}".format(len(cluster_list), [round(x, 2) for x in farthest_point]))
        except:
            if self.multpoint_dis_flag:
                print("Cluster number: {}, mean target point:{}".format(len(cluster_list), farthest_point))

        main.Node_save.append(main.process_data[8])


        action_deal(ui.pushButton_33)

        action_deal(ui.pushButton_13)

    def smooth_camera_move(self, renderer, start_position, start_focal_point, target_position, target_focal_point,
                           duration=0.5, fps=10):

        camera = renderer.GetActiveCamera()


        num_frames = int(duration * fps)


        up_vector = [0, 0, 1]
        camera.SetViewUp(up_vector)
        camera.SetViewAngle(135)




        # Ensure that all positions and focuses are lists and are of length 3, and that the contents are of numeric type (int or float)
        if not (all(isinstance(x, (int, float)) for x in start_position) and
                all(isinstance(x, (int, float)) for x in target_position)):
            raise ValueError("Start and target positions must be lists of numbers (int or float).")

        if not (all(isinstance(x, (int, float)) for x in start_focal_point) and
                all(isinstance(x, (int, float)) for x in target_focal_point)):
            raise ValueError("Start and target focal points must be lists of numbers (int or float).")

        # Smooth transitions
        for i in range(num_frames + 1):
            t = i / num_frames

            # Calculate the position and focus of each frame
            new_position = [
                start_position[j] + t * (target_position[j] - start_position[j])
                for j in range(3)
            ]

            new_focal_point = [
                start_focal_point[j] + t * (target_focal_point[j] - start_focal_point[j])
                for j in range(3)
            ]



            camera.SetPosition(new_position)
            camera.SetFocalPoint(new_focal_point)


            renderer.GetRenderWindow().Render()


            time.sleep(1.0 / fps)

    def move_actor_towards_target(self, actor, target_position, move_distance):

        current_position = actor.GetPosition()

        direction_vector = (
            target_position[0] - current_position[0],
            target_position[1] - current_position[1],
            target_position[2] - current_position[2]
        )

        distance_to_target = vtk.vtkMath.Distance2BetweenPoints(current_position, target_position) ** 0.5

        if distance_to_target < move_distance:
            new_position = target_position
        else:

            normalized_direction = (
                direction_vector[0] / distance_to_target,
                direction_vector[1] / distance_to_target,
                direction_vector[2] / distance_to_target
            )


            new_position = (
                current_position[0] + normalized_direction[0] * move_distance,
                current_position[1] + normalized_direction[1] * move_distance,
                current_position[2] + normalized_direction[2] * move_distance
            )


        actor.SetPosition(new_position)

    def blindpoint_display_1(self, cluster_list, force_state=0, color="blue"):  # Display blind locator point set with duplicate judgment
        for i in range(len(self.blindpoint_actor_list1)):
            self.ren.RemoveActor(self.blindpoint_actor_list1[i])
            self.new_ren.RemoveActor(self.blindpoint_actor_list1[i])
        self.blindpoint_actor_list1.clear()
        color_list = ["yellow", "blue", "teal", "grey", "black", "orange", "white", "pink", "purple", "syan", "tan", "green", "aqua", "silver", "violet"]
        hn_sum = []
        aver_point_list = []
        actor_point_list = []

        farthest_point = []
        ske_p_array = np.load(centerline_path, allow_pickle=True)
        ske_p_list = list(ske_p_array)
        if len(ske_p_list) > 0:
            cluster_sec = []


            for i in range(len(cluster_list)):
                points = vtk.vtkPoints()
                temp_array = np.array([0, 0, 0])
                for j in range(len(cluster_list[i])):
                    points.InsertNextPoint(cluster_list[i][j])
                    temp_array = temp_array + cluster_list[i][j]
                aver_point_list.append(list(temp_array / len(cluster_list[i])))

                actor_point_list.append(points)

            for i in range(len(aver_point_list)):
                temp_dis = []
                temp_copy = []
                for j in range(len(ske_p_list)):
                    temp_dis.append(self.points_distance(aver_point_list[i], ske_p_list[j]))
                temp_copy = temp_dis.copy()
                temp_copy.sort()
                sec_index = [temp_dis.index(temp_copy[0]), temp_dis.index(temp_copy[1])]
                cluster_sec.append(max(sec_index))


            temp_cluster_sec = cluster_sec.copy()
            temp_cluster_sec.sort(reverse=True)
            max1_index = 0
            max2_index = 0
            max_index = max(cluster_sec)
            # print(cluster_sec)
            try:
                if temp_cluster_sec[1] != 1:
                    max1_index = cluster_sec.index(temp_cluster_sec[1])
                    # print('max1', max1_index)
                if temp_cluster_sec[2] != 1:
                    max2_index = cluster_sec.index(temp_cluster_sec[2])
                    # print('max2', max2_index)
            except:
                pass

            max_index = max(cluster_sec)
            index_sum = []
            dis_sum = []
            temp_copy = []
            for i in range(len(cluster_sec)):
                if cluster_sec[i] == max_index:
                    index_sum.append(i)
                    dis_sum.append(
                        self.points_distance(aver_point_list[i], ske_p_list[max_index]))
            temp_copy = dis_sum.copy()
            temp_copy.sort()
            farthest_point = aver_point_list[index_sum[dis_sum.index(temp_copy[0])]]
            # print(farthest)


            cen_index = np.where(np.all(np.round(ske_p_list[max_index]) == main.skeleton_point, axis=1))
            cen_num = sum(1 for sublist in self.skeleton_edge if cen_index in sublist)


            index = self.ske_path_point.index(cen_index[0])


            len_extracte = 2
            start_index = max(0, index - len_extracte)
            end_index = min(len(self.ske_path_point), index + len_extracte + 1)
            extracted_values = self.ske_path_point[start_index:end_index]

            count_cen_num = 0
            for i in extracted_values:
                cen_index = np.where(np.all(np.round(ske_p_list[self.ske_path_point.index(i)]) == main.skeleton_point, axis=1))
                cen_num = sum(1 for sublist in self.skeleton_edge if cen_index in sublist)
                count_cen_num += cen_num

            self.ske_point_num = count_cen_num / len(extracted_values)


            if force_state == 1:

                sum_cluster_list = []
                for i in cluster_list:
                    if len(i) > 200:
                        i = random.sample(i, 30)

                    temp_aver = np.mean(np.array(i), axis=0)
                    aver_dis = self.points_distance(temp_aver, farthest_point)
                    if aver_dis < 50:
                        sum_cluster_list = sum_cluster_list + i

                startt = time.time()
                # (len(sum_cluster_list))
                # new_cluster_list = a.cluster_MeanShift(sum_cluster_list, quantile=0.2)
                new_cluster_list = a.cluster_GMM(sum_cluster_list, n_components=3)

                endt = time.time()
                print(endt - startt)

                aver_point_list = []

                for i in new_cluster_list:
                    aver = np.mean(np.array(i), axis=0)
                    aver_point_list.append(aver)
                cluster_sec = []
                for i in range(len(aver_point_list)):
                    temp_dis = []
                    temp_copy = []
                    for j in range(len(ske_p_list)):
                        temp_dis.append(self.points_distance(aver_point_list[i], ske_p_list[j]))
                    temp_copy = temp_dis.copy()
                    temp_copy.sort()
                    sec_index = [temp_dis.index(temp_copy[0]), temp_dis.index(temp_copy[1])]
                    cluster_sec.append(max(sec_index))

                temp_cluster_sec = cluster_sec.copy()
                temp_cluster_sec.sort(reverse=True)
                max1_index = 0
                max2_index = 0
                max_index = max(cluster_sec)
                print(cluster_sec)
                try:
                    if temp_cluster_sec[1] != 1:
                        max1_index = cluster_sec.index(temp_cluster_sec[1])
                        # print('max1', max1_index)
                    if temp_cluster_sec[2] != 1:
                        max2_index = cluster_sec.index(temp_cluster_sec[2])
                        # print('max2', max2_index)
                except:
                    pass


                nearest_branch_point = []
                min_dis = 9999
                res = 0
                for i in self.branch_point:
                    res = self.points_distance(i, aver_point_list[max1_index])
                    if res < min_dis:
                        min_dis = res
                        nearest_branch_point = i


                min_dis = 9999
                select_branch = []
                select_pointset = []
                select_pointset_index = 0
                res = 0
                for i in aver_point_list:
                    res = self.points_distance(i, nearest_branch_point)
                    if res < min_dis:
                        min_dis = res
                        select_branch = j
                        select_pointset = i
                        select_pointset_index = [idx for idx, arr in enumerate(aver_point_list) if np.array_equal(arr, i)][0]


                if min_dis > 10:

                    pass
                else:

                    actor_point_list.clear()
                    for i in new_cluster_list:
                        points = vtk.vtkPoints()
                        for j in i:
                            points.InsertNextPoint(j)
                        actor_point_list.append(points)

                    farthest_point = select_pointset


                max_index = max(cluster_sec)
                index_sum = []
                dis_sum = []
                temp_copy = []
                for i in range(len(cluster_sec)):
                    if cluster_sec[i] == max_index:
                        index_sum.append(i)
                        dis_sum.append(
                            self.points_distance(aver_point_list[i], ske_p_list[max_index]))
                temp_copy = dis_sum.copy()
                temp_copy.sort()
                # farthest_point = aver_point_list[index_sum[dis_sum.index(temp_copy[0])]]
                # print(farthest)

            elif force_state == 2:

                pass


        self.avg_point_list.append(farthest_point)

        if len(farthest_point) == 0:
            farthest_point = '-No endpoint selected-'
        else:

            sphereSource = vtk.vtkSphereSource()
            sphereSource.SetCenter(farthest_point)
            sphereSource.SetRadius(5)
            # Create a mapper and actor
            targetPoint_mapper = vtk.vtkPolyDataMapper()
            targetPoint_mapper.SetInputConnection(sphereSource.GetOutputPort())
            targetPoint_actor = vtk.vtkActor()
            targetPoint_actor.SetMapper(targetPoint_mapper)

            colors = vtk.vtkNamedColors()
            targetPoint_actor.GetProperty().SetColor(0, 0, 0)
            # self.blindpoint_actor_list1.append(targetPoint_actor)

        select_actor_num = 0
        for i in range(len(actor_point_list)):
            polydata = vtk.vtkPolyData()
            polydata.SetPoints(actor_point_list[i])
            vertex = vtk.vtkVertexGlyphFilter()
            vertex.SetInputData(polydata)
            mapper = vtk.vtkPolyDataMapper()
            mapper.SetInputConnection(vertex.GetOutputPort())
            blindpoint_actor = vtk.vtkActor()
            blindpoint_actor.SetMapper(mapper)
            blindpoint_actor.GetProperty().SetPointSize(5)
            colors = vtk.vtkNamedColors()

            # Delivery of status judgments
            if force_state == 0:
                if i == index_sum[dis_sum.index(temp_copy[0])]:
                    select_actor_num = i
                    blindpoint_actor.GetProperty().SetColor(1, 1, 0)
                    blindpoint_actor.GetProperty().SetOpacity(0.4)
                    # if self.cath_comp_flag:
                    #     move_distance = 8.0
                    #     self.move_actor_towards_target(blindpoint_actor, self.avg_point_list[-3], move_distance)


                elif max1_index != 0 and i == max1_index:
                    blindpoint_actor.GetProperty().SetColor(1, 0.8, 0.4)
                    if self.multpoint_dis_flag:
                        blindpoint_actor.GetProperty().SetOpacity(0.0)
                    else:
                        blindpoint_actor.GetProperty().SetOpacity(0.0)

                elif max2_index != 0 and i == max2_index:
                    blindpoint_actor.GetProperty().SetColor(1, 0.6, 0.6)
                    if self.multpoint_dis_flag:
                        blindpoint_actor.GetProperty().SetOpacity(0.0)
                    else:
                        blindpoint_actor.GetProperty().SetOpacity(0.0)

                else:

                    blindpoint_actor.GetProperty().SetColor(colors.GetColor3d("grey"))
                    if self.multpoint_dis_flag:
                        blindpoint_actor.GetProperty().SetOpacity(0.0)
                    else:
                        blindpoint_actor.GetProperty().SetOpacity(0.0)


            self.blindpoint_actor_list1.append(blindpoint_actor)




        self.current_avg_pointset = farthest_point


        num = select_actor_num
        list_point = numpy_support.vtk_to_numpy(actor_point_list[num].GetData())
        mean_array = np.array(list_point).mean(axis=0)

        scaling_ratio = 0.5
        for i in range(len(list_point)):
            offset = list_point[i] - mean_array
            new_ = mean_array + offset * scaling_ratio

            cube = vtk.vtkCubeSource()
            cube.SetCenter(new_)
            cube.Update()

            cube_mapper = vtk.vtkPolyDataMapper()
            cube_mapper.SetInputData(cube.GetOutput())

            cube_actor = vtk.vtkActor()
            cube_actor.SetMapper(cube_mapper)

            cube_actor.GetProperty().SetColor(1.0, 1.0, 1.0)

            cube_actor.GetProperty().SetOpacity(0.1)
            cube_actor.GetProperty().SetLineWidth(0.5)

            self.blindpoint_actor_list1.append(cube_actor)


        for i in range(len(self.blindpoint_actor_list1)):
            self.ren.AddActor(self.blindpoint_actor_list1[i])
            self.new_ren.AddActor(self.blindpoint_actor_list1[i])
        # self.ren.ResetCamera()

        cameraposition = self.avg_point_list[-2]
        lookatpoint = self.avg_point_list[-1]


        # camera = self.new_ren.GetActiveCamera()

        self.smooth_camera_move(self.new_ren, self.current_camera_pos, cameraposition, self.current_camera_focus, lookatpoint)
        self.current_camera_pos = cameraposition
        self.current_camera_focus = lookatpoint

        if self.current_light:
            self.new_ren.RemoveLight(self.current_light)


        light = vtk.vtkLight()
        light.SetPosition(self.avg_point_list[-2])
        light.SetFocalPoint(self.avg_point_list[-1])
        light.SetColor(0, 0, 1)
        light.SetIntensity(10.0)


        self.new_ren.AddLight(light)


        self.current_light = light


        self.iren.Initialize()
        self.new_iren.Initialize()





    def cal_center_sector(self, current_point, ske_point_list):
        min_dis = 9999
        min_index = 0
        for i, element in enumerate(ske_point_list):
            current_dis = self.points_distance(current_point, element)
            if current_dis < min_dis:
                min_dis = current_dis
                min_index = i

        if min_index == 0:
            min_index = 0
        elif min_index == len(ske_point_list):
            min_index = min_index - 1
        else:
            dis1 = self.points_distance(current_point, ske_point_list[min_index-1])
            dis2 = self.points_distance(current_point, ske_point_list[min_index+1])
            if dis1 < dis2:
                min_index = min_index - 1
            else:
                pass


        return min_index

    def sphere_collision_detect(self, test_point, vascular_polydata, radius=1, tolerance=0.01):  # Sphere collision detection Inputs: point to be measured, vessel model, test radius, tolerance Outputs: collision 1 no collision 0
        startt = datetime.datetime.now()
        direction_array = np.array([[1, 0, 0], [0, 1, 0], [-1, 0, 0], [0, -1, 0], [0, 0, 1], [0, 0, -1], \
                      [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0], [1, 0, 1], [0, 1, 1], \
                      [-1, 0, -1], [0, -1, -1], [1, 0, -1], [-1, 0, 1], [0, 1, -1], [0, -1, 1], \
                      [1, 1, 1], [-1, -1, -1], [1, -1, -1], [-1, 1, -1], [-1, -1, 1], [1, 1, -1], \
                      [1, -1, 1], [-1, 1, 1]])


        temp_point = np.array(test_point)
        points = vtk.vtkPoints()
        points_array = direction_array * radius + temp_point
        for i in range(len(points_array)):
            points.InsertNextPoint(points_array[i])
        pointsPolydata = vtk.vtkPolyData()
        pointsPolydata.SetPoints(points)


        checkInside = vtkmodules.vtkFiltersModeling.vtkSelectEnclosedPoints()
        checkInside.SetInputData(pointsPolydata)  # Input the set of points to be judged
        checkInside.SetSurfaceData(vascular_polydata)
        checkInside.SetTolerance(tolerance)
        checkInside.Update()

        for i in range(len(points_array)):
            if bool(1 - checkInside.IsInside(i)):
                return 1
        return 0

    def skeleton_display(self, pointlist, linecolor, linewidth):
        self.ske_display_actor = self.VTK_mutilinedrew(pointlist, color=linecolor, width=linewidth)
        colors = vtk.vtkNamedColors()
        self.ske_display_actor.GetProperty().SetColor(colors.GetColor3d(linecolor))
        self.ske_display_actor.GetProperty().SetLineWidth(linewidth)
        self.ren.AddActor(self.ske_display_actor)
        self.ren.ResetCamera()
        self.iren.Initialize()

    def skeleton_display_new(self, pointlist, sector_avg_branch_list, linewidth=5):  # The centerline is displayed in segmented colors according to the number of nodes.
        color_list = ['teal','green','yellow','pink','red','red','red','red','red','red','red','red','red']
        try:
            for i in range(len(pointlist)):
                temp_list = []
                temp_list = [pointlist[i], pointlist[i+1]]
                sector_num = sector_avg_branch_list[i]
                self.ske_display_actor = self.VTK_mutilinedrew(temp_list, color=color_list[sector_num], width=linewidth)
                colors = vtk.vtkNamedColors()
                self.ske_display_actor.GetProperty().SetColor(colors.GetColor3d(color_list[sector_num]))
                self.ske_display_actor.GetProperty().SetLineWidth(linewidth)
                self.ren.AddActor(self.ske_display_actor)
                self.new_ren.AddActor(self.ske_display_actor)

        except:
            pass

        # self.ren.ResetCamera()
        self.iren.Initialize()
        self.new_iren.Initialize()


    def calculate_total_length_of_trajectory(self, traj_points_list):
        def euclidean_distance(point1, point2):
            return np.sqrt(sum((x - y) ** 2 for x, y in zip(point1, point2)))

        def calculate_trajectory_length(traj):
            total_length = 0.0
            for i in range(1, len(traj)):
                total_length += euclidean_distance(traj[i - 1], traj[i])
            return total_length

        total_length = calculate_trajectory_length(traj_points_list)
        return total_length


    def cath_comp_display(self, pointlist, linewidth=7, color='white'):
        color_list = ['teal','green','yellow','pink','red','red','red','red','red','red','red','red','red']
        try:
            for i in self.cath_comp_path_actor_list:
                self.ren.RemoveActor(i)
        except Exception as e:
            # print(e)
            pass

        try:

            for i in range(len(pointlist)):
                temp_list = []
                temp_list = [pointlist[i], pointlist[i+1]]
                cath_comp_path_actor = self.VTK_mutilinedrew(temp_list, color = color, width=linewidth)
                self.cath_comp_path_actor_list.append(cath_comp_path_actor)

        except Exception as e:
            # print(e)
            pass

        try:

            for i in self.cath_comp_path_actor_list:
                colors = vtk.vtkNamedColors()
                i.GetProperty().SetColor(0, 0.7, 0.9)
                i.GetProperty().SetLineWidth(linewidth)
                self.ren.AddActor(i)

        except Exception as e:
            print(e)
            pass

        # self.ren.ResetCamera()
        self.iren.Initialize()


    def interpolate_trajectory(self, points_list, max_distance=5):
        def euclidean_distance(p1, p2):

            return np.linalg.norm(np.array(p1) - np.array(p2))

        def interpolate_points(p1, p2, max_distance):

            direction_vector = np.array(p2) - np.array(p1)
            direction_vector /= np.linalg.norm(direction_vector)

            num_points = int(np.ceil(euclidean_distance(p1, p2) / max_distance))
            if num_points <= 1:
                return [p1, p2]

            step_size = euclidean_distance(p1, p2) / num_points

            interpolated_points = [p1]
            for _ in range(num_points - 1):
                new_point = np.array(interpolated_points[-1]) + step_size * direction_vector
                interpolated_points.append(new_point.tolist())

            interpolated_points.append(p2)
            return interpolated_points


        interpolated_trajectory = [points_list[0]]

        for i in range(len(points_list) - 1):
            current_point = points_list[i]
            next_point = points_list[i + 1]

            interpolated_segment = interpolate_points(current_point, next_point, max_distance)
            interpolated_trajectory.extend(interpolated_segment[1:])

        return interpolated_trajectory

    def truncate_and_interpolate_trajectory(self, points_list, target_length, max_distance=5):
        def euclidean_distance(p1, p2):

            return np.linalg.norm(np.array(p1) - np.array(p2))

        def interpolate_points(p1, p2, max_distance):

            direction_vector = np.array(p2) - np.array(p1)
            direction_vector /= np.linalg.norm(direction_vector)

            interpolated_points = [p1]

            while euclidean_distance(interpolated_points[-1], p2) > max_distance:
                new_point = np.array(interpolated_points[-1]) + max_distance * direction_vector
                interpolated_points.append(new_point.tolist())

            interpolated_points.append(p2)
            return interpolated_points


        new_trajectory = [points_list[0]]
        accumulated_distance = 0

        for i in range(1, len(points_list)):

            segment_distance = euclidean_distance(points_list[i - 1], points_list[i])


            if accumulated_distance + segment_distance <= target_length:
                new_trajectory.append(points_list[i])
                accumulated_distance += segment_distance
            else:

                remaining_distance = target_length - accumulated_distance
                interpolated_segment = interpolate_points(points_list[i - 1], points_list[i], max_distance)
                for point in interpolated_segment[1:]:
                    new_trajectory.append(point)
                    accumulated_distance += euclidean_distance(new_trajectory[-2], new_trajectory[-1])
                    if accumulated_distance >= target_length:
                        break
                break

        return new_trajectory

    def dwt_distance(self, seq1, seq2):
        def euclidean_distance(p1, p2):

            return np.linalg.norm(np.array(p1) - np.array(p2))


        n, m = len(seq1), len(seq2)
        dp = np.zeros((n, m))

        for i in range(n):
            for j in range(m):

                distance = euclidean_distance(seq1[i], seq2[j])

                if i == 0 and j == 0:
                    dp[i][j] = distance
                elif i == 0:
                    dp[i][j] = distance + dp[i][j - 1]
                elif j == 0:
                    dp[i][j] = distance + dp[i - 1][j]
                else:
                    dp[i][j] = distance + min(dp[i - 1][j], dp[i][j - 1], dp[i - 1][j - 1])

        return dp[n - 1][m - 1]

    def load_pcd_data(self, file_path):
        pcd = o3d.io.read_point_cloud(file_path)
        # print(np.asarray(pcd.points))
        colors = np.asarray(pcd.colors) * 255
        points = np.asarray(pcd.points)
        # print(points.shape, colors.shape)
        # return np.concatenate([points, colors], axis=-1)
        return np.asarray(pcd.points)


    def modelSize_Measurement(self):
        triangleFilter = vtk.vtkTriangleFilter()
        triangleFilter.SetInputData(self.vascular_polydata)
        triangleFilter.Update()
        polygonProperties = vtk.vtkMassProperties()
        polygonProperties.SetInputData(triangleFilter.GetOutput())
        polygonProperties.Update()

        area = polygonProperties.GetSurfaceArea()

        vol = polygonProperties.GetVolume()

        try:

            your_mesh = mesh.Mesh.from_file(self.modelFilePath)
            volume, cog, inertia = your_mesh.get_mass_properties()
            len_xyz = (your_mesh.max_ - your_mesh.min_)

            sizel = round(len_xyz[0], 2)
            sizew = round(len_xyz[1], 2)
            sizeh = round(len_xyz[2], 2)

            print("Number of triangular faces: {}\t Number of vertices：{}".format(len(your_mesh.points), len(your_mesh.points)*3))
            print("X length: {}\tY length: {}\tZ length：{}".format(int(sizel), int(sizew), int(sizeh)))
            print("Max_X：{}\tMax_Y：{}\tMax_Z：{}\tMin_X：{}\tMin_Y：{}\tMin_Z：{}".format(int(your_mesh.max_[0]), int(your_mesh.max_[1]), int(your_mesh.max_[2]), int(your_mesh.min_[0]), int(your_mesh.min_[1]), int(your_mesh.min_[2])))
        except:
            pass
        print("Volume: {}\t surface area：{}".format(int(vol), int(area)))


    def is_number(self, s):  # Determine if it's a number
        try:
            float(s)
            return True
        except ValueError:
            pass
        try:
            import unicodedata
            unicodedata.numeric(s)
            return True
        except (TypeError, ValueError):
            pass
        return False

    def road_distance(self, pointlist):  # Calculate the absolute distance of path points
        distance_list = []
        for x in range(len(pointlist) - 1):
            if x > 0:
                distance_list.append(distance_list[x - 1] + self.points_distance(pointlist[x],pointlist[x + 1]))
            else:
                distance_list.append(self.points_distance(pointlist[x],pointlist[x + 1]))
        return distance_list

    # NDI Data Analysis
    def Read_NDI(self):
        # print(self.NDI_data)
        # Whether the data is updated
        if self.NDI_data != self.NDI_data_old:
            self.NDI_data_processed = [["Tx"], ["Ty"], ["Tz"], ["Q0"], ["Qx"], ["Qy"], ["Qz"]]
            # Data reading Mr. Chow's original NDI all str!!!!
            self.NDI_data_processed[0] = int(float(self.NDI_data[0][0:8]))     # Tx
            self.NDI_data_processed[2] = -int(float(self.NDI_data[0][8:16]))    # Ty  Inversion Exchange
            self.NDI_data_processed[1] = int(float(self.NDI_data[0][16:24]))   # Tz  Inversion Exchange
            self.NDI_data_processed[3] = int(float(self.NDI_data[0][48:56]))   # Q0
            self.NDI_data_processed[4] = int(float(self.NDI_data[0][24:32]))   # Qx
            self.NDI_data_processed[5] = int(float(self.NDI_data[0][32:40]))   # Qy
            self.NDI_data_processed[6] = int(float(self.NDI_data[0][40:48]))   # Qz
            # print("\033[0;32mNDI:\033[0m\nTx:{} Ty:{} Tz:{}\nQ0:{} Qx:{} Qy:{} Qz:{}".format(self.NDI_data_processed[0], self.NDI_data_processed[1],
            #                                                                 self.NDI_data_processed[2], self.NDI_data_processed[3],
            #                                                                 self.NDI_data_processed[4], self.NDI_data_processed[5],
            #                                                                 self.NDI_data_processed[6]))

            # Show NDI real coordinates
            # print("\033[0;32mNDI:\033[0m\nTx:{} Ty:{} Tz:{}".format(self.NDI_data_processed[0], self.NDI_data_processed[1], self.NDI_data_processed[2]))

            self.NDI_data_old = self.NDI_data.copy()

            # Determine whether to perform ICP transformations
            if type(self.Transform_matrix) != int:
                NDI_Pos_Source = np.array([self.NDI_data_processed[0], self.NDI_data_processed[1], self.NDI_data_processed[2], 1]).reshape(4,1)
                res_Pos = np.dot(self.Transform_matrix.reshape(4, 4), NDI_Pos_Source)
                # Transformed ICP position
                self.NDI_Pos_ICP = [-res_Pos[0][0] + self.x_offset, res_Pos[1][0] + self.y_offset, -res_Pos[2][0] + self.z_offset]
                # print("NDI transformed position：{}".format(self.NDI_Pos_ICP))

        self.NDI_Read_timer = threading.Timer(0.1, self.Read_NDI)  # Updating NDI data
        self.NDI_Read_timer.start()
    # Coordinates of change after alignment
    def NDI_Pos_Display(self, NDI_PointList, PointSize=5, PointColorIndex=3):
        # 3DDisplay NDI position
        try:
            self.ren.RemoveActor(self.NDI_Pos_actor)
        except:
            pass
        NDI_Point = vtk.vtkPoints()
        NDI_Point.InsertNextPoint(NDI_PointList)
        NDI_Point_polydata = vtk.vtkPolyData()
        NDI_Point_polydata.SetPoints(NDI_Point)
        NDI_vertex = vtk.vtkVertexGlyphFilter()
        NDI_vertex.SetInputData(NDI_Point_polydata)
        NDI_mapper = vtk.vtkPolyDataMapper()
        NDI_mapper.SetInputConnection(NDI_vertex.GetOutputPort())
        self.NDI_Pos_actor = vtk.vtkActor()
        self.NDI_Pos_actor.SetMapper(NDI_mapper)
        self.NDI_Pos_actor.GetProperty().SetPointSize(PointSize)  # Change the size of the point
        colors = vtk.vtkNamedColors()
        color_list = ["yellow", "blue", "teal", "red", "black", "orange", "white", "pink", "purple", "syan", "tan",
                      "green", "aqua", "silver", "violet"]
        self.NDI_Pos_actor.GetProperty().SetColor(colors.GetColor3d(color_list[PointColorIndex]))
        self.ren.AddActor(main.NDI_Pos_actor)
        self.iren.Initialize()

class AttentionLSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(AttentionLSTMModel, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(input_size, hidden_size)
        self.attention = nn.Linear(hidden_size, 1)
        self.fc = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        attention_logits = self.attention(lstm_out)
        attention_weights = torch.softmax(attention_logits, dim=0)

        weighted_lstm_out = lstm_out * attention_weights
        out = self.fc(torch.sum(weighted_lstm_out, dim=0))
        return out


def NN_init(input_size=3,hidden_size=64,output_size=3):
    model = AttentionLSTMModel(input_size, hidden_size, output_size).to(device)
    model.load_state_dict(torch.load('./CS3P_and_DPAC_codes/AttentionLSTM_model_save.ckpt'))

    return model


def NN_predict(input, model):
    input_data = torch.tensor(input, dtype=torch.float32).unsqueeze(0).to(device)
    output = model(input_data.permute(1, 0, 2))
    predicted_label = torch.argmax(output.squeeze(), dim=0)

    return predicted_label

def str_torque_cal(input_str):  # Convert string type torque data
    # Extracting numbers
    pattern = r"(\d+)"
    matches = re.findall(pattern, input_str)
    numbers = [int(match) for match in matches]

    # Calculated results
    res = ((numbers[0] * 3600 + numbers[1] * 60 + numbers[2]) * 10 + numbers[3])


    hex_str = hex(res)


    low_bytes = hex_str[-4:]


    signed_dec = int(low_bytes, 16)
    if signed_dec >= 32768:
        signed_dec = signed_dec - 65536

    return signed_dec

class CircularSpace:
    def __init__(self, size):
        self.size = size
        self.space = [[0] * size[1] for _ in range(size[0])]
        self.index = 0
        self.is_full = False

    def write(self, item):
        self.space[self.index] = item
        self.index = (self.index + 1) % self.size[0]
        if self.index == 0:
            self.is_full = True

    def read(self):
        return [list(row) for row in self.space]

    def display(self):
        for row in self.space:
            print(row)

def kalman_filter_sliding_window(history_data, new_data, window_size=10, initial_state=0, initial_estimate_error=1, process_noise=0.1, measurement_noise=1):
    # Intercept recent historical data based on window size
    if len(history_data) >= window_size:
        history_data = history_data[-window_size:]

    # Initialization state and state estimation error
    state = initial_state
    estimate_error = initial_estimate_error

    # Perform Kalman filtering
    for data in history_data:
        # Forecasting steps
        predicted_state = state
        predicted_error = estimate_error + process_noise

        # Update steps
        kalman_gain = predicted_error / (predicted_error + measurement_noise)
        state = predicted_state + kalman_gain * (data - predicted_state)
        estimate_error = (1 - kalman_gain) * predicted_error

    # Filtering of new data
    predicted_state = state
    predicted_error = estimate_error + process_noise
    kalman_gain = predicted_error / (predicted_error + measurement_noise)
    filtered_data = predicted_state + kalman_gain * (new_data - predicted_state)

    return filtered_data

# Generate a list of speeds based on minimum and maximum speeds and maximum number of segments
def speed_list_generate(min_, max_, branch):
    if branch <= 0:
        return []

    interval = (max_ - min_) // (branch - 1)
    remainder = (max_ - min_) % (branch - 1)

    result = []
    current_value = min_

    for _ in range(branch - 1):
        result.append(current_value)
        current_value += interval

        if remainder > 0:
            current_value += 1
            remainder -= 1

    result.append(max_)
    result.reverse()  # Reverse the list of results
    return result

# vtk timer callback function to work around in-thread inability to refresh VTKs
class vtkTimerCallback():
    def __init__(self):
        self.timer_count = 0

    def execute(self, obj, event):
        # print("vtk timer")
        # main.torque_label_input = 0
        if main.current_display_pointlist != main.process_data[8]:
            if main.flag_blindpoint_actor:
                for i in range(len(main.blindpoint_actor_list)):
                    main.ren.RemoveActor(main.blindpoint_actor_list[i])
            main.blindpoint_display(main.process_data[8])
            # ui.pushButton_8.setEnabled(False)
            # ui.pushButton_8.setText("已刷新")

        if main.process_data[4] == 0 and main.process_data[5] == 0 and main.flag_Blind_Switch == 1:
            ui.pushButton_12.setStyleSheet("background-color: rgb(0,255,0)")
        elif main.flag_Blind_Switch == 1:
            ui.pushButton_12.setStyleSheet("background-color: rgb(255,255,0)")

        if type(main.NDI_Pos_ICP) != int:
            # ICP aligned, showing aligned position
            main.NDI_Pos_Display(main.NDI_Pos_ICP)
        elif len(main.NDI_data_processed) > 0:
            # No ICP alignment, shows original position
            main.NDI_Pos_Display([main.NDI_data_processed[0] + main.x_offset, main.NDI_data_processed[1] + main.z_offset, main.NDI_data_processed[2] + main.y_offset])

        # Update the absolute position
        ui.label_2.setText(str(round((socket2robot.ActualAdvCountOfGw - main.guidewire_zero_position) * main.adv_rate)))
        main.robot_dis_1 = round((socket2robot.ActualAdvCountOfGw - main.guidewire_zero_position) * main.adv_rate)

        res_dis = round((socket2robot.ActualAdvCountOfGw - main.guidewire_zero_position) * main.adv_rate)

        tor = socket2robot.ActualAdvTorqueOfGw - main.current_tor
        # tor = socket2robot.ActualAdvCurrentOfGw
        main.current_tor = socket2robot.ActualAdvTorqueOfGw

        filtered_data = 0
        # Reading torque data
        if main.torque_flag:
            # filtered_data = kalman_filter_sliding_window(main.list_torque, socket2robot.ActualAdvTorqueOfGw - main.torque_noise)
            # filtered_data = kalman_filter_sliding_window(main.list_torque,
            #                                              socket2robot.ActualAdvTorqueOfGw - main.torque_noise - main.current_tor)
            filtered_data = tor
            main.list_torque.append(filtered_data)
            main.list_res_dis.append(res_dis)


        # Robot displacement color
        if res_dis % 10 < 1:  # 0
            ui.label_2.setStyleSheet("background-color: rgb(0,255,0)")
        elif res_dis % 10 < 2:  # 1
            ui.label_2.setStyleSheet("background-color: rgb(255,255,0)")
        elif res_dis % 10 < 3:  # 2
            ui.label_2.setStyleSheet("background-color: rgb(255,125,0)")
        elif res_dis % 10 < 4:  # 3
            ui.label_2.setStyleSheet("background-color: rgb(255,55,0)")
        elif res_dis % 10 < 5:  # 4
            ui.label_2.setStyleSheet("background-color: rgb(255,0,0)")
        elif res_dis % 10 < 6:  # 5
            ui.label_2.setStyleSheet("background-color: rgb(255,0,0)")
        elif res_dis % 10 < 7:  # 6
            ui.label_2.setStyleSheet("background-color: rgb(255,0,0)")
        elif res_dis % 10 < 8:  # 7
            ui.label_2.setStyleSheet("background-color: rgb(255,55,0)")
        elif res_dis % 10 < 9:  # 8
            ui.label_2.setStyleSheet("background-color: rgb(255,125,0)")
        else:  # 9
            ui.label_2.setStyleSheet("background-color: rgb(255,255,0)")
        cur_step = round(res_dis / 10)

        # Force feedback prediction
        main.NN_input.clear()  # Neural network data input
        dis = main.robot_dis_1  # Displacement
        vel = socket2robot.ActualAdvSpeedOfGw  # Speed
        cur = socket2robot.ActualAdvCurrentOfGw  # Current
        # tor = socket2robot.ActualAdvTorqueOfGw  # Torque


        # tor = tor - main.current_tor
        # main.current_tor = tor

        # vel = DWT.DWT_filter(vel)
        # cur = DWT.DWT_filter(cur)
        # tor = DWT.DWT_filter(tor)


        main.NN_input = [main.ske_point_num, vel, filtered_data]
        # print(main.torque_label_input)
        temp_NN_input = main.NN_input.copy()
        temp_NN_input.append(main.torque_label_input)  # Input labels into historical data
        if main.torque_flag == 1:  # Open save data
            main.NN_input_save.append(temp_NN_input)  # Historical data saved

        torque_data.write(main.NN_input)  # Access to torque_data
        main.force_state = NN_predict(torque_data.read(), model)  # Input time window data into the model


        if main.force_state == 2:
            if not ui.pushButton_5.isEnabled():
                main.force_2D_actor.SetInput("lesion crossing")
                main.force_2D_actor.SetDisplayPosition(512, 512)  # Position
                main.force_2D_actor.GetTextProperty().SetColor(1, 0, 0)  # Color
                main.ren.AddActor(main.force_2D_actor)
                main.torque_delay = 3
                print("dis:",dis)
                main.rot_dir = main.rot_dir + 1
                if main.rot_dir % 10 < 5:
                    socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x02, 'PVM')
                    socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x02, 200)

                else:
                    socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x02, 'PVM')
                    socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x02, -200)


        elif main.force_state == 1:
            if not ui.pushButton_5.isEnabled():
                main.force_2D_actor.SetInput("branch entering")
                main.force_2D_actor.GetTextProperty().SetColor(1, 1, 0)  # Color
                main.ren.AddActor(main.force_2D_actor)
                main.torque_delay = 1
                main.rot_dir = main.rot_dir + 1
                if main.rot_dir < 10 and main.rot_dir >= 0:
                    socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x02, 'PVM')
                    socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x02, -200)
                else:
                    socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x02, 'PVM')
                    socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x02, 200)
                # print("进分支开始旋转")



        else:
            main.force_state = 0


        # print(main.torque_delay)

        if main.torque_delay:
            main.torque_delay = main.torque_delay - 1
        else:
            main.ren.RemoveActor(main.force_2D_actor)


        if main.force_state > 0 and main.block_pos == 0:
            main.block_pos = cur_step
        elif main.force_state > 0 and main.block_pos != 0:
            pass
        elif main.force_state == 0:
            main.block_pos = 0

        if cur_step != main.blind_current_step and cur_step > 0 and cur_step < len(main.Node_list):
            current_node = main.Node_list[cur_step - 1]
            main.blindpoint_display_1(current_node)  # Show blindly positioned points
            main.blind_current_step = cur_step
            ui.label_5.setText(str(cur_step * 10))

        current_ske_sector = 0
        max_velocity = 15
        min_velocity = 5
        speed_set = 0
        try:
            max_branch = max(main.sector_avg_branch_list)
            speed_list = speed_list_generate(min_velocity, max_velocity, max_branch)
            current_ske_sector = main.cal_center_sector(main.current_avg_pointset, main.skeleton_plan_point)
            # print(speed_list)
            # print(main.sector_avg_branch_list[current_ske_sector])
            current_speed = speed_list[main.sector_avg_branch_list[current_ske_sector]]
            speed_set = current_speed

        except:  # When there is no mean point for the first time
            speed_set = min_velocity

        # Stop at the target point
        if (main.target_dis - 1) < main.robot_dis_1 and main.auto_del_flag == 1:
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 0)
            main.auto_del_flag = 0

        # Pull back to stop at target point
        if (main.target_dis + 1) > main.robot_dis_1 and main.auto_del_flag == 2:
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 0)
            main.auto_del_flag = 0

        # Autonomous delivery Speed control
        if main.auto_del_flag == 1:
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            # print('speed_set:', speed_set)
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, speed_set)


        if main.torque_flag:
            # Data loaded
            main.torque_curve.setData(main.list_res_dis, main.list_torque, pen=main.torque_pen)  # Displacement as X-axis

            if len(main.avg_list) != 0:
                # Calculate the current torque noise value
                main.torque_noise = main.avg_list[(res_dis - main.torque_res_dis + main.phase_bias + ui.spinBox_2.value()) % len(main.avg_list)]
                # print(main.torque_noise)


            scope = 100
            if max(main.list_res_dis) > scope:
                # main.torque_curve.setPos(-1000, 0)
                # main.torque_plot.setLimits(xMin=len(main.list_torque) - scope, xMax=len(main.list_torque))
                main.torque_plot.setLimits(xMin=max(main.list_res_dis) - scope, xMax=max(main.list_res_dis))
            else:
                main.torque_curve.setPos(0, max(main.list_res_dis))

        if main.torqueinit_flag:
            temp_dis = round(socket2robot.ActualAdvCountOfGw * main.adv_rate)
            main.torqueinit_data[0].append(temp_dis)
            main.torqueinit_data[1].append(tor)

            # Record the motor position value after torque initialization
            main.torque_res_dis = res_dis
        else:
            pass


        if main.force_state == 0:
            ui.label_4.setStyleSheet("background-color: rgb(0,255,0)")
        elif main.force_state == 1:
            ui.label_4.setStyleSheet("background-color: rgb(255,255,0)")
        elif main.force_state == 2:
            ui.label_4.setStyleSheet("background-color: rgb(255,0,0)")

        if main.auto_del_flag:
            try:
                NDI_pos = [main.NDI_data_processed[0] + main.x_offset, main.NDI_data_processed[1] + main.z_offset, main.NDI_data_processed[2] + main.y_offset]
            except:

                NDI_pos = [0, 0, 0]
            SPPN_pos = main.avg_point_list[len(main.avg_point_list)- 1]
            main.data_record.append([dis, vel, cur, tor, NDI_pos, SPPN_pos])



        try:
            #SPPN_pos = main.avg_point_list[len(main.avg_point_list) - 1]
            main.data_record1.append([dis, vel, cur, tor])
        except:
            pass

    def pos_callback(self):
        # main.auto_del_flag = main.auto_del_flag + 1
        # main.voice_play = 0
        socket2robot.Control_Change()
        socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PPM')
        socket2robot.sendTargetPosition_WithFrameIdAndTime(0x01, int(main.target_dis / main.adv_rate))
        socket2robot.sendTargetPosition_WithFrameIdAndTime(0x01, int(main.target_dis / main.adv_rate))
        socket2robot.sendTargetPosition_WithFrameIdAndTime(0x01, int(main.target_dis / main.adv_rate))

        print(main.target_dis)


# Used to fix self.vtkWidget.CreateRepeatingTimer not being executed
class MyQVTKRenderWindowInteractor(QVTKRenderWindowInteractor):
   def __init__(self, *arg):
       super(MyQVTKRenderWindowInteractor, self).__init__(*arg)
       self._TimerDuration = 10 # default value

   def CreateTimer(self, obj, event):
       self._Timer.start(self._TimerDuration) # self._Timer.start(10) in orginal

   def KillTimer(self):
       super(MyQVTKRenderWindowInteractor, self).GetRenderWindow().GetInteractor().DestroyTimer()

   def CreateRepeatingTimer(self, duration):
       self._TimerDuration = duration
       super(MyQVTKRenderWindowInteractor, self).GetRenderWindow().GetInteractor().CreateRepeatingTimer(duration)
       self._TimeDuration = 10

def json_load():
    with open('./DSA_Json.json', 'r', encoding='utf8')as fp:
        json_data = json.load(fp)
    main.json_list = json_data['shapes'][0]['points']
    # print('num:', len(json_list), 'data:', json_list)

def Blind_init(inc):
    if main.flag_Blind_Switch == 0:
        # 盲定位开启
        main.flag_Blind_Switch = 1
        main.blind_timer = threading.Timer(inc, scheduler_Blind, (inc,))  # Enable timer Cycle execution scheduler_Blind
        main.blind_timer.start()
        print("\033[0;30;42mBlind Positioning ON\033[0m")
        # ui.pushButton_12.setText("Blind Positioning ON")
        ui.pushButton_12.setStyleSheet("background-color: rgb(0,255,0)")
        # vtkTimer_cb = vtkTimerCallback()
        # main.vtkWidget.AddObserver('TimerEvent', vtkTimer_cb.execute)
        # timeid = main.vtkWidget.CreateRepeatingTimer(200)  # 1s
    else:

        main.flag_Blind_Switch = 0
        main.blind_timer.cancel()
        print("\033[0;31mBlind Positioning OFF\033[0m")
        # ui.pushButton_12.setText("Blind Positioning OFF")
        ui.pushButton_12.setStyleSheet("background-color: rgb(255,0,0)")
        # main.vtkWidget.KillTimer()

def scheduler_Blind(inc):  # Blind positioning cycle execution process inc is execution cycle in seconds
    # Blind positioning [guidewire displacement] [guidewire angle] process blockage
    if main.process_data[4] == -1:
        temp_guidewire_control_msg = main.guidewire_control_msg.copy()  # Copy, prevent data from being updated during runtime #
        main.frame_guidewire_control_msg -= 1
        if temp_guidewire_control_msg[main.frame_guidewire_control_msg][0] != 0:  # Position Data
            guidewire_displacement = temp_guidewire_control_msg[main.frame_guidewire_control_msg][0]
            process_type = 1
            process_data[4] = 1  # Run flag (end of program)
            main.frame_guidewire_control_msg += 1

        if temp_guidewire_control_msg[main.frame_guidewire_control_msg][1] != 0:  # Angles Data
            guidewire_angle = temp_guidewire_control_msg[main.frame_guidewire_control_msg][1]
            process_type = 2
            process_data[5] = 1  # Run flag (end of program)
            main.frame_guidewire_control_msg += 1

        # Blind positioning program execution
        if process_type == 1:  # Position Blind Positioning
            main.Openlist = main.process_data[1]  # Openlist updates
            main.hashset = main.process_data[2]  # hashset update
            main.col_hashset = main.process_data[3]  # col_hashset
            main.blind_process = Process(target=a.Blind_Delivery, args=(main.path_startpoint, guidewire_displacement, a.cal_hn, a.gen_child_Blind, main.Openlist, main.hashset,main.col_hashset, main.process_data, main.skeleton_plan_point, main.modelFilePath,))
            main.blind_process.start()  # Blind localization.

        if process_type == 2:  # Angle Blind positioning
            # Angle Blind positioning program
            pass

        # Enable timer Periodic execution of scheduler_Blind
        main.blind_timer = threading.Timer(inc, scheduler_Blind, (inc,))
        main.blind_timer.start()
        print("_Process blocking program executed_")
        return

    # [Guidewire] Encoder has been updated
    if main.frame_guidewire_control_msg < len(main.guidewire_control_msg):
        if main.process_data[4] == 0 or main.process_data[5] == 0:  # Blind positioning [guidewire displacement] [guidewire angle] process is not activated, and displacement delivery can be calculated only after guidewire rotation is calculated.
            temp_guidewire_control_msg = main.guidewire_control_msg.copy()  # copy, preventing data updates during runtime

            process_type = 0  # Type of process currently executing 1:Position 2:Angle

            # Clear Zero data for both position and angle
            while temp_guidewire_control_msg[main.frame_guidewire_control_msg][0] == 0 and temp_guidewire_control_msg[main.frame_guidewire_control_msg][1] == 0:
                main.frame_guidewire_control_msg += 1

            # Position data reading
            guidewire_displacement = 0
            while main.frame_guidewire_control_msg < len(temp_guidewire_control_msg):
                if temp_guidewire_control_msg[main.frame_guidewire_control_msg][0] != 0:  # Location Data
                    guidewire_displacement = temp_guidewire_control_msg[main.frame_guidewire_control_msg][0]
                    main.frame_guidewire_control_msg += 1
                    process_type = 1
                else:
                    break

            # Angle data reading
            guidewire_angle = 0
            while main.frame_guidewire_control_msg < len(temp_guidewire_control_msg) and process_type == 0:
                if temp_guidewire_control_msg[main.frame_guidewire_control_msg][1] != 0:  # Perspective Data
                    guidewire_angle = temp_guidewire_control_msg[main.frame_guidewire_control_msg][1]
                    main.frame_guidewire_control_msg += 1
                    process_type = 2
                else:
                    break

            # Blind positioning program execution
            if process_type == 1:  # Position Blind Positioning
                while main.process_data[4] == 1 or main.process_data[5] == 1:  # There is a blind localization process currently executing
                    pass  # Waiting for program execution to complete
                main.Openlist = main.process_data[1]  # Openlist
                main.hashset = main.process_data[2]  # hashset
                main.col_hashset = main.process_data[3]  # col_hashset
                main.blind_process = Process(target=a.Blind_Delivery, args=(main.path_startpoint, main.path_endpoint, guidewire_displacement, a.cal_hn, a.gen_child_Blind, main.Openlist, main.hashset, main.col_hashset, main.process_data, main.skeleton_plan_point, main.modelFilePath,))
                main.blind_process.start()  # Blind localization.

            if process_type == 2:  # Angle Blind positioning
                while main.process_data[4] == 1 or main.process_data[5] == 1:  # There is a blind localization process currently executing
                    pass  # Waiting for program execution to complete
                main.Openlist = main.process_data[1]  # Openlist
                main.hashset = main.process_data[2]  # hashset
                main.col_hashset = main.process_data[3]  # col_hashset
                main.blind_process = Process(target=a.Blind_Twist, args=(main.path_startpoint, main.path_endpoint, guidewire_angle, a.cal_hn, a.gen_child_Blind, main.Openlist, main.hashset,main.col_hashset, main.process_data, main.modelFilePath,))
                main.blind_process.start()  # Blind localization.

            ui.label_5.setText(str(guidewire_displacement))




    # Enable timer Periodic execution of scheduler_Blind
    main.blind_timer = threading.Timer(inc, scheduler_Blind, (inc,))
    main.blind_timer.start()

class KeyboardThread(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)
        self.running = True

    def run(self):
        def on_key_press(event):
            if event.name == 'esc':

                self.running = False
            elif event.name == 'z':
                main.torque_label_input = 0
                # print(main.torque_label_input)

            elif event.name == 'x':
                main.torque_label_input = 1
                # print(main.torque_label_input)

            elif event.name == 'c':
                main.torque_label_input = 2
                # print(main.torque_label_input)


        keyboard.on_press(on_key_press)
        keyboard.wait('esc')

    def stop(self):
        self.running = False

def Torque_data_interpolation(list_Displacement_data, list_Torque_data):

    unique_data2, unique_indices = np.unique(list_Displacement_data, return_inverse=True)


    mean_data1 = np.zeros_like(unique_data2)
    for i, val in enumerate(unique_data2):
        list_Torque_data = np.array(list_Torque_data)

        mean_data1[i] = np.mean(list_Torque_data[np.where(unique_indices == i)])


    new_data2 = np.arange(np.min(list_Displacement_data), np.max(list_Displacement_data) + 1)


    interpolated_data1 = np.interp(new_data2, unique_data2, mean_data1)

    return new_data2, interpolated_data1


def cal_branch_num(branch_point_list, center_point_list, area_size=10):
    count_list = []
    for i in center_point_list:
        count = 0
        for j in branch_point_list:
            if main.points_distance(i, j) < area_size:
                count = count + 1
        count_list.append(count)

    return count_list

def cal_centerpath_branch(num_list):
    sector_num_list = []
    for i in range(len(num_list) - 1):
        try:
            sector_num = int(np.round((num_list[i + 1] + num_list[i]) / 2, 0))
            sector_num_list.append(sector_num)
        except:
            sector_num = int(np.round(num_list[i], 0))
            sector_num_list.append(sector_num)

    return sector_num_list

def action_deal(btn):
    if btn == ui.pushButton:  # 【MapLine】 Vascular Contour Extraction

        if main.mapDraw_flag:
            main.mapDraw_flag = False
        else:
            main.mapDraw_flag = True
    elif btn == ui.pushButton_2 :  # [Torque] Turn on torque monitoring.
        if main.torque_flag == 0:
            main.torque_flag = 1
            ui.pushButton_2.setStyleSheet("background-color: rgb(0,255,0)")

            main.workbook = openpyxl.Workbook()

            main.sheet = main.workbook.active
        else:
            main.torque_flag = 0
            time_num = filter(str.isdigit, str(datetime.datetime.now())[:-7])
            upDir = os.path.pardir
            os.chdir('torque_data_save')
            np.save('torque' + ''.join(list(time_num)), main.list_torque)
            ui.pushButton_2.setStyleSheet("background-color: rgb(255,0,0)")
            # os.chdir(upDir)


            for i, sublist in enumerate(main.NN_input_save):
                for j in range(len(sublist)):
                    value = sublist[j]
                    main.sheet.cell(row=i + 1, column=j + 1, value=value)

            os.chdir('xlsx_save')

            time_num = ''.join(filter(str.isdigit, str(datetime.datetime.now())[:-7]))
            file_name = 'NN_input_save' + time_num + '.xlsx'
            main.workbook.save(file_name)

    elif btn == ui.pushButton_3 :  # Map Road] Vascular centerline extraction

        if main.roadDraw_flag is not True:
            print("\033[1;30mVascular centerline extraction in！\033[0m")
            # main.map_skeleton(main.skeImagePath)
            main.vessels_skeleton()
        else:
            main.roadDraw_flag = False
    elif btn == ui.pushButton_4 :  # 【initial position】
        main.zero_init()  # Setting the [guide wire] [catheter] 0 position
        main.auto_del_flag = 0
        main.voice_play = 0
        print("[Guide wire] [Catheter] Zero position setting")
    elif btn == ui.pushButton_5 :  # 【Connect】
        # print("Connect_Button Pressed")
        # main.tcp_connect()
        try:
            # socket2robot.ServerStart(socket2robot.getLocalIP(), int(ui.lineEdit_2.text()))
            socket2robot.ClientStart(ui.lineEdit.text(), int(ui.lineEdit_2.text()))
        except:
            # socket2robot.ServerStart('192.168.1.22', int(ui.lineEdit_2.text()))
            socket2robot.ClientStart(ui.lineEdit.text(), int(ui.lineEdit_2.text()))
        ui.pushButton_5.setEnabled(False)
        ui.pushButton_6.setEnabled(True)

    elif btn == ui.pushButton_6:  # 【Disconnect】
        # print("Disconnect_Button Pressed")
        # main.tcp_disconnect()
        socket2robot.socketWatch.close()
        ui.pushButton_5.setEnabled(True)
        ui.pushButton_6.setEnabled(False)
        ui.pushButton_5.setText("Connect")

    elif btn == ui.pushButton_7:  # 【Data Retention】

        df = pd.DataFrame(main.data_record1)

        # Get current date and time
        current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

        excel_filename = f"manual_DataSave_{current_datetime}.xlsx"

        df.to_excel(excel_filename, index=False)

    elif btn == ui.pushButton_8:  # 【Perspective switching】
        main.swap_widget()

    elif btn == ui.pushButton_10:  # 【Path Plan】 route planning
        startt = a.datetime.datetime.now()  # A*
        print("\033[0;30;44mA* Route planning in progress (do not operate!)\033[0m")
        myNode = a.a_star(main.path_startpoint, main.path_endpoint, main.skeleton_plan_point, main.Reader)  # Main thread execution. The interface will be interrupted.
        print("")  #
        currentt = a.datetime.datetime.now()
        print("\033[1;34mPlanning completed！\033[0m")
        myNodes = a.print_result(myNode)  # Path data
        main.A_star_pathdata = a.data_process(myNodes)
        if len(main.A_star_pathdata) > 0:
            main.A_star_pathdata[len(main.A_star_pathdata) - 1] = main.path_endpoint
            print(main.A_star_pathdata)
            main.path_display(main.A_star_pathdata)
        else:

            pass
        print("规划总用时:", (currentt - startt).total_seconds(), 's')  # Print Runtime
    elif btn == ui.pushButton_11:  # 【TorqueInit】 Force sensing correction
        if main.torqueinit_flag == 0:

            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 15)
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 15)
            ui.pushButton_11.setStyleSheet("background-color: rgb(255,255,0)")
            print("Turn on torque calibration")
            main.torqueinit_flag = 1
        else:
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 0)
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 0)
            ui.pushButton_35.setStyleSheet("background-color: rgb(0,0,255)")

            main.torqueinit_data[0], main.torqueinit_data[1] = Torque_data_interpolation(main.torqueinit_data[0], main.torqueinit_data[1])

            delete_len = 20
            main.torqueinit_data[0] = main.torqueinit_data[0][delete_len:]
            main.torqueinit_data[1] = main.torqueinit_data[1][delete_len:]

            main.avg_list, main.phase_bias = torque_period.period_analyse(main.torqueinit_data)

            main.torqueinit_flag = 0

        # main.path_advance()
        # socket2robot.ActualAdvCountOfGw
        pass
    elif btn == ui.pushButton_12:  # 【Blind Positioning】
        # ui.pushButton_8.setEnabled(False)  #
        Blind_init(0.2)

    elif btn == ui.pushButton_13:  # 【advance】 Guide Wire Delivery Test
        # main.path_display(main.process_data[0])
        main.advance_test += 10
        main.guidewire_control_msg.append([main.advance_test,0])
        print("\033[1;36mGuide wire\033[0m relative\033[1;35m displacement\033[0m input\033[1;35m{}\033[0m".format(main.advance_test))
    elif btn == ui.pushButton_14:  # 【Cal Vel Diameter】 Centerline Node Vessel Diameter Measurement
        print(len(main.avg_point_list), main.avg_point_list)
        diameter_list, min_diameter = a.cal_vel_diameter(main.avg_point_list, main.vascular_polydata)  # Generate 3D skeleton point diameters list
        if min_diameter != -1:
            print("Maximum instrument diameter:", round(min_diameter, 2))
        else:
            print("too short a distance")
    elif btn == ui.pushButton_15:  # 【Cath_Comp】 Catheter compensation
        if main.cath_comp_flag:
            main.cath_comp_flag = 0
        else:
            main.cath_comp_flag = 1
            main.blind_current_step = main.blind_current_step - 2
            main.blindpoint_display_1(main.Node_list[main.blind_current_step])
            print('avg:', main.avg_point_list)
            interpolated_trajectory = main.interpolate_trajectory(main.skeleton_plan_point)
            main.cath_comp_display(interpolated_trajectory[:-7])


    elif btn == ui.pushButton_16:  # 【twist】 Torsion testing of guide wires
        main.twist_test = 10
        main.guidewire_control_msg.append([0, main.twist_test])
        print("\033[1;36mWire guide \033[0m relative \033[1;33m rotary \033[0m input\033[1;33m{}\033[0m".format(main.twist_test))
    elif btn == ui.pushButton_17:  # 【adv_2D】 button
        main.path_startpoint_2d = [179, 504]
        main.path_endpoint_2d = [139, 125]
        main.guidewire_displacement_2d += 2
        while main.process_data_2d[4] == 1 or main.process_data_2d[5] == 1:  # There is a blind localization process currently executing
            pass  #
        main.Openlist_2d = main.process_data_2d[1]  # Openlist
        main.hashset_2d = main.process_data_2d[2]  # hashset
        main.col_hashset_2d = main.process_data_2d[3]  # col_hashset
        a_2d.Blind_Delivery_2D(main.path_startpoint_2d, main.path_endpoint_2d, main.guidewire_displacement_2d, main.Openlist_2d, main.hashset_2d, main.col_hashset_2d, main.process_data_2d)
        # print(main.process_data_2d[6])
    elif btn == ui.pushButton_18:  # 【NDI Pos Record】 button
        #
        NDI_Pos = [main.NDI_data_processed[0], main.NDI_data_processed[1], main.NDI_data_processed[2]]
        #
        main.NDI_Pos_Record.append(NDI_Pos)
        print("Record NDI raw coordinate points: {}, number of{}/49：".format(main.NDI_Pos_Record[len(main.NDI_Pos_Record) - 1], len(main.NDI_Pos_Record)))

    elif btn == ui.spinBox_2:
        main.x_offset = ui.spinBox_2.value()
    elif btn == ui.spinBox_3:
        main.y_offset = ui.spinBox_3.value()
    elif btn == ui.spinBox_4:
        main.z_offset = ui.spinBox_4.value()

    elif btn == ui.pushButton_19:  # 【Matching】 button
        source_point = [[45.236, 14.883, 88.713], [-57.272, 15.101, 93.805], [37.288, 77.392, 86.538],
                        [-39.772, 45.404, 57.885], [8.91, 84.579, 91.788], [-30.275, 77.349, 88.795],
                        [2.009, 109.75, 100.973], [10.419, 141.586, 102.622], [11.838, 182.229, 95.714],
                        [35.033, 202.348, 72.984], [-22.685, 213.837, 65.811], [-17.306, 217.32, 104.826],
                        [1.459, 244, 104.331], [-17.906, 253.838, 86.131], [7.548, 251.32, 75.567],
                        [11.453, 293.308, 68.139], [26.494, 324.089, 55.51], [33.758, 376.591, 30.32],
                        [31.913, 414.925, 45.657], [14.059, 479.878, 94.016], [-35.308, 476.774, 85.009],
                        [-12.585, 456.873, 107.508], [-9.391, 419.289, 120.72], [-9.726, 381.122, 126.766],
                        [21.414, 345.257, 122.412], [-0.244, 365.308, 79.99], [-6.052, 403.757, 84.615],
                        [10.436, 437.377, 57.459], [-30.424, 475.67, 60.493], [16.715, 476.441, 69.7],
                        [22.353, 434.828, 23.51], [32.445, 381.525, 1.553], [36.276, 332.801, 11.522],
                        [37.584, 314.481, 25.376], [17.526, 279.45, 41.869], [17.149, 235.414, 51.193],
                        [2.427, 242.296, 88.165], [-13.361, 221.686, 90.444], [-19.209, 211.825, 54.789],
                        [39.648, 209.613, 39.282], [12.817, 197.622, 64.602], [11.34, 164.614, 77.104],
                        [2.978, 111.201, 79.013], [-23.242, 86.297, 73.161], [15.4, 85.895, 73.859],
                        [-36.998, 40.609, 42.132], [59.285, 56.151, 78.944], [-54.587, 9.604, 77.609],
                        [47.392, 14.879, 74.915]]
        # Serial number of sampled NDI calibration points
        select_index = [1,2,4,5,6,8,10,13,19,20,21,22,24,25]
        select_index = [1, 2, 19, 20, 21]
        temp_point_list = []
        for i in select_index:
            temp_point_list.append(source_point[i - 1])
        source_point = temp_point_list

        main.Transform_matrix = ICP_matching(main.NDI_Pos_Record, source_point)

    elif btn == ui.pushButton_20:  # ↑advance
        if ui.pushButton_20.isDown():
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 10)
            socket2robot.Control_Change()
            print("forward speed10")
        else:
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 0)
            socket2robot.Control_Change()
            print("Delivery stops")

    elif btn == ui.pushButton_21:  # ↓be back
        if ui.pushButton_21.isDown():
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, -10)
            socket2robot.Control_Change()
            print("Reverse speed10")
        else:
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 0)
            socket2robot.Control_Change()
            print("Delivery stops")

    elif btn == ui.pushButton_22:  # ←levitra
        if ui.pushButton_22.isDown():
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x02, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x02, 10)
            socket2robot.Control_Change()
            print("left-hand speed10")
        else:
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x02, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x02, 0)
            socket2robot.Control_Change()
            print("cessation")

    elif btn == ui.pushButton_23:  # →right-hand side
        if ui.pushButton_23.isDown():
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x02, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x02, -10)
            socket2robot.Control_Change()
            print("Right-hand speed10")
        else:
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x02, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x02, 0)
            socket2robot.Control_Change()
            print("cessation")

    elif btn == ui.pushButton_24:  # clamped
        socket2robot.sendGWInstall()
        # print("clamped")

    elif btn == ui.pushButton_25:  # spread
        socket2robot.sendGWRemove()
        # print("spread")

    elif btn == ui.pushButton_26:  # ↑10
        socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PPM')
        print(main.robot_dis)
        main.robot_dis = main.robot_dis + 10
        socket2robot.sendTargetPosition_WithFrameIdAndTime(0x01, int(main.robot_dis / main.adv_rate))


    elif btn == ui.pushButton_27:  # ↓10
        socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PPM')
        print(main.robot_dis)
        main.robot_dis = main.robot_dis - 10
        socket2robot.sendTargetPosition_WithFrameIdAndTime(0x01, int(main.robot_dis / main.adv_rate))


    elif btn == ui.pushButton_28:  # ←10
        socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x02, 'PPM')
        main.robot_ang = main.robot_ang + 10
        socket2robot.sendTargetPosition_WithFrameIdAndTime(0x02, int(main.robot_ang * main.rot_rate))
        print("left10")

    elif btn == ui.pushButton_29:  # →10
        socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x02, 'PPM')
        main.robot_ang = main.robot_ang - 10
        socket2robot.sendTargetPosition_WithFrameIdAndTime(0x02, int(main.robot_ang * main.rot_rate))
        print("right10")

    elif btn == ui.pushButton_9:  # 【tw_2D】 button
        pass
        main.guidewire_angle_2d += 2
        while main.process_data_2d[4] == 1 or main.process_data_2d[5] == 1:
            pass
        main.Openlist_2d = main.process_data_2d[1]  # Openlist
        main.hashset_2d = main.process_data_2d[2]  # hashset
        main.col_hashset_2d = main.process_data_2d[3]  # col_hashset
        a_2d.Blind_Twist(main.guidewire_angle_2d, main.Openlist_2d, main.process_data_2d)

    elif btn == ui.pushButton_30:  # 【save】 NDIParameter saving
        np.save("TransMatrix", main.Transform_matrix)
        list_offset = []
        list_offset.append(main.x_offset)
        list_offset.append(main.y_offset)
        list_offset.append(main.z_offset)
        np.save("OffsetValue", list_offset)
        print("NDIParameters are saved")

    elif btn == ui.pushButton_31:  # 【load】 NDIParameter reading
        main.Transform_matrix = np.load("TransMatrix.npy")
        list_offset = []
        list_offset = np.load("OffsetValue.npy")
        main.x_offset = list_offset[0]
        main.y_offset = list_offset[1]
        main.z_offset = list_offset[2]
        ui.spinBox_2.setValue(main.x_offset)
        ui.spinBox_3.setValue(main.y_offset)
        ui.spinBox_4.setValue(main.z_offset)
        print("NDIParameters read")
    elif btn == ui.pushButton_32:  # 【Blind_1cm】 Blind positioning into a point set
        try:
            current_node = main.Node_list[main.blind_current_step]
            main.blindpoint_display_1(current_node)  # Show blindly positioned points
            main.blind_current_step = main.blind_current_step + 1
            ui.label_5.setText(str(main.blind_current_step*10))

            interpolated_trajectory = main.interpolate_trajectory(main.skeleton_plan_point, max_distance=5)

            avg_dis = main.blind_current_step*10


            cut_cen = main.truncate_and_interpolate_trajectory(interpolated_trajectory, avg_dis)

            DWT_dis = main.dwt_distance(cut_cen, main.avg_point_list)

            diff_DWT_dis = DWT_dis - main.last_DWT_Dis
            main.last_DWT_Dis = DWT_dis

            # print("Diff DWT_dis:", diff_DWT_dis)

            # main.cath_comp_display(cut_cen)

        except Exception as e:
            print("Exceeds storage range！")
            print(e)

    elif btn == ui.pushButton_33:  # 【Node_save】 Blind localization save point set
        data_save = np.array(main.Node_save)
        np.save("Node_SAVE", data_save)

    elif btn == ui.pushButton_34:  # 【Map_Load】 Load Centerline Map

        ske_p_array = np.load(centerline_path, allow_pickle=True)

        main.path_startpoint = list(ske_p_array[0])
        main.path_endpoint = list(ske_p_array[len(ske_p_array)-1])
        main.skeleton_plan_point = list(ske_p_array)

        if len(main.avg_point_list) == 0:
            main.avg_point_list.append(main.path_startpoint)


        Node_array = np.load(point_file, allow_pickle=True)
        main.Node_list = list(Node_array)

        main.branch_point = main.branch_cal()
        print("branch point：", main.branch_point)

        print("center point：{}".format(main.skeleton_plan_point))

        banch_num_list = cal_branch_num(main.branch_point, main.skeleton_plan_point)
        # print(banch_num_list)
        # print(len(banch_num_list))

        main.sector_avg_branch_list = cal_centerpath_branch(banch_num_list)
        # print(main.sector_avg_branch_list)
        # print(len(main.sector_avg_branch_list))


        main.skeleton_display(list(ske_p_array), linecolor="lime", linewidth=1)
        main.skeleton_display(main.skeleton_plan_point, linecolor="lightgreen", linewidth=0.1)


    elif btn == ui.pushButton_35:
        if main.auto_del_flag == 0:

            main.auto_del_flag = 1
            main.data_record.clear()
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 15)
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 15)
            ui.pushButton_35.setStyleSheet("background-color: rgb(0,255,0)")
            socket2robot.Voice_Play(0)
            # print(main.auto_del_flag)
        else:

            main.auto_del_flag = 0
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 0)
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 0)
            socket2robot.Control_Change()  # Velocity mode switching position mode


            ui.pushButton_35.setStyleSheet("background-color: rgb(255,0,0)")
            # print(main.auto_del_flag)


            df = pd.DataFrame(main.data_record)

            column_names = ['dis', 'vel', 'cur', 'tor', 'NDI_pos', 'SPPN_pos']
            df.columns = column_names

            current_datetime = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

            excel_filename = f"AutoDelivery_DataSave_{current_datetime}.xlsx"

            df.to_excel(excel_filename, index=False)

    elif btn == ui.pushButton_42:  # PTCA GW to Ostium
        if main.auto_del_flag == 0:
            main.auto_del_flag = 1
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 15)
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 15)
            ui.pushButton_42.setStyleSheet("background-color: rgb(0,255,0)")
            socket2robot.Voice_Play(0)
            main.target_dis = 520
            # print(main.auto_del_flag)
        else:
            main.auto_del_flag = 0
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 0)
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 0)
            ui.pushButton_42.setStyleSheet("background-color: rgb(255,0,0)")
            # print(main.auto_del_flag)

    elif btn == ui.pushButton_43:  # Retr PTCA GW
        if main.auto_del_flag == 0:
            main.auto_del_flag = 2
            main.cur_instr_dis = main.robot_dis_1
            # print(main.cur_instr_dis)
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, -15)
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, -15)
            ui.pushButton_43.setStyleSheet("background-color: rgb(0,255,0)")
            socket2robot.Voice_Play(0)
            main.target_dis = 0
            # print(main.auto_del_flag)
        else:
            main.auto_del_flag = 0
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 0)
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 0)
            ui.pushButton_43.setStyleSheet("background-color: rgb(255,0,0)")
            # print(main.auto_del_flag)

    elif btn == ui.pushButton_36:  # GW to Leision
        if main.auto_del_flag == 0:
            main.auto_del_flag = 1
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 15)
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 15)
            ui.pushButton_36.setStyleSheet("background-color: rgb(0,255,0)")
            socket2robot.Voice_Play(0)
            main.target_dis = 520
            # print(main.auto_del_flag)
        else:
            main.auto_del_flag = 0
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 0)
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 0)
            ui.pushButton_36.setStyleSheet("background-color: rgb(255,0,0)")
            # print(main.auto_del_flag)
    elif btn == ui.pushButton_37:  # GW Thr Leision
        if main.auto_del_flag == 0:
            main.auto_del_flag = 1
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 15)
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 15)
            ui.pushButton_37.setStyleSheet("background-color: rgb(0,255,0)")
            socket2robot.Voice_Play(0)
            main.target_dis = 600
            # print(main.auto_del_flag)
        else:
            main.auto_del_flag = 0
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 0)
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 0)
            ui.pushButton_37.setStyleSheet("background-color: rgb(255,0,0)")
            # print(main.auto_del_flag)
    elif btn == ui.pushButton_38:  # Bal to Leision
        if main.auto_del_flag == 0:
            main.multpoint_dis_flag = 0
            main.zero_init()
            print("[Guide wire] [Catheter] Zero position setting")
            main.auto_del_flag = 1
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 15)
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 15)
            ui.pushButton_38.setStyleSheet("background-color: rgb(0,255,0)")
            socket2robot.Voice_Play(0)
            main.target_dis = 520
            # print(main.auto_del_flag)
        else:
            main.auto_del_flag = 0
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 0)
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 0)
            ui.pushButton_38.setStyleSheet("background-color: rgb(255,0,0)")
            # print(main.auto_del_flag)

    elif btn == ui.pushButton_39:  # Bal Thr Leision
        if main.auto_del_flag == 0:
            main.auto_del_flag = 1
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 15)
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 15)
            ui.pushButton_39.setStyleSheet("background-color: rgb(0,255,0)")
            socket2robot.Voice_Play(0)
            main.target_dis = 530
            # print(main.auto_del_flag)
        else:
            main.auto_del_flag = 0
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 0)
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 0)
            ui.pushButton_39.setStyleSheet("background-color: rgb(255,0,0)")
            # print(main.auto_del_flag)

    elif btn == ui.pushButton_40:  # Retr GW
        if main.auto_del_flag == 0:
            main.auto_del_flag = 2
            main.cur_instr_dis = main.robot_dis_1
            # print(main.cur_instr_dis)
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, -15)
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, -15)
            ui.pushButton_40.setStyleSheet("background-color: rgb(0,255,0)")
            socket2robot.Voice_Play(0)
            main.target_dis = 0
            # print(main.auto_del_flag)
        else:
            main.auto_del_flag = 0
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 0)
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 0)
            ui.pushButton_40.setStyleSheet("background-color: rgb(255,0,0)")
            # print(main.auto_del_flag)
    elif btn == ui.pushButton_41:  # Retr Bal
        if main.auto_del_flag == 0:
            # print(main.guidewire_zero_position)
            main.guidewire_zero_position = main.guidewire_zero_position - int(main.cur_instr_dis/main.adv_rate)
            print(main.guidewire_zero_position)
            main.auto_del_flag = 2
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, -15)
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, -15)
            ui.pushButton_41.setStyleSheet("background-color: rgb(0,255,0)")
            socket2robot.Voice_Play(0)
            main.target_dis = 0
            # print(main.auto_del_flag)
        else:
            main.auto_del_flag = 0
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 0)
            socket2robot.sendTargetModeOfOperation_WithFrameIdAndTime(0x01, 'PVM')
            socket2robot.sendTargetVelocity_WithFrameIdAndTime(0x01, 0)
            ui.pushButton_41.setStyleSheet("background-color: rgb(255,0,0)")
            # print(main.auto_del_flag)

def signal_init():  # Signal Binding Slot Functions
    ui.pushButton.clicked.connect(lambda: action_deal(ui.pushButton))
    ui.pushButton_2.clicked.connect(lambda: action_deal(ui.pushButton_2))
    ui.pushButton_3.clicked.connect(lambda: action_deal(ui.pushButton_3))
    ui.pushButton_4.clicked.connect(lambda: action_deal(ui.pushButton_4))
    ui.pushButton_5.clicked.connect(lambda: action_deal(ui.pushButton_5))
    ui.pushButton_6.clicked.connect(lambda: action_deal(ui.pushButton_6))
    ui.pushButton_7.clicked.connect(lambda: action_deal(ui.pushButton_7))
    ui.pushButton_8.clicked.connect(lambda: action_deal(ui.pushButton_8))
    ui.pushButton_9.clicked.connect(lambda: action_deal(ui.pushButton_9))
    ui.pushButton_10.clicked.connect(lambda: action_deal(ui.pushButton_10))
    ui.pushButton_11.clicked.connect(lambda: action_deal(ui.pushButton_11))
    ui.pushButton_12.clicked.connect(lambda: action_deal(ui.pushButton_12))
    ui.pushButton_13.clicked.connect(lambda: action_deal(ui.pushButton_13))
    ui.pushButton_14.clicked.connect(lambda: action_deal(ui.pushButton_14))
    ui.pushButton_15.clicked.connect(lambda: action_deal(ui.pushButton_15))
    ui.pushButton_16.clicked.connect(lambda: action_deal(ui.pushButton_16))
    ui.pushButton_17.clicked.connect(lambda: action_deal(ui.pushButton_17))
    ui.pushButton_18.clicked.connect(lambda: action_deal(ui.pushButton_18))
    ui.pushButton_19.clicked.connect(lambda: action_deal(ui.pushButton_19))
    ui.pushButton_20.pressed.connect(lambda: action_deal(ui.pushButton_20))
    ui.pushButton_20.released.connect(lambda: action_deal(ui.pushButton_20))
    ui.pushButton_21.pressed.connect(lambda: action_deal(ui.pushButton_21))
    ui.pushButton_21.released.connect(lambda: action_deal(ui.pushButton_21))
    ui.pushButton_22.pressed.connect(lambda: action_deal(ui.pushButton_22))
    ui.pushButton_22.released.connect(lambda: action_deal(ui.pushButton_22))
    ui.pushButton_23.pressed.connect(lambda: action_deal(ui.pushButton_23))
    ui.pushButton_23.released.connect(lambda: action_deal(ui.pushButton_23))
    ui.pushButton_24.clicked.connect(lambda: action_deal(ui.pushButton_24))
    ui.pushButton_25.clicked.connect(lambda: action_deal(ui.pushButton_25))
    ui.pushButton_26.clicked.connect(lambda: action_deal(ui.pushButton_26))
    ui.pushButton_27.clicked.connect(lambda: action_deal(ui.pushButton_27))
    ui.pushButton_28.clicked.connect(lambda: action_deal(ui.pushButton_28))
    ui.pushButton_29.clicked.connect(lambda: action_deal(ui.pushButton_29))
    ui.pushButton_30.clicked.connect(lambda: action_deal(ui.pushButton_30))
    ui.pushButton_31.clicked.connect(lambda: action_deal(ui.pushButton_31))
    ui.pushButton_32.clicked.connect(lambda: action_deal(ui.pushButton_32))
    ui.pushButton_33.clicked.connect(lambda: action_deal(ui.pushButton_33))
    ui.pushButton_34.clicked.connect(lambda: action_deal(ui.pushButton_34))
    ui.pushButton_35.clicked.connect(lambda: action_deal(ui.pushButton_35))
    ui.pushButton_36.clicked.connect(lambda: action_deal(ui.pushButton_36))
    ui.pushButton_37.clicked.connect(lambda: action_deal(ui.pushButton_37))
    ui.pushButton_38.clicked.connect(lambda: action_deal(ui.pushButton_38))
    ui.pushButton_39.clicked.connect(lambda: action_deal(ui.pushButton_39))
    ui.pushButton_40.clicked.connect(lambda: action_deal(ui.pushButton_40))
    ui.pushButton_41.clicked.connect(lambda: action_deal(ui.pushButton_41))
    ui.pushButton_42.clicked.connect(lambda: action_deal(ui.pushButton_42))
    ui.pushButton_43.clicked.connect(lambda: action_deal(ui.pushButton_43))
    ui.spinBox_2.valueChanged.connect(lambda: action_deal(ui.spinBox_2))
    ui.spinBox_3.valueChanged.connect(lambda: action_deal(ui.spinBox_3))
    ui.spinBox_4.valueChanged.connect(lambda: action_deal(ui.spinBox_4))
    ui.pushButton_6.setEnabled(False)
    # ui.horizontalSlider.valueChanged.connect(lambda: action_deal(ui.horizontalSlider))

def ui_init(obj):  # Interface initialization configuration function
    main.init_video_display()
    obj.pushButton_12.setStyleSheet("background-color: rgb(255,0,0)")  # Set the blind positioning button to a red background
    obj.pushButton_2.setStyleSheet("background-color: rgb(255,0,0)")
    obj.pushButton_11.setStyleSheet("background-color: rgb(255,0,0)")
    obj.pushButton_36.setStyleSheet("background-color: rgb(255,255,0)")
    obj.pushButton_37.setStyleSheet("background-color: rgb(255,255,0)")
    obj.pushButton_38.setStyleSheet("background-color: rgb(255,255,0)")
    obj.pushButton_39.setStyleSheet("background-color: rgb(255,255,0)")
    obj.pushButton_40.setStyleSheet("background-color: rgb(255,255,0)")
    obj.pushButton_41.setStyleSheet("background-color: rgb(255,255,0)")
    obj.pushButton_42.setStyleSheet("background-color: rgb(255,255,0)")
    obj.pushButton_43.setStyleSheet("background-color: rgb(255,255,0)")
    obj.spinBox_2.setValue(main.x_offset)     # X_offset
    obj.spinBox_3.setValue(main.y_offset)   # Y_offset
    obj.spinBox_4.setValue(main.z_offset)      # Z_offset
    try:
        # ui.lineEdit.setText(socket2robot.getLocalIP())
        ui.lineEdit.setText('192.168.0.22')
        ui.lineEdit_2.setText('22')
        main.robot_dis = main.robot_dis_1
    except:
        ui.lineEdit.setText('192.168.1.22')

# Point cloud matching Quantity needs to be consistent
def ICP_matching(source, target,iteration_num=200):
    # Original marker
    sourcePoints = vtk.vtkPoints()
    for i in range(len(source)):
        sourcePoints.InsertNextPoint(source[i])

    # Target marking points
    targetPoints = vtk.vtkPoints()
    for i in range(len(target)):
        targetPoints.InsertNextPoint(target[i])

    # Find the maximum and minimum XYZ values of the set of points
    sourceBounds = sourcePoints.GetBounds()  # xmin,xmax, ymin,ymax, zmin,zmax
    targetBounds = targetPoints.GetBounds()  # xmin,xmax, ymin,ymax, zmin,zmax


    source_Len = [[sourceBounds[1] - sourceBounds[0]], [sourceBounds[3] - sourceBounds[2]], [sourceBounds[5] - sourceBounds[4]]]
    target_Len = [[targetBounds[1] - targetBounds[0]], [targetBounds[3] - targetBounds[2]], [targetBounds[5] - targetBounds[4]]]

    scale = np.array(target_Len) / np.array(source_Len)

    # Original data
    source_ = vtk.vtkPolyData()
    source_.SetPoints(sourcePoints)

    # Target data
    target_ = vtk.vtkPolyData()
    target_.SetPoints(targetPoints)

    sourceGlypyFilter = vtk.vtkVertexGlyphFilter()
    sourceGlypyFilter.SetInputData(source_)
    sourceGlypyFilter.Update()

    targetGlyphFilter = vtk.vtkVertexGlyphFilter()
    targetGlyphFilter.SetInputData(target_)
    targetGlyphFilter.Update()

    # Perform ICP alignment for transformation matrix
    icpTransform = vtk.vtkIterativeClosestPointTransform()
    icpTransform.SetSource(sourceGlypyFilter.GetOutput())
    icpTransform.SetTarget(targetGlyphFilter.GetOutput())

    # ※※※※※※※※※※※※※※※※※ #
    # icpTransform.GetLandmarkTransform().SetModeToRigidBody()
    icpTransform.GetLandmarkTransform().SetModeToSimilarity()
    # ※※※※※※※※※※※※※※※※※ #

    icpTransform.SetMaximumNumberOfIterations(iteration_num)
    icpTransform.StartByMatchingCentroidsOn()
    icpTransform.Modified()


    icpTransform.GetMatrix().SetElement(0, 0, scale[0])
    icpTransform.GetMatrix().SetElement(1, 1, scale[1])
    icpTransform.GetMatrix().SetElement(2, 2, scale[2])


    icpTransform.Update()

    Transform_matrix_list = []
    for i in range(4):
        temp_list = []
        for j in range(4):
            temp_list.append(icpTransform.GetMatrix().GetElement(i, j))
        Transform_matrix_list.append(temp_list)
    Transform_matrix = np.array(Transform_matrix_list)
    print("conversion matrix：\n", Transform_matrix, Transform_matrix.shape)
    return Transform_matrix

def torque_display(bg='w',linewidth=2):

    main.keyboard_thread = KeyboardThread()
    main.keyboard_thread.start()

    main.torque_plot = pg.PlotWidget(background='k')
    main.torque_curve = main.torque_plot.plot(pen='w')
    main.torque_pen = pg.mkPen(color=(255, 255, 255), width=linewidth)
    main.torque_plot.showGrid(x=True, y=True)
    main.torque_plot.setLogMode(x=False, y=False)

    ui.verticalLayout_4.addWidget(main.torque_plot)


if __name__ == '__main__':
    print("\033[1;30;41m---VasCure blind positioning system---\033[0m")
    print("Python: {}".format(sys.version))
    print(vtkmodules.vtkCommonCore.vtkVersion.GetVTKSourceVersion())
    print("CPU Number of logical cores: {}".format(multiprocessing.cpu_count()))

    print("\033[1;32m_loading screen_\033[0m")
    app = QtWidgets.QApplication(sys.argv)
    main = MainWin()
    ui = QT_UI.Ui_MainWindow()
    ui.setupUi(main)
    ui_init(ui)

    # Create a black color palette
    palette = QtGui.QPalette()
    palette.setColor(QtGui.QPalette.Window, QtCore.Qt.black)
    palette.setColor(QtGui.QPalette.WindowText, QtCore.Qt.white)  # Set the label text color to white

    # Apply color palettes to windows
    main.setPalette(palette)

    # Start an NDI listening thread
    main.NDI_listern_thread = threading.Thread(target=WM_COPYDATA.listen_func, args=(main.NDI_data,))
    main.NDI_listern_thread.start()
    main.NDI_Read_timer = threading.Timer(0.1, main.Read_NDI)  # NDI data updated
    main.NDI_Read_timer.start()
    print("\033[1;32m_NDIListening on_\033[0m")


    main.setWindowIcon(QIcon("Vascure.ico"))


    main.setWindowTitle("A Level 2 Autonomous Surgical Robotic System for Coronary Interventional Surgery")


    main.resize(1350, 540)
    # main.setFixedSize(1350, 540)


    main.status = main.statusBar()


    # json_load()


    signal_init()


    main.model_display()


    model = NN_init()


    torque_data = CircularSpace([1, 3])


    torque_display()

    win = Process(target=main.show()).start()
    main.show()

    sys.exit(app.exec_())


