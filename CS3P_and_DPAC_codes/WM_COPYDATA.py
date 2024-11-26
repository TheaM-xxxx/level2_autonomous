# -*- coding: utf-8 -*-
# @Author   : Linsen Zhang
import multiprocessing
import threading
import time

import ctypes
import ctypes.wintypes
import win32api
import win32con
import win32gui


class COPYDATASTRUCT(ctypes.Structure):
    _fields_ = [
        ('dwData', ctypes.wintypes.LPARAM),
        ('cbData', ctypes.wintypes.DWORD),
        ('lpData', ctypes.c_void_p)
    ]

PCOPYDATASTRUCT = ctypes.POINTER(COPYDATASTRUCT)

class EndOfTime(threading.Thread):
    def __init__(self):
        threading.Thread.__init__(self)

    def run(self):
        while 1:
            # print(1)
            pass
            time.sleep(3)

class Listener:
    def __init__(self, rec_data):
        WindowName ="StartForm"
        message_map = {
            win32con.WM_COPYDATA: self.OnCopyData
        }
        wc = win32gui.WNDCLASS()
        wc.lpfnWndProc = message_map
        wc.lpszClassName = WindowName
        hinst = wc.hInstance = win32api.GetModuleHandle(None)
        classAtom = win32gui.RegisterClass(wc)

        win_name = "StartForm"  # Send data by looking up the window name or address
        self.hwnd = win32gui.CreateWindow(
            classAtom,
            win_name,
            0,
            0,
            0,
            win32con.CW_USEDEFAULT,
            win32con.CW_USEDEFAULT,
            0,
            0,
            hinst,
            None
        )
        self.data_rec = rec_data
        print("WM_COPYDATAmonitor:{}, window name:{}".format(self.hwnd, win_name))

    def OnCopyData(self, hwnd, msg, wparam, lparam):
        pCDS = ctypes.cast(lparam, PCOPYDATASTRUCT)
        rec = ctypes.string_at(pCDS.contents.lpData).decode()
        self.data_rec[0] = rec

        return 1

def listen_func(rec_data):
    l = Listener(rec_data)
    win32gui.PumpMessages()




