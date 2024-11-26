# -*- coding: utf-8 -*-
# @Author   : Linsen Zhang
import time
import paras
import socket
import threading



u32_FrameId_Tx = 0  # of frames sent
u32_FrameId_Rx = 0  # of frames received
tickStart = int(time.time() * 1000)
salve_ip = '127.0.0.1'
salve_port = 22
master_port = 1005
socketWatch = socket.socket()  # Define a socket to listen for messages from clients
socketConnect = socket.socket()  # Create a socket responsible for communicating with the client
CONST_CLIENT_NUMBER = 1  # Maximum number of clients allowed
isSlaveConnected = False

# Encapsulation format of communication packets: FrameHead + Data + CheckSum + FrameTail.
# FrameHead is two consecutive 0xAAs, FrameTail is two consecutive 0x55s.
# If Data contains 0xA5, 0xAA, 0x55 (i.e., a special character), a control character 0xA5 is added before sending that character.
# CheckSum is the 8-bit checksum, i.e., the lower eight bits of the sum of all the data in the Data
# 如0xAA 0xAA frameID data0 data1 data2 data3 CheckSum 0x55 0x55，
# 如0xAA 0xAA frameID data0 data1 data2 data3 data4 data5 data6 data7 CheckSum 0x55 0x55
NET_FRAME_HEAD = 0xAA  # Frame header
NET_FRAME_TAIL = 0x55  # End of frame
NET_FRAME_CTRL = 0xA5  # Escape
m_TotalRxLostPackages = 0  # packets lost
validNetRxBufDataCount = 0  # valid bytes received

NET_MaxPackageSize = 500  # Maximum number of bytes in a valid frame
g_RxBuf_Net = bytearray(NET_MaxPackageSize)  # Arrays of deposits of valid data

initFlag = True  # Initial reading identifier
initGwCount = 0  # Initial Count before delivery
ReceivedSlaveHandshake = False  # Has the master received the handshake command?
SlaveState = paras.MS_STAT_E['MS_STAT_RESET']

ActualAdvCountOfGw = int(0)  # Actual count of the guidewire
ActualAdvSpeedOfGw = int(0)  # Actual speed of the guidewire
ActualAdvCountOfGw1 = int(0)  # Fuse 1 delivery count
ActualAdvSpeedOfGw1 = int(0)  # Delivery speed of guidewire 1
ActualRotCountOfGw1 = int(0)  # Guide wire 1 angle of rotation count
ActualRotSpeedOfGw1 = int(0)  # Rotation speed of guidewire 1
GW_ROT_MiddlePos = -155000  # Intermediate position of rotation (absolute)
ActualAdvCountOfCath = int(0)  # catheters actually pushed: unit count
ActualAdvSpeedOfCath = int(0)  # catheters actually pushed: unit count
CATH_ADV_MiddlePos = 750000  # Push intermediate position (absolute)
ActualRotCountOfCath = int(0)  # Actual catheter rotation angle: unit count
ActualRotSpeedOfCath = int(0)  # Actual catheter rotation speed: in count
g_channel = int(1)  # Current channel

ActualAdvCurrentOfGw = int(0)
ActualAdvTorqueOfGw = int(0)  # Torque transducer range is 5k to 15k, zero point is 10k

global flag_ballon_install
flag_ballon_install = 0

# Record synchronized frame information and time delay
tSyncFrame = {'FrameId': int(-1), 'SendTime': int(-1), 'ReceiveTime': int(-1), 'Delta': int(-1), 'TimeDelay': int(-1)}


def sendTargetVelocity_WithFrameIdAndTime(node: int, targetVelocity: int):

    assert node in paras.SLAVE_EPOS_NODEID_E.values(), 'No motor corresponding to node %d exists' % node
    global u32_FrameId_Tx
    u32_FrameId_Tx += 1
    tmpNetMsg = bytearray(24)

    tmpNetMsg[0:1] = paras.MS_COM_PROTOCOL_STDID_E['M2S_CMD_VELOCITY'].to_bytes(2, 'little')
    tmpNetMsg[1] = 0xff
    tmpNetMsg[2] = 0  # RTR
    tmpNetMsg[3] = 5  # DLC
    tmpNetMsg[4] = node
    if targetVelocity < 0:
        targetVelocity += 2 ** 32
    tmpNetMsg[5:9] = targetVelocity.to_bytes(4, 'little')
    tmpNetMsg[9:12] = 0x00.to_bytes(2, 'little')
    tmpNetMsg[12:16] = 0x03.to_bytes(4, 'little')
    tmpNetMsg[16:20] = u32_FrameId_Tx.to_bytes(4, 'little')
    g_u32RunTime = int(time.time() * 1000) - tickStart
    tmpNetMsg[20:24] = g_u32RunTime.to_bytes(4, 'little')
    PackAndSendNetMessage(tmpNetMsg)
    # return tmpNetMsg


def sendTargetPosition_WithFrameIdAndTime(node: int, targetPosition: int):

    assert node in paras.SLAVE_EPOS_NODEID_E.values(), 'No motor corresponding to node %d exists' % node
    global u32_FrameId_Tx
    u32_FrameId_Tx += 1
    tmpNetMsg = bytearray(24)

    tmpNetMsg[0:1] = paras.MS_COM_PROTOCOL_STDID_E['M2S_CMD_POSITION'].to_bytes(2, 'little')
    tmpNetMsg[1] = 0xff
    tmpNetMsg[2] = 0  # RTR
    tmpNetMsg[3] = 5  # DLC
    tmpNetMsg[4] = node
    if targetPosition < 0:
        targetPosition += 2 ** 32
    tmpNetMsg[5:9] = targetPosition.to_bytes(4, 'little')
    tmpNetMsg[9:12] = 0x00.to_bytes(2, 'little')
    tmpNetMsg[12:16] = 0x03.to_bytes(4, 'little')
    tmpNetMsg[16:20] = u32_FrameId_Tx.to_bytes(4, 'little')
    g_u32RunTime = int(time.time() * 1000) - tickStart
    tmpNetMsg[20:24] = g_u32RunTime.to_bytes(4, 'little')
    PackAndSendNetMessage(tmpNetMsg)
    # return tmpNetMsg


def sendSyncFrame_WithFrameIdAndTime():

    global u32_FrameId_Tx
    u32_FrameId_Tx += 1
    tmpNetMsg = bytearray(24)
    tmpNetMsg[0:1] = paras.MS_COM_PROTOCOL_STDID_E['M2S_CMD_SyncFrame'].to_bytes(2, 'little')
    tmpNetMsg[1] = 0xff
    tmpNetMsg[2] = 0  # RTR
    tmpNetMsg[3] = 0x04  # DLC
    tmpNetMsg[4] = 0x01
    tmpNetMsg[5:12] = 0x00.to_bytes(7, 'little')
    tmpNetMsg[12:16] = 0x03.to_bytes(4, 'little')
    tmpNetMsg[16:20] = u32_FrameId_Tx.to_bytes(4, 'little')
    g_u32RunTime = int(time.time() * 1000) - tickStart
    tmpNetMsg[20:24] = g_u32RunTime.to_bytes(4, 'little')
    PackAndSendNetMessage(tmpNetMsg)
    tSyncFrame['FrameId'] = int(u32_FrameId_Tx)
    tSyncFrame['SendTime'] = int(g_u32RunTime)


def sendHeartBeatFrame_WithFrameIdAndTime():

    global u32_FrameId_Tx
    u32_FrameId_Tx += 1
    tmpNetMsg = bytearray(24)
    tmpNetMsg[0:1] = paras.MS_COM_PROTOCOL_STDID_E['M2S_CMD_HeartBeat'].to_bytes(2, 'little')
    tmpNetMsg[1] = 0xff
    tmpNetMsg[2] = 0  # RTR
    tmpNetMsg[3] = 0x04  # DLC
    tmpNetMsg[4] = 0x01
    tmpNetMsg[5:12] = 0x00.to_bytes(7, 'little')
    tmpNetMsg[12:16] = 0x03.to_bytes(4, 'little')
    tmpNetMsg[16:20] = u32_FrameId_Tx.to_bytes(4, 'little')
    g_u32RunTime = int(time.time() * 1000) - tickStart
    tmpNetMsg[20:24] = g_u32RunTime.to_bytes(4, 'little')
    PackAndSendNetMessage(tmpNetMsg)



def sendTargetModeOfOperation_WithFrameIdAndTime(node: int, targetMode: str):

    assert node in paras.SLAVE_EPOS_NODEID_E.values(), 'No motor corresponding to node %d exists' % node
    assert targetMode in paras.OPERATION_MOD_E.keys(), 'Motion mode does not exist' + targetMode
    global u32_FrameId_Tx
    u32_FrameId_Tx += 1
    tmpNetMsg = bytearray(24)

    tmpNetMsg[0:1] = paras.MS_COM_PROTOCOL_STDID_E['M2S_CMD_OperationMode'].to_bytes(2, 'little')
    tmpNetMsg[1] = 0xff
    tmpNetMsg[2] = 0x00  # RTR
    tmpNetMsg[3] = 0x02  # DLC
    tmpNetMsg[4:5] = node.to_bytes(1, 'little')
    tmpNetMsg[5:6] = paras.OPERATION_MOD_E[targetMode].to_bytes(1, 'little')
    tmpNetMsg[6:12] = 0x00.to_bytes(6, 'little')
    tmpNetMsg[12:16] = 0x03.to_bytes(4, 'little')
    tmpNetMsg[16:20] = u32_FrameId_Tx.to_bytes(4, 'little')
    g_u32RunTime = int(time.time() * 1000) - tickStart
    tmpNetMsg[20:24] = g_u32RunTime.to_bytes(4, 'little')
    PackAndSendNetMessage(tmpNetMsg)
    # return tmpNetMsg


def sendGWRemove():

    global u32_FrameId_Tx
    u32_FrameId_Tx += 1

    tmpNetMsg = bytearray(24)
    tmpNetMsg[0:1] = paras.MS_COM_PROTOCOL_STDID_E['M2S_CMD_GW_REMOVE1'].to_bytes(2, 'little')
    tmpNetMsg[1] = 0xff
    tmpNetMsg[2] = 0x00  # RTR
    tmpNetMsg[3] = 0x04  # DLC
    tmpNetMsg[4:8] = 0x00.to_bytes(4, 'little')
    tmpNetMsg[8:12] = 0x00.to_bytes(4, 'little')
    tmpNetMsg[12:16] = 0x03.to_bytes(4, 'little')
    tmpNetMsg[16:20] = u32_FrameId_Tx.to_bytes(4, 'little')
    g_u32RunTime = int(time.time() * 1000) - tickStart
    tmpNetMsg[20:24] = g_u32RunTime.to_bytes(4, 'little')
    PackAndSendNetMessage(tmpNetMsg)


def sendGWInstall():

    global u32_FrameId_Tx
    u32_FrameId_Tx += 1

    tmpNetMsg = bytearray(24)
    tmpNetMsg[0:1] = paras.MS_COM_PROTOCOL_STDID_E['M2S_CMD_GW_REMOVE1'].to_bytes(2, 'little')
    tmpNetMsg[1] = 0xff
    tmpNetMsg[2] = 0x00  # RTR
    tmpNetMsg[3] = 0x04  # DLC
    tmpNetMsg[4:8] = 0x01.to_bytes(4, 'little')
    tmpNetMsg[8:12] = 0x00.to_bytes(4, 'little')
    tmpNetMsg[12:16] = 0x03.to_bytes(4, 'little')
    tmpNetMsg[16:20] = u32_FrameId_Tx.to_bytes(4, 'little')
    g_u32RunTime = int(time.time() * 1000) - tickStart
    tmpNetMsg[20:24] = g_u32RunTime.to_bytes(4, 'little')
    PackAndSendNetMessage(tmpNetMsg)

def Control_Change():

    global u32_FrameId_Tx
    u32_FrameId_Tx += 1

    tmpNetMsg = bytearray(24)
    tmpNetMsg[0:1] = paras.MS_COM_PROTOCOL_STDID_E['M2S_CONTROL_CHANGE'].to_bytes(2, 'little')
    tmpNetMsg[1] = 0xff
    tmpNetMsg[2] = 0x00  # RTR
    tmpNetMsg[3] = 0x01  # DLC
    tmpNetMsg[4:8] = 0x01.to_bytes(4, 'little')
    tmpNetMsg[8:12] = 0x00.to_bytes(4, 'little')
    tmpNetMsg[12:16] = 0x03.to_bytes(4, 'little')
    tmpNetMsg[16:20] = u32_FrameId_Tx.to_bytes(4, 'little')
    g_u32RunTime = int(time.time() * 1000) - tickStart
    tmpNetMsg[20:24] = g_u32RunTime.to_bytes(4, 'little')
    PackAndSendNetMessage(tmpNetMsg)

def Operate_Send():

    global u32_FrameId_Tx
    u32_FrameId_Tx += 1

    tmpNetMsg = bytearray(24)
    tmpNetMsg[0:1] = paras.MS_COM_PROTOCOL_STDID_E['Operate_New'].to_bytes(2, 'little')
    tmpNetMsg[1] = 0xff
    tmpNetMsg[2] = 0x00  # RTR
    tmpNetMsg[3] = 0x06  # DLC
    tmpNetMsg[4] = 0x11
    tmpNetMsg[5] = 0x22
    tmpNetMsg[6] = 0x33
    tmpNetMsg[7] = 0x44
    tmpNetMsg[8] = 0x55
    tmpNetMsg[9] = 0x66
    tmpNetMsg[10] = 0x00
    tmpNetMsg[11] = 0x00
    tmpNetMsg[12:16] = 0x03.to_bytes(4, 'little')
    tmpNetMsg[16:20] = u32_FrameId_Tx.to_bytes(4, 'little')
    g_u32RunTime = int(time.time() * 1000) - tickStart
    tmpNetMsg[20:24] = g_u32RunTime.to_bytes(4, 'little')
    PackAndSendNetMessage(tmpNetMsg)

def Voice_Play(num):
    global u32_FrameId_Tx
    u32_FrameId_Tx += 1
    voice_list = [0xD0, 0xD1, 0xD2, 0xD3, 0xD4, 0xD5, 0xD6, 0xDF]
    tmpNetMsg = bytearray(24)
    tmpNetMsg[0:1] = paras.MS_COM_PROTOCOL_STDID_E['Voice_Play'].to_bytes(2, 'little')
    tmpNetMsg[1] = 0xff
    tmpNetMsg[2] = 0x00  # RTR
    tmpNetMsg[3] = 0x01  # DLC
    tmpNetMsg[4] = voice_list[num]
    tmpNetMsg[5] = 0x00
    tmpNetMsg[6] = 0x00
    tmpNetMsg[7] = 0x00
    tmpNetMsg[8] = 0x00
    tmpNetMsg[9] = 0x00
    tmpNetMsg[10] = 0x00
    tmpNetMsg[11] = 0x00
    tmpNetMsg[12:16] = 0x03.to_bytes(4, 'little')
    tmpNetMsg[16:20] = u32_FrameId_Tx.to_bytes(4, 'little')
    g_u32RunTime = int(time.time() * 1000) - tickStart
    tmpNetMsg[20:24] = g_u32RunTime.to_bytes(4, 'little')
    PackAndSendNetMessage(tmpNetMsg)

def Install_Ballon():
    global u32_FrameId_Tx
    u32_FrameId_Tx += 1
    tmpNetMsg = bytearray(24)
    tmpNetMsg[0:1] = paras.MS_COM_PROTOCOL_STDID_E['Install_Ballon'].to_bytes(2, 'little')
    tmpNetMsg[1] = 0xff
    tmpNetMsg[2] = 0x00  # RTR
    tmpNetMsg[3] = 0x01  # DLC
    tmpNetMsg[4] = 0x31
    tmpNetMsg[5] = 0x00
    tmpNetMsg[6] = 0x00
    tmpNetMsg[7] = 0x00
    tmpNetMsg[8] = 0x00
    tmpNetMsg[9] = 0x00
    tmpNetMsg[10] = 0x00
    tmpNetMsg[11] = 0x00
    tmpNetMsg[12:16] = 0x03.to_bytes(4, 'little')
    tmpNetMsg[16:20] = u32_FrameId_Tx.to_bytes(4, 'little')
    g_u32RunTime = int(time.time() * 1000) - tickStart
    tmpNetMsg[20:24] = g_u32RunTime.to_bytes(4, 'little')
    PackAndSendNetMessage(tmpNetMsg)


def PackAndSendNetMessage(byteArray: bytearray):

    tmpByteArray = bytearray(100)
    CheckSum = 0
    # len = 0
    tmpByteArray[0] = NET_FRAME_HEAD
    tmpByteArray[1] = NET_FRAME_HEAD
    length = 2
    for iter in range(len(byteArray)):
        if byteArray[iter] == NET_FRAME_CTRL or byteArray[iter] == NET_FRAME_HEAD or byteArray[iter] == NET_FRAME_TAIL:
            tmpByteArray[length] = NET_FRAME_CTRL
            length += 1
        tmpByteArray[length] = byteArray[iter]
        CheckSum += byteArray[iter]
        CheckSum = CheckSum % 256
        length += 1
    if CheckSum == NET_FRAME_CTRL or CheckSum == NET_FRAME_HEAD or CheckSum == NET_FRAME_TAIL:
        tmpByteArray[length] = NET_FRAME_CTRL
        length += 1
    tmpByteArray[length] = CheckSum
    length += 1
    tmpByteArray[length] = NET_FRAME_TAIL
    length += 1
    tmpByteArray[length] = NET_FRAME_TAIL
    length += 1
    socketConnect.send(tmpByteArray)

def getLocalIP():

    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(('8.8.8.8', 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


def ServerStart(ip, port):

    global socketWatch
    socketWatch.bind(("", port))
    socketWatch.listen(CONST_CLIENT_NUMBER)

    threadWatch = threading.Thread(target=ThreadFunc_SendMsgToSlave)
    print(f'Host server startup Host ip is {ip} Port is{port}')

    threadWatch.start()

def ClientStart(ip, port):
    threadWatch = threading.Thread(target=ThreadFunc_SendMsgToServer, args=(ip, port,))
    print(f'Connection server ip is {ip} port is{port}')
    threadWatch.start()

def ThreadFunc_SendMsgToSlave():

    u16_SyncIndex = 0
    u8_HeartBeatIndex = 0x00
    u16_PrintIndex = 950
    global socketConnect, isSlaveConnected
    while not isSlaveConnected:
        socketConnect, _ = socketWatch.accept()  # This method blocks the current thread: continuously listening to the
        print('Slave Connection Successful')
        thr = threading.Thread(target=ThreadFunc_DataReceive, args=(socketConnect,))
        thr.start()
        isSlaveConnected = True

    while SlaveState is not paras.MS_STAT_E['MS_STAT_OK']:
        time.sleep(0.1)  # Note that in python time.sleep() is only valid for the current process, so it is equivalent to Thread.Sleep() in c#.
    print('Commencement of normal communications')
    time.sleep(2)
    # threadContorl = threading.Thread(target=ThreadFunc_SendTargetVelocity)
    # threadContorl.start()
    # sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'], 10)
    while True:
        u16_SyncIndex += 1
        if u16_SyncIndex == 1000:
            u16_SyncIndex = 0
            sendSyncFrame_WithFrameIdAndTime()
        u8_HeartBeatIndex += 1
        if u8_HeartBeatIndex == 1:
            u8_HeartBeatIndex = 0
            if u32_FrameId_Tx < 1000000:
                sendHeartBeatFrame_WithFrameIdAndTime()

def ThreadFunc_SendMsgToServer(ip, port):

    u16_SyncIndex = 0
    u8_HeartBeatIndex = 0x00
    u16_PrintIndex = 950

    global socketWatch
    # Create sockets
    socketWatch = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    # Set up port multiplexing so that ports are released immediately after program exit
    socketWatch.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, True)

    # Client connection program
    server_addr = (ip, int(port))  # IP address + port

    global socketConnect, isSlaveConnected
    while not isSlaveConnected:
        socketWatch.connect(server_addr)  # Connection
        socketConnect = socketWatch
        print('Successful slave connection (client mode)')
        thr = threading.Thread(target=ThreadFunc_DataReceive, args=(socketConnect,))
        thr.start()
        print("THR OK")
        isSlaveConnected = True
        Operate_Send()

    SlaveState = paras.MS_STAT_E['MS_STAT_OK']
    while SlaveState is not paras.MS_STAT_E['MS_STAT_OK']:
        time.sleep(0.1)  # Note that in python time.sleep() is only valid for the current process, so it is equivalent to Thread.Sleep() in c#.
    print('Commencement of normal communications')
    time.sleep(2)
    # threadContorl = threading.Thread(target=ThreadFunc_SendTargetVelocity)
    # threadContorl.start()
    # sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'], 10)
    while True:
        time.sleep(1)
        u16_SyncIndex += 1
        if u16_SyncIndex == 1000:
            u16_SyncIndex = 0
            sendSyncFrame_WithFrameIdAndTime()
        u8_HeartBeatIndex += 1
        if u8_HeartBeatIndex == 1:
            u8_HeartBeatIndex = 0
            if u32_FrameId_Tx < 1000000:
                sendHeartBeatFrame_WithFrameIdAndTime()


def ThreadFunc_DataReceive(socketServer):
    global u32_FrameId_Rx, ActualAdvCountOfGw, ActualAdvSpeedOfGw, ActualAdvCurrentOfGw, ActualAdvTorqueOfGw
    # isRxValidPackage = False
    canMessage = {}
    time.sleep(0.5)
    while True:
        byteArray = socketServer.recv(100)
        for index in range(len(byteArray)):
            tmpByte = byteArray[index]
            isRxValidPackage = ParseRxByteFromNet(tmpByte)
            if isRxValidPackage:
                u32_FrameId_Rx += 1
                canMessage['StdId'] = int.from_bytes(g_RxBuf_Net[0:1], 'little')
                # if canMessage['StdId'] == paras.MS_COM_PROTOCOL_STDID_E['S2M_RPL_SyncFrame']:  # Host computes network latency
                if canMessage['StdId'] == paras.MS_COM_PROTOCOL_STDID_E['S2M_RPL_PosVelTorqAndSensor']:
                    ActualAdvCountOfGw = int.from_bytes(g_RxBuf_Net[4:8], 'little', signed=True)
                    ActualAdvSpeedOfGw = int.from_bytes(g_RxBuf_Net[8:12], 'little', signed=True)
                    # ActualAdvCurrentOfGw = int.from_bytes(g_RxBuf_Net[12:16], 'little', signed=True)
                    # ActualAdvTorqueOfGw = int.from_bytes(g_RxBuf_Net[16:20], 'little')

                    # [Current] Installed devices or non-installed devices
                    if g_RxBuf_Net[14] != 0 or g_RxBuf_Net[14] == 0:
                        # Splice two hexadecimal numbers
                        hex1 = g_RxBuf_Net[21]  # First hexadecimal value 21
                        hex2 = g_RxBuf_Net[20]  # Second hexadecimal value 20
                        combined_hex = (hex1 << 8) | hex2
                        # print(hex(combined_hex))

                        # To signed decimal torque data
                        ActualAdvCurrentOfGw = combined_hex if combined_hex < 0x8000 else combined_hex - 0x10000
                        # print(ActualAdvCurrentOfGw)

                    # [Torque] Installed instruments or non-installed instruments
                    if g_RxBuf_Net[14] != 0 or g_RxBuf_Net[14] == 0:
                        # Splice two hexadecimal numbers
                        hex1 = g_RxBuf_Net[23]  # First hexadecimal value 21
                        hex2 = g_RxBuf_Net[22]  # Second hexadecimal value 20
                        combined_hex = (hex1 << 8) | hex2
                        # print(hex(combined_hex))

                        # To signed decimal torque data
                        ActualAdvTorqueOfGw = combined_hex if combined_hex < 0x8000 else combined_hex - 0x10000
                        # print(ActualAdvTorqueOfGw)

                    # print(ActualAdvCountOfGw, ActualAdvSpeedOfGw, ActualAdvCurrentOfGw, ActualAdvTorqueOfGw)
                    global initFlag, initGwCount
                    if initFlag:
                        initFlag = False
                        initGwCount = ActualAdvCountOfGw

                elif g_RxBuf_Net[0] == 0x32 and g_RxBuf_Net[3] == 0x05 and g_RxBuf_Net[7] == 0x02:  # Balloon completion
                    global flag_ballon_install
                    flag_ballon_install = 1
                    print("Balloon installation complete")

                else:
                    canMessage['RTR'] = g_RxBuf_Net[2]
                    canMessage['DLC'] = g_RxBuf_Net[3]
                    canMessage['data'] = g_RxBuf_Net[4:12]
                    ParseSlaveMsg(canMessage)


# The following variables are used for parsing: these variables can only be used in the ParseRxByteFromNet function!!!!
NET_LastByte = 0
NET_BeginFlag = False
NET_CtrlFlag = False
NET_RevOffset = 0  # of bytes of data
NET_CheckSum = 0  # Checksums


def ParseRxByteFromNet(data: int) -> bool:

    global NET_RevOffset, NET_BeginFlag, NET_LastByte, m_TotalRxLostPackages, validNetRxBufDataCount, NET_CheckSum, g_RxBuf_Net, NET_CtrlFlag
    if (data == NET_FRAME_HEAD and NET_LastByte == NET_FRAME_HEAD) or NET_RevOffset > NET_MaxPackageSize:
        if 25 > NET_RevOffset > 0:
            m_TotalRxLostPackages += 1
        # reset
        NET_RevOffset = 0
        NET_BeginFlag = True
        NET_LastByte = data
        return False
    if data == NET_FRAME_TAIL and NET_LastByte == NET_FRAME_TAIL and NET_BeginFlag:
        NET_RevOffset -= 1  # Get the number of bytes of data minus the header, tail, and control characters.
        validNetRxBufDataCount = NET_RevOffset - 1  # Get the number of data bytes excluding header and tail, control characters and checksums
        NET_CheckSum -= NET_FRAME_TAIL
        NET_CheckSum -= g_RxBuf_Net[validNetRxBufDataCount]
        NET_LastByte = data
        NET_BeginFlag = False
        NET_CheckSum = NET_CheckSum % 256
        if NET_CheckSum == g_RxBuf_Net[validNetRxBufDataCount]:
            NET_CheckSum = 0
            return True
        else:
            m_TotalRxLostPackages += 1
            NET_CheckSum = 0
            return False
    NET_LastByte = data
    if NET_BeginFlag:
        if NET_CtrlFlag:
            g_RxBuf_Net[NET_RevOffset] = data
            NET_RevOffset += 1
            NET_CheckSum += data
            NET_CtrlFlag = False
            NET_LastByte = NET_FRAME_CTRL  # Why is it NET_FRAME_CTRL and not data?
        elif data == NET_FRAME_CTRL:
            NET_CtrlFlag = True
        else:
            g_RxBuf_Net[NET_RevOffset] = data
            NET_RevOffset += 1
            NET_CheckSum += data
    return False


def ParseSlaveMsg(canMessage: dict):

    global ReceivedSlaveHandshake, u32_FrameId_Tx, SlaveState, ActualAdvCountOfGw1, ActualAdvSpeedOfGw1, \
        ActualRotCountOfGw1, ActualRotSpeedOfGw1, ActualAdvCountOfCath, ActualAdvSpeedOfCath, ActualRotCountOfCath, \
        ActualRotSpeedOfCath, g_channel
    if canMessage['StdId'] == paras.MS_COM_PROTOCOL_STDID_E['S2M_RPL_HANDSHAKE']:
        if canMessage['DLC'] == 1:
            # If the slave has never sent a handshake signal and the slave has never sent an OPERATE signal and the received message is a handshake message
            if ReceivedSlaveHandshake != True and SlaveState != paras.MS_STAT_E['MS_STAT_OK'] and canMessage['data'][0] \
                    == paras.MS_STAT_E['MS_STAT_OK']:
                print('Master received handshake')
                ReceivedSlaveHandshake = True

                # Sent by the host
                u32_FrameId_Tx += 1
                tmpNetMsg = bytearray(13)
                tmpNetMsg[0:4] = u32_FrameId_Tx.to_bytes(4, 'little')  # Frame number
                g_u32RunTime = int(time.time() * 1000) - tickStart
                tmpNetMsg[4:8] = g_u32RunTime.to_bytes(4, 'little')  # The moment of sending
                tmpNetMsg[8:10] = paras.MS_COM_PROTOCOL_STDID_E['M2S_CMD_OPERATION'].to_bytes(2, 'little')
                tmpNetMsg[10] = 0x00  # RTRs
                tmpNetMsg[11] = 0x01  # DLC
                tmpNetMsg[12] = 0x01
                PackAndSendNetMessage(tmpNetMsg)
                print('master transmit operate')
        return
    elif canMessage['StdId'] == paras.MS_COM_PROTOCOL_STDID_E['S2M_RPL_OPERATION']:
        if ReceivedSlaveHandshake and SlaveState != paras.MS_STAT_E['MS_STAT_OK'] and canMessage['DLC'] == 1 and \
                canMessage['data'][0] == 1:
            SlaveState = paras.MS_STAT_E['MS_STAT_OK']
            print('slave reply operate')
        return
    elif canMessage['StdId'] == paras.MS_COM_PROTOCOL_STDID_E['S2M_RPL_PosAndVel']:
        if canMessage['DLC'] == 8:
            if canMessage['RTR'] == paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV']:
                ActualAdvCountOfGw1 = int.from_bytes(canMessage['data'][0:4], 'little')
                ActualAdvSpeedOfGw1 = int.from_bytes(canMessage['data'][4:8], 'little')
                return
            elif canMessage['RTR'] == paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ROT']:
                ActualRotCountOfGw1 = int.from_bytes(canMessage['data'][0:4], 'little')
                ActualRotSpeedOfGw1 = int.from_bytes(canMessage['data'][4:8], 'little')
                ActualRotCountOfGw1 = ActualRotCountOfGw1 - GW_ROT_MiddlePos
                return
            elif canMessage['RTR'] == paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_CATH_ADV']:
                ActualAdvCountOfCath = int.from_bytes(canMessage['data'][0:4], 'little')
                ActualAdvSpeedOfCath = int.from_bytes(canMessage['data'][4:8], 'little')
                ActualAdvCountOfCath = ActualAdvCountOfCath - CATH_ADV_MiddlePos
                return
            elif canMessage['RTR'] == paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_CATH_ROT']:
                ActualRotCountOfCath = int.from_bytes(canMessage['data'][0:4], 'little')
                ActualRotSpeedOfCath = int.from_bytes(canMessage['data'][4:8], 'little')
                return
        return
    elif canMessage['StdId'] == paras.MS_COM_PROTOCOL_STDID_E['S2M_RPL_CHANNEL']:
        if canMessage['DLC'] == 1:
            g_channel = canMessage['data'][0]
        return

def ThreadFunc_SendTargetVelocity(v):
    sendTargetVelocity_WithFrameIdAndTime(paras.SLAVE_EPOS_NODEID_E['SLAVE_NODEID_GW_ADV'], v)


