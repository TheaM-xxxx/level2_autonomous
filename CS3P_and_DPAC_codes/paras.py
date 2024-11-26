# -*- coding: utf-8 -*-
# @Author   : Linsen Zhang
# node id of each EPOS on the slave side
SLAVE_EPOS_NODEID_E = {'SLAVE_NODEID_GW_ADV': 0x01,  # Guide Wire Push
                       'SLAVE_NODEID_GW_ROT': 0x02,  # Guide wire rotation
                       'SLAVE_NODEID_CATH_ADV': 0x03,  # catheter push
                       'SLAVE_NODEID_CATH_ROT': 0x04,  # Catheter rotation
                       'SLAVE_NODEID_YVAL_ROT': 0x05,  # Y valve opening and closing
                       'SLAVE_NODEID_CH1_FRONT': 0x06,  # Channel 1 front end clamping motor
                       'SLAVE_NODEID_CH1_REAR': 0x07,  # Channel 1 Tail Clamp Motor
                       'SLAVE_NODEID_CH2_FRONT': 0x08,  # Channel 2 front end clamping motor
                       'SLAVE_NODEID_CH2_REAR': 0x09,  # Channel 2 Tail Clamp Motor
                       }

# StdID of the master-slave communication protocol
MS_COM_PROTOCOL_STDID_E = {'M2S_CMD_HeartBeat': 0x03,  # Master sends “heartbeat” to slave, format: 0x11, 00, 00, 01, 00
                           'S2M_RPL_HeartBeat': 0x13,  # Master sends “heartbeat” to slave, format: 0x12, 00, 00, 01, 00

                           'M2S_CMD_EMER': 0x01,  # Master sends “emergency stop” to slave, format: 0x21, 00, 00, 01, 00
                           'S2M_CMD_EMER': 0x11,  # The master sends the slave an “emergency stop” #

                           'M2S_CMD_HANDSHAKE': 0x31,  # Handshake signal from master to slave, format: 0x31, 00, 00, 01, MS_STAT_OK
                           'S2M_RPL_HANDSHAKE': 0x32,  # Handshake signals from slave to master

                           'M2S_CMD_OPERATION': 0x2B,  # Master to slave: official start? Format: 0x41, 00, 00, 01, 01
                           'S2M_RPL_OPERATION': 0x3B,  # Slave to master: official start! Format: 0x42, 00, 00, 01, 01

                           'M2S_CMD_CLEARERROR': 0x0F,  # master to slave: clear EPOS error code
                           'S2M_RPL_ERRORCODE': 0x1F,  # slave to master: EPOS error code

                           'M2S_CMD_POSITION': 0x07,  # master to slave: desired location of each EPOS data[0]=nodeID, data[1-4]
                           'M2S_CMD_VELOCITY': 0x09,  # master to slave: desired speed for each EPOS data[0]=nodeID, data[1-4]
                           # Note that this is not the same as when communicating with a master-slave CAN!!!!
                           'S2M_RPL_PosAndVel': 0x37,  # Slave to master: actual position of each EPOS RTR=nodeID, position data[0-3], velocity data[4-7]

                           'M2S_CMD_GW_REMOVE1': 0x21,  # Master to Slave: Release Guide Wire #1
                           'M2S_CMD_GW_INSTALL1': 0x21,  # Master to Slave: Installation of #1 guide wire

                           'M2S_CMD_CLAMP': 0x21,  # Master to slave: desired state of each clamping motor
                           'S2M_RPL_CLAMP': 0x31,  # Slave to master: actual status of each clamping motor

                           'M2S_CMD_OperationMode': 0x05,
                           # master sends to slave: EPOS movement mode, data[0]': nodeid, data[1]=PPM, PVM (OPERATION_MOD_E)

                           'S2M_RPL_TORQUE': 0x1E,  # Slave feedback to master torque value (frequency 15K-25K Hz): u16

                           'M2S_CMD_SyncFrame': 0x04,  # Synchronization frames: format Master sends frame number Master sends time t1 0xB1, 00, 00, 01, 01
                           'S2M_RPL_SyncFrame': 0x14,
                           # Synchronized frames: format Slave sends frame number Slave takes delta 0xB2, 00, 00, 08 from receive to send, master sends frame number Master sends time t1

                           'S2M_RPL_CHANNEL': 0x38,  # Reply to the current channel from the end

                           'S2M_RPL_PosVelTorqAndSensor': 0x17,
                           # Slave sends to master when antegrade wire/balloon push motor: position, speed, motor torque, sensor frequency, format: 0xD2, 0x00, int32 position, int32 speed, int16 motor torque, u16 sensor frequency

                           'M2S_CMD_PROVELOCITY': 0xE1,  # Master sets the maximum slave motor operating speed

                           'M2S_CONTROL_CHANGE': 0x41,  # Speed-position mode switching

                           'Operate_New': 0x26,  # New slave Operate

                            'Voice_Play': 0x29,  # New slave playback voice

                            'Install_Ballon': 0x22

                           }

MS_STAT_E = {'MS_STAT_RESET': 0,  # Reset
             'MS_STAT_INI': 1,  # Initialization in progress: self-test, check EPOS
             'MS_STAT_OK': 2,  # ok
             'MS_STAT_ERR': 3  # Error
             }

# Sport mode
OPERATION_MOD_E = {'MODE_NC': 0,
                   'PPM': 1,  # Profile Position Mode
                   'PVM': 3,  # Profile Velocity Mode
                   'CSP': 8,  # cyclic synchronous position mode
                   'CSV': 9,  # cyclic synchronous velocity mode
                   'CST': 10,  # cyclic synchronous torque mode
                   'HMM': 6,  # Homing Mode
                   }
