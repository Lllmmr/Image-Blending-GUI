# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'gui.ui'
#
# Created by: PyQt5 UI code generator 5.14.2
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets
from label import *

class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(917, 760)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(MainWindow.sizePolicy().hasHeightForWidth())
        MainWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.gridLayout = QtWidgets.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName("gridLayout")
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.label_img = MyLabel(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_img.sizePolicy().hasHeightForWidth())
        self.label_img.setSizePolicy(sizePolicy)
        self.label_img.setMinimumSize(QtCore.QSize(641, 641))
        font = QtGui.QFont()
        font.setPointSize(16)
        font.setStrikeOut(False)
        font.setKerning(True)
        self.label_img.setFont(font)
        self.label_img.setText("")
        self.label_img.setObjectName("label_img")
        self.horizontalLayout.addWidget(self.label_img)
        self.frame = QtWidgets.QFrame(self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame.sizePolicy().hasHeightForWidth())
        self.frame.setSizePolicy(sizePolicy)
        self.frame.setMinimumSize(QtCore.QSize(250, 641))
        self.frame.setMaximumSize(QtCore.QSize(240, 16777215))
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.frame)
        self.verticalLayout.setObjectName("verticalLayout")
        self.radio_source = QtWidgets.QRadioButton(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.radio_source.sizePolicy().hasHeightForWidth())
        self.radio_source.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.radio_source.setFont(font)
        self.radio_source.setObjectName("radio_source")
        self.verticalLayout.addWidget(self.radio_source)
        self.radio_Mixed = QtWidgets.QRadioButton(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.radio_Mixed.sizePolicy().hasHeightForWidth())
        self.radio_Mixed.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.radio_Mixed.setFont(font)
        self.radio_Mixed.setObjectName("radio_Mixed")
        self.verticalLayout.addWidget(self.radio_Mixed)
        self.radio_Transfer = QtWidgets.QRadioButton(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.radio_Transfer.sizePolicy().hasHeightForWidth())
        self.radio_Transfer.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.radio_Transfer.setFont(font)
        self.radio_Transfer.setObjectName("radio_Transfer")
        self.verticalLayout.addWidget(self.radio_Transfer)
        self.radio_Local = QtWidgets.QRadioButton(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.radio_Local.sizePolicy().hasHeightForWidth())
        self.radio_Local.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.radio_Local.setFont(font)
        self.radio_Local.setObjectName("radio_Local")
        self.verticalLayout.addWidget(self.radio_Local)
        self.check_ForeFla = QtWidgets.QCheckBox(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.check_ForeFla.sizePolicy().hasHeightForWidth())
        self.check_ForeFla.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.check_ForeFla.setFont(font)
        self.check_ForeFla.setObjectName("check_ForeFla")
        self.verticalLayout.addWidget(self.check_ForeFla)
        self.widget_3 = QtWidgets.QWidget(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_3.sizePolicy().hasHeightForWidth())
        self.widget_3.setSizePolicy(sizePolicy)
        self.widget_3.setMinimumSize(QtCore.QSize(240, 40))
        self.widget_3.setObjectName("widget_3")
        self.horizontalLayout_5 = QtWidgets.QHBoxLayout(self.widget_3)
        self.horizontalLayout_5.setObjectName("horizontalLayout_5")
        self.label_3 = QtWidgets.QLabel(self.widget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_3.sizePolicy().hasHeightForWidth())
        self.label_3.setSizePolicy(sizePolicy)
        self.label_3.setMinimumSize(QtCore.QSize(25, 0))
        font = QtGui.QFont()
        font.setPointSize(9)
        self.label_3.setFont(font)
        self.label_3.setObjectName("label_3")
        self.horizontalLayout_5.addWidget(self.label_3)
        self.Slider_L = QtWidgets.QSlider(self.widget_3)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Slider_L.sizePolicy().hasHeightForWidth())
        self.Slider_L.setSizePolicy(sizePolicy)
        self.Slider_L.setMinimumSize(QtCore.QSize(0, 0))
        self.Slider_L.setOrientation(QtCore.Qt.Horizontal)
        self.Slider_L.setObjectName("Slider_L")
        self.horizontalLayout_5.addWidget(self.Slider_L)
        self.verticalLayout.addWidget(self.widget_3)
        self.widget_4 = QtWidgets.QWidget(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_4.sizePolicy().hasHeightForWidth())
        self.widget_4.setSizePolicy(sizePolicy)
        self.widget_4.setMinimumSize(QtCore.QSize(240, 40))
        self.widget_4.setObjectName("widget_4")
        self.horizontalLayout_6 = QtWidgets.QHBoxLayout(self.widget_4)
        self.horizontalLayout_6.setObjectName("horizontalLayout_6")
        self.label_4 = QtWidgets.QLabel(self.widget_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Preferred)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_4.sizePolicy().hasHeightForWidth())
        self.label_4.setSizePolicy(sizePolicy)
        self.label_4.setMinimumSize(QtCore.QSize(25, 0))
        self.label_4.setObjectName("label_4")
        self.horizontalLayout_6.addWidget(self.label_4)
        self.Slider_H = QtWidgets.QSlider(self.widget_4)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Slider_H.sizePolicy().hasHeightForWidth())
        self.Slider_H.setSizePolicy(sizePolicy)
        self.Slider_H.setMinimumSize(QtCore.QSize(0, 0))
        self.Slider_H.setOrientation(QtCore.Qt.Horizontal)
        self.Slider_H.setObjectName("Slider_H")
        self.horizontalLayout_6.addWidget(self.Slider_H)
        self.verticalLayout.addWidget(self.widget_4)
        self.check_ForeIllu = QtWidgets.QCheckBox(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.check_ForeIllu.sizePolicy().hasHeightForWidth())
        self.check_ForeIllu.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.check_ForeIllu.setFont(font)
        self.check_ForeIllu.setObjectName("check_ForeIllu")
        self.verticalLayout.addWidget(self.check_ForeIllu)
        self.widget = QtWidgets.QWidget(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget.sizePolicy().hasHeightForWidth())
        self.widget.setSizePolicy(sizePolicy)
        self.widget.setMinimumSize(QtCore.QSize(240, 40))
        self.widget.setObjectName("widget")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout(self.widget)
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.label = QtWidgets.QLabel(self.widget)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label.setFont(font)
        self.label.setObjectName("label")
        self.horizontalLayout_3.addWidget(self.label)
        self.Slider_a = QtWidgets.QSlider(self.widget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Expanding, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Slider_a.sizePolicy().hasHeightForWidth())
        self.Slider_a.setSizePolicy(sizePolicy)
        self.Slider_a.setOrientation(QtCore.Qt.Horizontal)
        self.Slider_a.setObjectName("Slider_a")
        self.horizontalLayout_3.addWidget(self.Slider_a)
        self.verticalLayout.addWidget(self.widget)
        self.widget_2 = QtWidgets.QWidget(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.widget_2.sizePolicy().hasHeightForWidth())
        self.widget_2.setSizePolicy(sizePolicy)
        self.widget_2.setMinimumSize(QtCore.QSize(240, 40))
        self.widget_2.setObjectName("widget_2")
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout(self.widget_2)
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.label_2 = QtWidgets.QLabel(self.widget_2)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.label_2.setFont(font)
        self.label_2.setObjectName("label_2")
        self.horizontalLayout_4.addWidget(self.label_2)
        self.Slider_b = QtWidgets.QSlider(self.widget_2)
        self.Slider_b.setMinimumSize(QtCore.QSize(0, 0))
        self.Slider_b.setOrientation(QtCore.Qt.Horizontal)
        self.Slider_b.setObjectName("Slider_b")
        self.horizontalLayout_4.addWidget(self.Slider_b)
        self.verticalLayout.addWidget(self.widget_2)
        self.check_ForeColor = QtWidgets.QCheckBox(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.check_ForeColor.sizePolicy().hasHeightForWidth())
        self.check_ForeColor.setSizePolicy(sizePolicy)
        font = QtGui.QFont()
        font.setPointSize(16)
        self.check_ForeColor.setFont(font)
        self.check_ForeColor.setObjectName("check_ForeColor")
        self.verticalLayout.addWidget(self.check_ForeColor)
        self.frame_2 = QtWidgets.QFrame(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.frame_2.sizePolicy().hasHeightForWidth())
        self.frame_2.setSizePolicy(sizePolicy)
        self.frame_2.setMinimumSize(QtCore.QSize(240, 40))
        self.frame_2.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame_2.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame_2.setObjectName("frame_2")
        self.horizontalLayout_2 = QtWidgets.QHBoxLayout(self.frame_2)
        self.horizontalLayout_2.setObjectName("horizontalLayout_2")
        self.Button_ForeColor = QtWidgets.QPushButton(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Button_ForeColor.sizePolicy().hasHeightForWidth())
        self.Button_ForeColor.setSizePolicy(sizePolicy)
        self.Button_ForeColor.setMinimumSize(QtCore.QSize(90, 30))
        self.Button_ForeColor.setMaximumSize(QtCore.QSize(60, 30))
        self.Button_ForeColor.setObjectName("Button_ForeColor")
        self.horizontalLayout_2.addWidget(self.Button_ForeColor)
        self.label_ForeColor = QtWidgets.QLabel(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.label_ForeColor.sizePolicy().hasHeightForWidth())
        self.label_ForeColor.setSizePolicy(sizePolicy)
        self.label_ForeColor.setMinimumSize(QtCore.QSize(70, 30))
        self.label_ForeColor.setMaximumSize(QtCore.QSize(60, 30))
        self.label_ForeColor.setText("")
        self.label_ForeColor.setObjectName("label_ForeColor")
        self.horizontalLayout_2.addWidget(self.label_ForeColor)
        self.check_Gray = QtWidgets.QCheckBox(self.frame_2)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.check_Gray.sizePolicy().hasHeightForWidth())
        self.check_Gray.setSizePolicy(sizePolicy)
        self.check_Gray.setObjectName("check_Gray")
        self.horizontalLayout_2.addWidget(self.check_Gray)
        self.verticalLayout.addWidget(self.frame_2)
        self.Button_poisson = QtWidgets.QPushButton(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Button_poisson.sizePolicy().hasHeightForWidth())
        self.Button_poisson.setSizePolicy(sizePolicy)
        self.Button_poisson.setMinimumSize(QtCore.QSize(240, 50))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.Button_poisson.setFont(font)
        self.Button_poisson.setObjectName("Button_poisson")
        self.verticalLayout.addWidget(self.Button_poisson)
        self.Button_Pyramid = QtWidgets.QPushButton(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Button_Pyramid.sizePolicy().hasHeightForWidth())
        self.Button_Pyramid.setSizePolicy(sizePolicy)
        self.Button_Pyramid.setMinimumSize(QtCore.QSize(240, 50))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.Button_Pyramid.setFont(font)
        self.Button_Pyramid.setObjectName("Button_Pyramid")
        self.verticalLayout.addWidget(self.Button_Pyramid)
        self.Button_DIH = QtWidgets.QPushButton(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Button_DIH.sizePolicy().hasHeightForWidth())
        self.Button_DIH.setSizePolicy(sizePolicy)
        self.Button_DIH.setMinimumSize(QtCore.QSize(240, 50))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.Button_DIH.setFont(font)
        self.Button_DIH.setObjectName("Button_DIH")
        self.verticalLayout.addWidget(self.Button_DIH)
        self.Button_GAN = QtWidgets.QPushButton(self.frame)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.Button_GAN.sizePolicy().hasHeightForWidth())
        self.Button_GAN.setSizePolicy(sizePolicy)
        self.Button_GAN.setMinimumSize(QtCore.QSize(240, 50))
        font = QtGui.QFont()
        font.setPointSize(16)
        self.Button_GAN.setFont(font)
        self.Button_GAN.setObjectName("Button_GAN")
        self.verticalLayout.addWidget(self.Button_GAN)
        self.horizontalLayout.addWidget(self.frame)
        self.gridLayout.addLayout(self.horizontalLayout, 0, 0, 1, 1)
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 917, 23))
        self.menubar.setObjectName("menubar")
        self.menuFile = QtWidgets.QMenu(self.menubar)
        self.menuFile.setObjectName("menuFile")
        self.menuOpen = QtWidgets.QMenu(self.menuFile)
        self.menuOpen.setObjectName("menuOpen")
        MainWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(MainWindow)
        self.statusbar.setObjectName("statusbar")
        MainWindow.setStatusBar(self.statusbar)
        self.action_dst_img = QtWidgets.QAction(MainWindow)
        self.action_dst_img.setObjectName("action_dst_img")
        self.action_src_img = QtWidgets.QAction(MainWindow)
        self.action_src_img.setObjectName("action_src_img")
        self.action_save_as = QtWidgets.QAction(MainWindow)
        self.action_save_as.setObjectName("action_save_as")
        self.actionSave_Mask = QtWidgets.QAction(MainWindow)
        self.actionSave_Mask.setObjectName("actionSave_Mask")
        self.actionSave_Src = QtWidgets.QAction(MainWindow)
        self.actionSave_Src.setObjectName("actionSave_Src")
        self.menuOpen.addAction(self.action_dst_img)
        self.menuOpen.addAction(self.action_src_img)
        self.menuFile.addAction(self.menuOpen.menuAction())
        self.menuFile.addAction(self.action_save_as)
        self.menuFile.addAction(self.actionSave_Mask)
        self.menuFile.addAction(self.actionSave_Src)
        self.menubar.addAction(self.menuFile.menuAction())

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "MainWindow"))
        self.radio_source.setText(_translate("MainWindow", "Normal"))
        self.radio_Mixed.setText(_translate("MainWindow", "Mixed"))
        self.radio_Transfer.setText(_translate("MainWindow", "Transfer"))
        self.radio_Local.setText(_translate("MainWindow", "Local Changes"))
        self.check_ForeFla.setText(_translate("MainWindow", "Flattening"))
        self.label_3.setText(_translate("MainWindow", "low"))
        self.label_4.setText(_translate("MainWindow", "high"))
        self.check_ForeIllu.setText(_translate("MainWindow", "Illumination"))
        self.label.setText(_translate("MainWindow", "a"))
        self.label_2.setText(_translate("MainWindow", "b"))
        self.check_ForeColor.setText(_translate("MainWindow", "Color"))
        self.Button_ForeColor.setText(_translate("MainWindow", "Color"))
        self.check_Gray.setText(_translate("MainWindow", "Gray"))
        self.Button_poisson.setText(_translate("MainWindow", "Poisson Image Editing"))
        self.Button_Pyramid.setText(_translate("MainWindow", "Convolution Pyramids"))
        self.Button_DIH.setText(_translate("MainWindow", "DIH"))
        self.Button_GAN.setText(_translate("MainWindow", "GP-GAN"))
        self.menuFile.setTitle(_translate("MainWindow", "File"))
        self.menuOpen.setTitle(_translate("MainWindow", "Open"))
        self.action_dst_img.setText(_translate("MainWindow", "dst image"))
        self.action_src_img.setText(_translate("MainWindow", "src image"))
        self.action_save_as.setText(_translate("MainWindow", "Save As"))
        self.actionSave_Mask.setText(_translate("MainWindow", "Save Mask"))
        self.actionSave_Src.setText(_translate("MainWindow", "Save Src"))
