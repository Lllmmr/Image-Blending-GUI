import sys	
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from functools import partial
from gui import *

class MyWindow(QMainWindow,Ui_MainWindow):
	def __init__(self, parent=None):
		super(MyWindow,self).__init__(parent)
		self.setupUi(self)
		self.slot_init()
		self.changeColor()

	def slot_init(self):
		self.action_dst_img.triggered.connect(partial(self.label_img.load_img,False))
		self.action_src_img.triggered.connect(partial(self.label_img.load_img,True))
		self.actionSave_Src.triggered.connect(partial(self.label_img.saveSrc,True))
		self.actionSave_Mask.triggered.connect(partial(self.label_img.saveSrc,False))
		self.action_save_as.triggered.connect(partial(self.label_img.save_img))
		self.radio_Local.toggled.connect(partial(self.label_img.SourceMode,self.radio_Local))
		self.Button_poisson.clicked.connect(partial(self.label_img.poissonEdit,self.radio_source,self.radio_Mixed,
			self.check_ForeFla,self.Slider_L,self.Slider_H,self.check_ForeIllu,self.Slider_a,self.Slider_b,
			self.check_ForeColor,self.check_Gray))
		self.Button_ForeColor.clicked.connect(partial(self.selectColor))
		self.Button_GAN.clicked.connect(self.label_img.GANEdit)
		self.Button_Pyramid.clicked.connect(self.label_img.PyramidEdit)
		self.Button_DIH.clicked.connect(self.label_img.DIHEdit)
		self.radio_source.setChecked(True)
		self.Slider_a.setValue(9)
		self.Slider_b.setValue(19)

	def changeColor(self):
		self.label_ForeColor.setStyleSheet('background-color:%s'%self.label_img.setColor.name())

	def selectColor(self):
		col= QColorDialog.getColor()
		if col.isValid():
			self.label_img.setColor=col
			self.changeColor()


if __name__ == '__main__':
	app = QApplication(sys.argv)
	myWin = MyWindow()
	myWin.show()
	sys.exit(app.exec_())