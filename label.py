import sys
import cv2
import numpy as np
from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from functools import partial

from GP_GAN_blending.run_gp_gan import *
from deepimageharm.deepHarmonization import *
from LaplacePyramids.LaplacePyramid import *

class MyLabel(QLabel):
	def __init__(self,parent=None):
		super(MyLabel,self).__init__(parent)
		self.setMouseTracking(False)
		self.grabKeyboard()
		self.pos_xy = []
		self.dst_img=None
		self.src_img=None
		self.result_img=None
		self.posX=0.0
		self.posY=0.0
		self.startX=0
		self.startY=0
		self.tmpX=0
		self.tmpY=0
		self.moveMode=False
		self.zoom=1.0
		self.showRes=False
		self.sourceMode=True
		self.setColor=QColor(255,255,255)

	def init_maskarea(self):
		self.mask_u=0
		self.mask_l=0
		self.mask_d=self.mask.shape[0]
		self.mask_r=self.mask.shape[1]

	def load_img(self,is_src):
		imgName,imgType=QFileDialog.getOpenFileName(self, "Open", "", "*.jpg *.png")
		if imgName=='':
			return
		if is_src:
			self.src_img=cv2.imread(imgName)
			if self.src_img is None:
				return
			self.mask=np.zeros([self.src_img.shape[0],self.src_img.shape[1]],np.uint8)
			self.mask[:,:]=255
			self.zoom=1.0
			self.posX=0.0
			self.posY=0.0
			self.init_maskarea()
		else:
			self.dst_img=cv2.imread(imgName)
			if self.dst_img is None:
				return
		self.display_img()

	def save_img(self):
		if self.result_img is None:
			return
		imgName,imgType=QFileDialog.getSaveFileName(self,"Save","","*.png")
		if imgName=='':
			return
		cv2.imwrite(imgName,self.result_img)

	def saveSrc(self,is_src):
		if self.src_img is None:
			return
		imgName,imgType=QFileDialog.getSaveFileName(self,"Save","","*.png")
		if imgName=='':
			return
		if self.dst_img is None:
			w=self.src_img.shape[1]
			h=self.src_img.shape[0]
		else:
			w=int(self.zoom*self.dst_img.shape[1])
			h=int(w/self.src_img.shape[1]*self.src_img.shape[0])
		if is_src:
			src=cv2.resize(self.src_img,(w,h),cv2.INTER_LINEAR)
			cv2.imwrite(imgName,src)
		else:
			mask=cv2.resize(self.mask,(w,h),cv2.INTER_LINEAR)
			cv2.imwrite(imgName,mask)

	def SourceMode(self,btn):
		if btn.isChecked():
			self.sourceMode=False
		else:
			self.sourceMode=True
		self.display_img()

	def display_img(self):
		self.showRes=False
		board=QPixmap(self.width(),self.height())
		board.fill(Qt.transparent)
		boPainter=QPainter(board)
		boPainter.setCompositionMode(QPainter.CompositionMode_Source)
		if self.dst_img is not None and self.sourceMode:
			img=QPixmap(QImage(cv2.cvtColor(self.dst_img,cv2.COLOR_BGR2RGB).data,self.dst_img.shape[1],self.dst_img.shape[0],QImage.Format_RGB888))
			img=img.scaledToWidth(self.width())
			boPainter.drawPixmap(0,0,img)
		if self.src_img is not None:
			img=cv2.merge((self.src_img,self.mask))
			img=QPixmap(QImage(img.data,self.src_img.shape[1],self.src_img.shape[0],QImage.Format_ARGB32))
			img=img.scaledToWidth(int((self.width())*self.zoom))
			if self.dst_img is not None and self.sourceMode:
				imgpainter=QPainter(img)
				imgpainter.setCompositionMode(QPainter.CompositionMode_DestinationIn)
				imgpainter.fillRect(img.rect(),QColor(0,0,0,200))
				imgpainter.end()
				boPainter.setCompositionMode(QPainter.CompositionMode_SourceAtop)
			boPainter.drawPixmap(int(self.width()*self.posX+self.tmpX),int(self.width()*self.posY+self.tmpY),img)
		boPainter.end()
		self.setPixmap(board)

	def display_res(self):
		self.showRes=True
		if self.result_img is None:
			return
		board=QPixmap(self.width(),self.height())
		board.fill(Qt.transparent)
		boPainter = QPainter(board)
		boPainter.setCompositionMode(QPainter.CompositionMode_Source)
		dis=QPixmap(QImage(cv2.cvtColor(self.result_img,cv2.COLOR_BGR2RGB).data,self.result_img.shape[1],self.result_img.shape[0],QImage.Format_RGB888))
		dis=dis.scaledToWidth(self.width())
		boPainter.drawPixmap(0,0,dis)
		boPainter.end()
		self.setPixmap(board)

	def getEditSrc(self,srcImg=None,fitDst=True):
		if srcImg is None:
			srcImg=self.src_img
		if fitDst:
			if self.tmpX!=0 or self.tmpY!=0:
				print(self.tmpX,self.tmpY)
			w=int(self.zoom*self.dst_img.shape[1])
			h=int(w/srcImg.shape[1]*srcImg.shape[0])
			x=int(self.posX*self.dst_img.shape[1])	
			y=int(self.posY*self.dst_img.shape[1])
			src=cv2.resize(srcImg,(w,h),cv2.INTER_LINEAR)
			mask=cv2.resize(self.mask,(w,h),cv2.INTER_LINEAR)		

			img_h,img_w=self.dst_img.shape[0:2]	

			times=w/self.mask.shape[1]
		else:
			h,w=srcImg.shape[0:2]
			img_h,img_w=h,w
			x,y=0,0
			src=srcImg.copy()
			mask=self.mask.copy()
			times=1

		src_u,src_d=max(0,int(times*self.mask_u)),min(src.shape[0],int(times*self.mask_d))
		src_l,src_r=max(0,int(times*self.mask_l)),min(src.shape[1],int(times*self.mask_r))
		roi_u,roi_d,roi_l,roi_r=y+src_u,y+src_d,x+src_l,x+src_r
		if roi_l<0:
			src_l=src_l-roi_l
			roi_l=0
		if roi_u<0:
			src_u=src_u-roi_u
			roi_u=0
		if roi_r>img_w:
			src_r=img_w-roi_l+src_l
			roi_r=img_w
		if roi_d>img_h:
			src_d=img_h-roi_u+src_u
			roi_d=img_h

		src=src[src_u:src_d,src_l:src_r]
		mask=mask[src_u:src_d,src_l:src_r]
		_x=(roi_l+roi_r)//2
		_y=(roi_u+roi_d)//2
		return src,mask,_x,_y


	def poissonEdit(self,btnNor,btnMix,cheFFla,sliL,sliH,cheIllu,slia,slib,cheFCo,cheGr):
		if self.sourceMode:
			if self.dst_img is None or self.src_img is None:
				return

			src,mask,_x,_y=self.getEditSrc()

			if btnNor.isChecked():
				cloneMode=cv2.NORMAL_CLONE
			elif btnMix.isChecked():
				cloneMode=cv2.MIXED_CLONE
			else:
				cloneMode=cv2.MONOCHROME_TRANSFER

			try:
				self.result_img=cv2.seamlessClone(src,self.dst_img,mask,(_x,_y),cloneMode)
			except Exception as e:
				print(e)
				print(roi_u,roi_d,roi_l,roi_r)
				print(src_u,src_d,src_l,src_r)
				return
			self.display_res()
			return
		if self.src_img is None:
			return
		res=self.src_img.copy()
		if cheFCo.isChecked():
			r=self.setColor.red()
			g=self.setColor.green()
			b=self.setColor.blue()
			if r+g+b!=765:
				_b,_g,_r=cv2.split(cv2.bitwise_and(res,cv2.merge((self.mask,self.mask,self.mask))))
				num=np.sum(self.mask/255)
				mulr,mulg,mulb=r*num/np.sum(_r),g*num/np.sum(_g),b*num/np.sum(_b)
				mulr,mulg,mulb=mulr**0.5,mulg**0.5,mulb**0.5
				if mulr<0.5:
					mulr=0.5
				if mulr>2.5:
					mulr=2.5
				if mulg<0.5:
					mulg=0.5
				if mulg>2.5:
					mulg=2.5
				if mulb<0.5:
					mulb=0.5
				if mulb>2.5:
					mulb=2.5
				print(mulr,mulg,mulb)
				res=cv2.colorChange(res,self.mask,mulr,mulg,mulb)
			if cheGr.isChecked():
				dst=cv2.cvtColor(cv2.cvtColor(self.src_img,cv2.COLOR_BGR2GRAY),cv2.COLOR_GRAY2BGR)
				src,mask,_x,_y=self.getEditSrc(res,False)
				res=cv2.seamlessClone(src,dst,mask,(_x,_y),cv2.NORMAL_CLONE)

		if cheFFla.isChecked():
			res=cv2.textureFlattening(res,self.mask,res,sliL.value(),sliH.value())
		if cheIllu.isChecked():
			res=cv2.illuminationChange(res,self.mask,res,(slia.value()+1)/50,(slib.value()+1)/100)

		self.result_img=res
		self.display_res()

	def GANEdit(self):
		if not self.sourceMode:
			return
		if self.dst_img is None or self.src_img is None:
			return
		src,mask,_x,_y=self.getEditSrc()
		try:
			self.result_img=gpganblending(src,self.dst_img,cv2.merge((mask,mask,mask)),(_x,_y))
		except Exception as e:
			print(e)
			print(_x,_y)
			return
		self.display_res()

	def PyramidEdit(self):
		if not self.sourceMode:
			return
		if self.dst_img is None or self.src_img is None:
			return
		src,mask,_x,_y=self.getEditSrc()
		self.result_img=LaplacePyramid(src,cv2.merge((mask,mask,mask)),self.dst_img,_x,_y)
		self.display_res()
		print("f")

	def DIHEdit(self):
		if not self.sourceMode:
			return
		if self.dst_img is None or self.src_img is None:
			return
		src,mask,_x,_y=self.getEditSrc()
		try:
			self.result_img=DIH(src,self.dst_img,cv2.merge((mask,mask,mask)),(_x,_y))
		except Exception as e:
			print(e)
			print(_x,_y)
			return
		self.display_res()

	def resizeEvent(self,event):
		if self.showRes:
			self.display_res()
		else:
			self.display_img()

	def paintEvent(self, event):
		super().paintEvent(event)
		painter = QPainter()
		painter.begin(self)
		pen = QPen(Qt.red, 1, Qt.SolidLine)
		painter.setPen(pen)
		if len(self.pos_xy) > 1:
			point_start = self.pos_xy[0]
			for pos_tmp in self.pos_xy:
				point_end = pos_tmp
				painter.drawLine(point_start[0], point_start[1], point_end[0], point_end[1])
				point_start = point_end
		painter.end()

	def keyPressEvent(self,event):
		if event.key() == Qt.Key_Space:
			self.moveMode=True
			self.pos_xy.clear()
			self.update()
			self.setCursor(Qt.PointingHandCursor)

	def keyReleaseEvent(self,event):
		if event.key() == Qt.Key_Space:
			self.moveMode=False
			self.setCursor(Qt.ArrowCursor)

	def wheelEvent(self,event):
		if self.src_img is None:
			return
		amount=event.angleDelta().y()/120
		self.zoom*=1.02**amount
		self.display_img()

	def mousePressEvent(self, event):
		if self.moveMode:
			self.startX,self.startY=event.pos().x(),event.pos().y()
			return
		if event.buttons() == QtCore.Qt.RightButton and self.src_img is not None:
			self.mask[:,:]=255
			self.init_maskarea()
			self.display_img()


	def mouseMoveEvent(self, event):
		if self.moveMode:
			x,y=event.pos().x(),event.pos().y()
			self.tmpX,self.tmpY=x-self.startX,y-self.startY
			self.display_img()
			return
		pos_tmp = (event.pos().x(), event.pos().y())
		self.pos_xy.append(pos_tmp)
		self.update()

	def mouseReleaseEvent(self, event):
		if self.moveMode or self.tmpX!=0 or self.tmpY!=0:
			self.posX+=self.tmpX/self.width()
			self.posY+=self.tmpY/self.width()
			self.tmpX=0
			self.tmpY=0
			self.pos_xy.clear()
			self.update()
			return
		if len(self.pos_xy)<3:
			self.pos_xy.clear()
			return
		if self.src_img is None:
			self.pos_xy.clear()
			return
		self.mask[:,:]=0
		roi_corners = np.array([self.pos_xy], dtype=np.int32)
		roi_corners=roi_corners-np.array([self.posX*self.width(),self.posY*self.width()])
		roi_corners=roi_corners*(self.src_img.shape[1]/(self.width())/self.zoom)
		roi_corners=roi_corners.astype(np.int32)
		self.mask_u=min(roi_corners[0,:,1])
		self.mask_d=max(roi_corners[0,:,1])
		self.mask_l=min(roi_corners[0,:,0])
		self.mask_r=max(roi_corners[0,:,0])
		cv2.fillPoly(self.mask,roi_corners,255)
		self.pos_xy.clear()
		self.display_img()
		self.update()
