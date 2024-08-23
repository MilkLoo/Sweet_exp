# import sys

import cv2
import numpy as np
# import time
# import threading

# from PyQt5 import QtCore
# from PyQt5.QtGui import *
# from PyQt5.QtCore import *
# from PyQt5.QtWidgets import *
import os
import json
import sys
# from PyQt5 import QtWidgets, QtCore, QtGui
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from functools import partial
import os
from MainWindow import Ui_Annotator
# import time
# import shutil
# import sip
import copy


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in ['.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG'])


class Main(QMainWindow, Ui_Annotator):
    def __init__(self):
        super(Main, self).__init__()
        self.setupUi(self)

        ##自适应屏幕大小
        self.desktop = QApplication.desktop()
        self.screenRect = self.desktop.screenGeometry()
        self.screenheight = self.screenRect.height()
        self.screenwidth = self.screenRect.width()
        self.height = int(self.screenheight * 0.9)
        self.width = int(self.screenwidth * 0.85)

        self.resize(self.width, self.height)

        self.open_file.clicked.connect(self.openimage)

        self.currentID = -1
        self.pic_path_list = []
        self.jsonlist = []
        self.imagelist = None
        self.imagepath = None
        self.scaleratio = 1.0
        self.keypoints = []
        self.cur_keypoint = np.zeros((17, 3), dtype=np.float32)
        self.cur_partID = 0
        self.cur_vis = True

        self.pos = None
        self.png = None
        self.str_ = None
        self.cur_part_keypoint = None
        self.nextImageAction = QAction("&NextImage", self)
        self.nextImageAction.setShortcut("D")
        self.nextImageAction.triggered.connect(partial(self.nextImage, +1))

        self.preImageAction = QAction("&PreImage", self)
        self.preImageAction.setShortcut("A")
        self.preImageAction.triggered.connect(partial(self.nextImage, -1))

        self.nextPartAction = QAction("&NextPart", self)
        self.nextPartAction.setShortcut("W")
        self.nextPartAction.triggered.connect(self.nextPart)

        self.mainMenu = self.menuBar()
        self.mainMenu.addAction(self.nextImageAction)
        self.mainMenu.addAction(self.preImageAction)
        self.mainMenu.addAction(self.nextPartAction)

        self.label_maxw = int(0.8 * self.width)

        self.label_maxh = int(0.85 * self.height)

        self.lastPoint = None
        self.pen_vis = QPen()
        self.pen_vis.setWidth(5)
        self.pen_vis.setBrush(Qt.red)
        self.pen_invis = QPen()
        self.pen_invis.setWidth(5)
        self.pen_invis.setBrush(Qt.green)
        self.pen_att = QPen()
        self.pen_att.setWidth(10)
        self.pen_att.setBrush(Qt.red)

        self.buttonlist = [self.nose, self.left_eye, self.right_eye, self.left_ear,
                           self.right_ear, self.left_shoulder, self.right_shoulder,
                           self.left_elbow, self.right_elbow, self.left_wrist,
                           self.right_wrist, self.left_hip, self.right_hip,
                           self.left_knee, self.right_knee, self.left_ankle, self.right_ankle]

        for i in range(17):
            button = self.buttonlist[i]
            button.clicked.connect(partial(self.changePart, i))

        self.buttonlist[0].setChecked(True)
        self.buttonlist[0].setStyleSheet("background-color: red")

    def newItem(self):
        self.keypoints = []
        self.cur_keypoint[:, 0:2] = self.cur_keypoint[:, 0:2] / self.scaleratio
        self.keypoints.append(copy.deepcopy(self.cur_keypoint.reshape(-1).tolist()))
        self.cur_keypoint = np.zeros((17, 3), dtype=np.float32)
        self.cur_partID = 0
        self.cur_vis = True

    def savejson(self):
        if self.imagepath:
            if np.sum(self.cur_keypoint) >= 0:
                self.newItem()
            if len(self.keypoints) > 0:
                res = {'imagepath': self.imagepath, 'scaleratio': self.scaleratio, 'keypoints': self.keypoints}
                savepath = os.path.splitext(self.currentPath)[0] + '.json'

                with open(savepath, 'w') as f:
                    json.dump(res, f)

    def print(self, log):
        print('-------------- < %s > ------------' % log)
        print('imagepath:', self.imagepath)
        print('scaleratio:', self.scaleratio)
        print('keypoints:', self.keypoints)
        print('cur_keypoint:', self.cur_keypoint)
        print('cur_partID:', self.cur_partID)
        print('cur_vis:', self.cur_vis)

    def openimage(self):
        self.file_path = QFileDialog.getExistingDirectory(None, "请选择文件路径")
        file_list = os.listdir(self.file_path)
        self.imagelist = [item for item in file_list if is_image_file(item)]
        self.jsonlist = [item for item in file_list if item.endswith('.json')]

        self.currentID = 0
        self.update()
        # for i in file_list:
        #     if is_image_file(i):
        #         pic_path = self.file_path + '/' + i
        #         self.pic_path_list.append((pic_path))

    def loadimg(self, filename):

        self.pos = None
        png = QPixmap(filename)

        ratio = min(self.label_maxw / png.width(), self.label_maxh / png.height())
        self.png = png.scaled(png.width() * ratio, png.height() * ratio)
        self.update()
        ##——修改 添加默认图像

    def paintEvent(self, e):
        qp = QPainter()

        qp.begin(self)
        if self.png:

            qp.drawPixmap(0, 0, self.png)

            self.cur_part_keypoint = self.cur_keypoint[self.cur_partID]

            qp.setPen(self.pen_att)
            qp.drawPoint(int(self.cur_part_keypoint[0]), int(self.cur_part_keypoint[1]))

            for keypoint in self.cur_keypoint:

                if keypoint[2] == 2:
                    qp.setPen(self.pen_vis)
                    qp.drawPoint(keypoint[0], keypoint[1])
                elif keypoint[2] == 1:
                    qp.setPen(self.pen_invis)
                    qp.drawPoint(int(keypoint[0]), int(keypoint[1]))

            qp.end()

    def mousePressEvent(self, e):
        if e.button() == Qt.LeftButton:
            self.pos = e.pos()
            # self.lastPoint = e.pos()
            self.update()

            if self.imagepath:
                self.cur_keypoint[self.cur_partID, 0] = self.pos.x()
                self.cur_keypoint[self.cur_partID, 1] = self.pos.y()
                self.cur_keypoint[self.cur_partID, 2] = 2

        if e.button() == Qt.RightButton:
            self.pos = e.pos()
            self.lastPoint = e.pos()
            self.update()

            if self.imagepath:
                self.cur_keypoint[self.cur_partID, 0] = self.pos.x()
                self.cur_keypoint[self.cur_partID, 1] = self.pos.y()
                self.cur_keypoint[self.cur_partID, 2] = 1

    def nextImage(self, direction):

        self.savejson()

        self.cur_partID = 0
        self.currentID += direction

        self.currentID = min(max(self.currentID, 0), len(self.imagelist) - 1)
        self.currentPath = '%s/%s' % (self.file_path, self.imagelist[self.currentID])

        img = cv2.imread(self.currentPath)
        height, width, bytesPerComponent = img.shape

        bytesPerLine = 3 * width
        cv2.cvtColor(img, cv2.COLOR_BGR2RGB, img)
        QImg = QImage(img.data, width, height, bytesPerLine, QImage.Format_RGB888)

        png = QPixmap.fromImage(QImg)

        # png = QPixmap(self.currentPath)

        ratio = min(self.label_maxw / png.width(), self.label_maxh / png.height())

        transform = QTransform()
        transform.rotate(90)

        # if png.width()<png.height():
        #     png = png.transformed(transform)
        # else:pass

        self.png = png.scaled(int(png.width() * ratio), int(png.height() * ratio))
        self.update()
        self.imagepath = self.currentPath
        self.scaleratio = ratio

        ##读取json
        file_list = os.listdir(self.file_path)
        self.jsonlist = [item for item in file_list if item.endswith('.json')]

        if self.imagelist[self.currentID]:
            current_img_name = self.imagelist[self.currentID].split(".")[0]

            current_json_name = current_img_name + '.json'
            if current_json_name in self.jsonlist:
                current_json_path = os.path.join(self.file_path, current_json_name)
                with open(current_json_path, 'r', encoding='utf-8') as f:
                    current_keypoints = json.load(f)['keypoints']
                    current_keypoints = np.array(current_keypoints).reshape(-1, 3)
                    current_keypoints[:, 0:2] = current_keypoints[:, 0:2] * self.scaleratio

                    self.cur_keypoint = current_keypoints

        else:
            pass

        self.buttonlist[self.cur_partID].setChecked(True)
        for bt in self.buttonlist:
            bt.setStyleSheet("background-color: None")
        self.buttonlist[self.cur_partID].setStyleSheet("background-color: red")

        self.str_ = '\n'.join(
            str(i) for i in self.cur_keypoint) + '\n' + f'------{self.currentID}/{len(self.imagelist)}'

        self.label.setText(self.str_)

    def nextPart(self):

        self.cur_partID += 1
        self.cur_partID = self.cur_partID % 17
        self.cur_vis = True
        self.buttonlist[self.cur_partID].setChecked(True)
        for bt in self.buttonlist:
            bt.setStyleSheet("background-color: None")
        self.buttonlist[self.cur_partID].setStyleSheet("background-color: red")

        self.str_ = '\n'.join(
            str(i) for i in self.cur_keypoint) + '\n' + f'------{self.currentID}/{len(self.imagelist)}'

        self.label.setText(self.str_)
        self.update()

    def changePart(self, id):

        self.cur_partID = id
        self.cur_vis = True
        self.buttonlist[self.cur_partID].setChecked(True)
        for bt in self.buttonlist:
            bt.setStyleSheet("background-color: None")
        self.buttonlist[self.cur_partID].setStyleSheet("background-color: red")

        self.str_ = '\n'.join(str(i) for i in self.cur_keypoint)

        self.label.setText(self.str_)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = Main()
    win.show()
    sys.exit(app.exec_())
