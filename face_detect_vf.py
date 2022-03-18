import sys
import os
import json
import numpy as np
import cv2
import threading
#from deepface import DeepFace
import time
from matplotlib import pyplot as plt

from PyQt5.QtCore import Qt, QPoint, QSize
from PyQt5.QtWidgets import QApplication
from PyQt5.QtWidgets import QLabel, QComboBox, QWidget, QPushButton, QCheckBox
from PyQt5.QtWidgets import QMainWindow, QFileDialog, QShortcut, QMessageBox

from PyQt5.QtGui import QFont, QPixmap, QKeySequence, QImage

from rmn import RMN

class Window(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle('ICE')
        self.setGeometry(10,40,1024,550)
        self.canW = 640
        self.canH = 480
        self.is_cam_on = False
        self.cap_obj = None
        self.cam_thread = None
        self.emotion_classifier = RMN()
        
        self.container = QWidget()
        self.setCentralWidget(self.container)
        
        calibri = QFont()
        calibri.setPointSize(14)
        calibri.setItalic(True)
        calibri.setFamily("Calibri")
        
        cambria = QFont()
        cambria.setPointSize(16)
        cambria.setFamily("Segoe UI")
        
        self.start_btn = QPushButton(text="Start", parent=self.container)
        self.start_btn.setFixedSize(150, 40)
        self.start_btn.move(10, 5)
        #self.start_btn.setStyleSheet('QPushButton::hover {background: #40c0ff;}')
        self.start_btn.setFont(calibri)
        self.start_btn.clicked.connect(self.start_cam)

        self.canvas = QLabel(parent=self.container)
        self.canvas.move(10, 50)
        self.canvas.setStyleSheet('background: #cecece')
        self.canvas.setFixedSize(self.canW, self.canH)
        self.canvas.setAlignment(Qt.AlignLeft)
        
        self.info = QLabel(parent=self.container)
        self.info.move(670, 50)
        self.info.setStyleSheet('background: #404040; color: #0080ff')
        self.info.setFont(cambria)
        self.info.setFixedSize(340, 480)
        self.info.setAlignment(Qt.AlignLeft)
    
    def start_cam(self):
        try:
            if not self.is_cam_on:
                self.cap_obj = cv2.VideoCapture('360p.mp4') #cv2.VideoCapture(0)
                self.is_cam_on = True
                self.start_btn.setText("Stop")
                self.cam_thread = threading.Thread(target=self.detect_faces)
                self.cam_thread.start()
            else:
                self.is_cam_on = False
                self.start_btn.setText("Start")
            
        except BaseException as err:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText(str(err))
            msg.setWindowTitle("Error")
            retval = msg.exec_()

    def detect_faces(self):
        emo_graph = []
        
        try:
            while(self.is_cam_on):
                pos_emotions = 0
                neg_emotions = 0
                neut_emo = 0
            
                if self.cap_obj != None:
                    ret, img = self.cap_obj.read()
                    
                    if ret == False:
                        msg = QMessageBox()
                        msg.setIcon(QMessageBox.Critical)
                        msg.setText('Cannot read the image from camera, exiting ....')
                        msg.setWindowTitle("Error")
                        retval = msg.exec_()
                        break
                
                    img = cv2.resize(img, (self.canW,self.canH), interpolation = cv2.INTER_LINEAR)
                    #img = cv2.flip(img, 1)
                    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                    
                    cimg = img[:,:,::-1] #cimg[:,:,::-1]
                    res = self.emotion_classifier.detect_emotion_for_single_frame(cimg)

                    for rr in res:
                        if rr['emo_label'] in ['happy', 'surprise']:
                            pos_emotions += 1
                        elif rr['emo_label'] == 'neutral':
                            neut_emo += 1
                        else:
                            neg_emotions += 1
                        

                        if rr['emo_label'] in ['fear', 'disgust', 'angry']:
                            emo = 'confused'
                        elif rr['emo_label'] == 'sad':
                            emo = 'bored'
                        elif rr['emo_label'] == 'surprise':
                            emo = 'happy'
                        else:
                            emo = rr['emo_label']
                        
                        if emo in ['confused','bored']:
                            cv2.rectangle(img, (rr['xmin'], rr['ymin']), (rr['xmax'], rr['ymax']), (0,0,255), 2)
                            img = cv2.putText(img, emo, (rr['xmin'], rr['ymin']-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2, cv2.LINE_AA)
                        elif emo == 'neutral':
                            cv2.rectangle(img, (rr['xmin'], rr['ymin']), (rr['xmax'], rr['ymax']), (220,220,220), 2)
                            img = cv2.putText(img, emo, (rr['xmin'], rr['ymin']-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (220,220,220), 2, cv2.LINE_AA)
                        else:
                            cv2.rectangle(img, (rr['xmin'], rr['ymin']), (rr['xmax'], rr['ymax']), (0,255,0), 2)
                            img = cv2.putText(img, emo, (rr['xmin'], rr['ymin']-10),
                                          cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,255,0), 2, cv2.LINE_AA)
    
                    time.sleep(0.1)
                    self.info.setText('  Positive\t\tNegative \n\n    {0}\t\t    {1}'.format(pos_emotions, neg_emotions))
                    
                    self.set_pixmap(img)

                    if pos_emotions >= neg_emotions and pos_emotions >= neut_emo:
                        emo_graph.append(2)
                    elif neut_emo > pos_emotions and neut_emo > neg_emotions:
                        emo_graph.append(1)
                    else:
                        emo_graph.append(0)
                
                
        except BaseException as err:
            msg = QMessageBox()
            msg.setIcon(QMessageBox.Critical)
            msg.setText(str(err))
            msg.setWindowTitle("Error")
            retval = msg.exec_()
        finally:
            self.cap_obj.release()
            self.is_cam_on = False
            self.start_btn.setText("Start")


            font_lab = {'family': 'serif',
                        'color':  'maroon',
                        'weight': 'normal',
                        'size': 14,
                        }
            font_title = {'family': 'serif',
                        'color':  '#0040a0',
                        'weight': 'bold',
                        'size': 16,
                        }
            x = np.arange(len(emo_graph[25:80]))
            y = np.array(emo_graph[25:80])
            fig = plt.figure(figsize=(15, 6))
            plt.xticks(np.arange(0, len(emo_graph[25:80]), step=12))
            plt.yticks([0,1,2], ['Negative', 'Neutral', 'Positive'])
            plt.xlabel('time (s)', fontdict=font_lab)
            plt.ylabel('Student\'s Emotions', fontdict=font_lab)
            plt.title('Lecture Analysis', fontdict=font_title)
            plt.grid(alpha=0.3, c='gray', linestyle='--')
            plt.scatter(x[y==1], y[y==1], c='b', marker='o')
            plt.scatter(x[y==2], y[y==2], c='g', marker='o')
            plt.scatter(x[y==0], y[y==0], c='r', marker='o')
            plt.savefig('./graph_emo.png')

    
    def set_pixmap(self, img):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channels = img.shape
        bytesPerLine = 3 * width
        qimg = QImage(img, width, height, bytesPerLine, QImage.Format_RGB888)
        self.canvas.setPixmap(QPixmap(qimg))
        
	
if __name__ == '__main__':
    app = QApplication(sys.argv)
    win = Window()
    win.show()
    sys.exit(app.exec_())