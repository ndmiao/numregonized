# -*- coding: utf-8 -*-

import sys  # 导入系统
from PyQt5.QtWidgets import QApplication, QMainWindow, QWidget, QPushButton,QLabel,QTextEdit,QFileDialog,QHBoxLayout,QVBoxLayout,QSplitter,QComboBox,QSpinBox
from PyQt5.Qt import QWidget, QColor,QPixmap,QIcon,QSize,QCheckBox
from PyQt5 import QtCore, QtGui
from Paintboard import PaintBoard
import tensorflow as tf
import numpy as np
import tensorflow.python.keras as keras

class FirstUi(QMainWindow):  # 第一个窗口类
    def __init__(self):
        super(FirstUi, self).__init__()
        self.init_ui()

    def init_ui(self):
        self.setFixedSize(self.width(), self.height())
        self.setWindowIcon(QtGui.QIcon('./icon.jpg'))
        self.resize(640, 500)  # 设置窗口大小
        self.setWindowTitle('数字识别')  # 设置窗口标题
        self.btn = QPushButton('手写数字识别', self)  # 设置按钮和按钮名称
        self.btn.setGeometry(245, 100, 150, 50)  # 前面是按钮左上角坐标，后面是窗口大小
        self.btn.clicked.connect(self.slot_btn_function)  # 将信号连接到槽
        self.btn2 = QPushButton('数字图片识别', self)
        self.btn2.setGeometry(245, 200,150,50)
        self.btn2.clicked.connect(self.slot_btn2_function)
        self.btn_exit = QPushButton('退出', self)
        self.btn_exit.setGeometry(245, 300, 150, 50)
        self.btn_exit.clicked.connect(self.Quit)
        self.label_name = QLabel('AUTHOR: 南岛鹋', self)
        self.label_name.setGeometry(460, 410, 200, 30)
        self.label_name = QLabel('BLOG: www.ndmiao.cn', self)
        self.label_name.setGeometry(460, 440, 200, 30)

    def Quit(self):
        self.close()

    def slot_btn_function(self):
        self.hide()  # 隐藏此窗口
        self.s = write_num()  # 将第二个窗口换个名字
        self.s.show()  # 经第二个窗口显示出来

    def slot_btn2_function(self):
        self.hide()  # 隐藏此窗口
        self.s = picture_num()
        self.s.show()


class write_num(QWidget):
    def __init__(self):
        super(write_num, self).__init__()
        self.__InitData()  # 先初始化数据，再初始化界面
        self.__InitView()

    def __InitData(self):
        '''
                  初始化成员变量
        '''
        self.__paintBoard = PaintBoard(self)
        # 获取颜色列表(字符串类型)
        self.__colorList = QColor.colorNames()

    def __InitView(self):
        '''
                  初始化界面
        '''
        self.setWindowIcon(QtGui.QIcon('./icon.jpg'))
        self.resize(640, 600)
        self.setFixedSize(self.width(), self.height())
        self.setWindowTitle("手写数字识别")

        self.label_name = QLabel('哔哩哔哩大学', self)
        self.label_name.setGeometry(500, 5, 120, 30)

        self.label_name = QLabel('知识学院', self)
        self.label_name.setGeometry(500, 35, 100, 30)

        self.label_name = QLabel('野生技术协会', self)
        self.label_name.setGeometry(500, 65, 100, 30)

        self.label_name = QLabel('南岛鹋', self)
        self.label_name.setGeometry(500, 95, 100, 30)

        self.edit = QTextEdit(self)
        self.edit.setGeometry(510, 160, 110, 60)

        # 新建一个水平布局作为本窗体的主布局
        main_layout = QHBoxLayout(self)
        # 设置主布局内边距以及控件间距为10px
        main_layout.setSpacing(10)

        # 在主界面左侧放置画板
        main_layout.addWidget(self.__paintBoard)

        # 新建垂直子布局用于放置按键
        sub_layout = QVBoxLayout()

        # 设置此子布局和内部控件的间距为5px
        sub_layout.setContentsMargins(5, 5, 5, 5)

        splitter = QSplitter(self)  # 占位符
        sub_layout.addWidget(splitter)

        self.__btn_Recognize = QPushButton("开始识别")
        self.__btn_Recognize.setParent(self)
        self.__btn_Recognize.clicked.connect(self.on_btn_Recognize_Clicked)
        sub_layout.addWidget(self.__btn_Recognize)

        self.__btn_Clear = QPushButton("清空画板")
        self.__btn_Clear.setParent(self)  # 设置父对象为本界面
        # 将按键按下信号与画板清空函数相关联
        self.__btn_Clear.clicked.connect(self.__paintBoard.Clear)
        sub_layout.addWidget(self.__btn_Clear)

        self.__btn_return = QPushButton("返回")
        self.__btn_return.setParent(self)  # 设置父对象为本界面
        self.__btn_return.clicked.connect(self.slot_btn_function)
        sub_layout.addWidget(self.__btn_return)

        self.__btn_Quit = QPushButton("退出")
        self.__btn_Quit.setParent(self)  # 设置父对象为本界面
        self.__btn_Quit.clicked.connect(self.Quit)
        sub_layout.addWidget(self.__btn_Quit)

        self.__btn_Save = QPushButton("保存作品")
        self.__btn_Save.setParent(self)
        self.__btn_Save.clicked.connect(self.on_btn_Save_Clicked)
        sub_layout.addWidget(self.__btn_Save)

        self.__cbtn_Eraser = QCheckBox("使用橡皮擦")
        self.__cbtn_Eraser.setParent(self)
        self.__cbtn_Eraser.clicked.connect(self.on_cbtn_Eraser_clicked)
        sub_layout.addWidget(self.__cbtn_Eraser)

        self.__label_penThickness = QLabel(self)
        self.__label_penThickness.setText("画笔粗细")
        self.__label_penThickness.setFixedHeight(20)
        sub_layout.addWidget(self.__label_penThickness)

        self.__spinBox_penThickness = QSpinBox(self)
        self.__spinBox_penThickness.setMaximum(20)
        self.__spinBox_penThickness.setMinimum(2)
        self.__spinBox_penThickness.setValue(10)  # 默认粗细为10
        self.__spinBox_penThickness.setSingleStep(2)  # 最小变化值为2
        self.__spinBox_penThickness.valueChanged.connect(self.on_PenThicknessChange)  # 关联spinBox值变化信号和函数on_PenThicknessChange
        sub_layout.addWidget(self.__spinBox_penThickness)

        self.__label_penColor = QLabel(self)
        self.__label_penColor.setText("画笔颜色")
        self.__label_penColor.setFixedHeight(20)
        sub_layout.addWidget(self.__label_penColor)

        self.__comboBox_penColor = QComboBox(self)
        self.__fillColorList(self.__comboBox_penColor)  # 用各种颜色填充下拉列表
        self.__comboBox_penColor.currentIndexChanged.connect(self.on_PenColorChange)  # 关联下拉列表的当前索引变更信号与函数on_PenColorChange
        sub_layout.addWidget(self.__comboBox_penColor)

        main_layout.addLayout(sub_layout)  # 将子布局加入主布局

    def __fillColorList(self, comboBox):

        index_black = 0
        index = 0
        for color in self.__colorList:
            if color == "black":
                index_black = index
            index += 1
            pix = QPixmap(70, 20)
            pix.fill(QColor(color))
            comboBox.addItem(QIcon(pix), None)
            comboBox.setIconSize(QSize(70, 20))
            comboBox.setSizeAdjustPolicy(QComboBox.AdjustToContents)

        comboBox.setCurrentIndex(index_black)

    def on_PenColorChange(self):
        color_index = self.__comboBox_penColor.currentIndex()
        color_str = self.__colorList[color_index]
        self.__paintBoard.ChangePenColor(color_str)

    def on_PenThicknessChange(self):
        penThickness = self.__spinBox_penThickness.value()
        self.__paintBoard.ChangePenThickness(penThickness)

    def on_btn_Save_Clicked(self):
        savePath = QFileDialog.getSaveFileName(self, 'Save Your Paint', '.\\', '*.png')
        print(savePath)
        if savePath[0] == "":
            print("Save cancel")
            return
        image = self.__paintBoard.GetContentAsQImage()
        image.save(savePath[0])
        print(savePath[0])

    def Quit(self):
        self.close()

    def on_cbtn_Eraser_clicked(self):
        if self.__cbtn_Eraser.isChecked():
            self.__paintBoard.EraserMode = True  # 进入橡皮擦模式
        else:
            self.__paintBoard.EraserMode = False  # 退出橡皮擦模式

    def on_btn_Recognize_Clicked(self):
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
        savePath = "./text.png"
        image = self.__paintBoard.GetContentAsQImage()
        image.save(savePath)
        print(savePath)
        # 加载图像
        img = keras.preprocessing.image.load_img(savePath, target_size=(28, 28))
        img = img.convert('L')
        x = keras.preprocessing.image.img_to_array(img)
        x = abs(255-x)
        #x = x.reshape(28,28)
        x = np.expand_dims(x, axis=0)
        x=x/255.0
        new_model = keras.models.load_model('./my_model.h5')
        prediction = new_model.predict(x)
        output = np.argmax(prediction, axis=1)
        #print("手写数字识别为：" + str(output[0]))
        self.edit.setText('识别的手写数字为:' + str(output[0]))

    def slot_btn_function(self):
        self.hide()  # 隐藏此窗口
        self.f = FirstUi()  # 将第一个窗口换个名字
        self.f.show()  # 将第一个窗口显示出来



class picture_num(QWidget):
    def __init__(self):
        super(picture_num, self).__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowIcon(QtGui.QIcon('./icon.jpg'))
        self.resize(640,520)
        self.setFixedSize(self.width(), self.height())
        self.setWindowTitle('图片数字识别')
        self.label_name1 = QLabel('哔哩哔哩大学', self)
        self.label_name1.setGeometry(500, 20, 120, 35)

        self.label_name2 = QLabel('知识学院', self)
        self.label_name2.setGeometry(500, 60, 100, 35)

        self.label_name3 = QLabel('野生技术协会', self)
        self.label_name3.setGeometry(500, 100, 100, 35)

        self.label_name4 = QLabel('南岛鹋', self)
        self.label_name4.setGeometry(500, 140, 100, 35)

        self.label_name5 = QLabel('待载入图片', self)
        self.label_name5.setGeometry(10, 20, 480, 480)
        self.label_name5.setStyleSheet("QLabel{background:gray;}"
                                 "QLabel{color:rgb(0,0,0,120);font-size:15px;font-weight:bold;font-family:宋体;}"
                                 )
        self.label_name5.setAlignment(QtCore.Qt.AlignCenter)
        
        self.edit = QTextEdit(self)
        self.edit.setGeometry(500, 220, 100, 60)

        self.btn_select = QPushButton('选择图片',self)
        self.btn_select.setGeometry(500, 320, 100, 30)
        self.btn_select.clicked.connect(self.select_image)

        self.btn_dis = QPushButton('识别图片',self)
        self.btn_dis.setGeometry(500, 370, 100, 30)
        self.btn_dis.clicked.connect(self.on_btn_Recognize_Clicked)

        self.btn = QPushButton('返回',self)
        self.btn.setGeometry(500, 420, 100, 30)
        self.btn.clicked.connect(self.slot_btn_function)

        self.btn_exit = QPushButton('退出',self)
        self.btn_exit.setGeometry(500, 470, 100, 30)
        self.btn_exit.clicked.connect(self.Quit)

    def select_image(self):
        global fname
        imgName, imgType = QFileDialog.getOpenFileName(self, "打开图片", "", "*.png;;*.jpg;;All Files(*)")
        jpg = QtGui.QPixmap(imgName).scaled(self.label_name5.width(), self.label_name5.height())
        self.label_name5.setPixmap(jpg)
        fname = imgName
        print(fname)

    def on_btn_Recognize_Clicked(self):
        global fname
        config = tf.compat.v1.ConfigProto(allow_soft_placement=True)
        config.gpu_options.per_process_gpu_memory_fraction = 0.3
        tf.compat.v1.keras.backend.set_session(tf.compat.v1.Session(config=config))
        savePath = fname
        # 加载图像
        img = keras.preprocessing.image.load_img(savePath, target_size=(28, 28))
        img = img.convert('L')
        x = keras.preprocessing.image.img_to_array(img)
        x = abs(255-x)
        #x = x.reshape(28,28)
        x = np.expand_dims(x, axis=0)
        x=x/255.0
        new_model = keras.models.load_model('./my_model.h5')
        prediction = new_model.predict(x)
        output = np.argmax(prediction, axis=1)
        self.edit.setText('识别的手写数字为:' + str(output[0]))
        
    def Quit(self):
        self.close()

    def slot_btn_function(self):
        self.hide()
        self.f = FirstUi()
        self.f.show()


def main():
    app = QApplication(sys.argv)
    w = FirstUi()  # 将第一和窗口换个名字
    w.show()  # 将第一和窗口换个名字显示出来
    sys.exit(app.exec_())  # app.exet_()是指程序一直循环运行直到主窗口被关闭终止进程（如果没有这句话，程序运行时会一闪而过）


if __name__ == '__main__':  # 只有在本py文件中才能用，被调用就不执行
    main()