from PySide2.QtGui import QIcon
from PySide2.QtWidgets import QWidget, QComboBox, QMessageBox, QFileDialog, QPushButton, QProgressBar, QHBoxLayout, QVBoxLayout, QApplication, QLabel, QStackedLayout
from PySide2.QtCore import QThreadPool


import os
import sys
import numpy as np
from PIL import Image
import test


from torchvision.models.segmentation import lraspp_mobilenet_v3_large, fcn_resnet101, deeplabv3_mobilenet_v3_large
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure



class DustIdentifier(QWidget):
    def __init__(self):
        super(DustIdentifier, self).__init__()

        self.ICON_NAME = 'icon.png'
        self.Init_diagram = 'diagram.png'
        self.FOLDER_NAME = '/app_data'
        self.Output_FOLDER = '/Output'
        
        self.setWindowTitle("Dust Cloud Identifier V0.1")
        ICON_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)) + self.FOLDER_NAME, self.ICON_NAME)
        
        self.DIAGRAM_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)) + self.FOLDER_NAME, self.Init_diagram)
        self.setWindowIcon(QIcon(ICON_PATH))

        self.Ml_ModelList = {'DeepLabV3-MobileNet-V3-Large' : deeplabv3_mobilenet_v3_large, 'Lraspp-MobileNet-V3-Large' : lraspp_mobilenet_v3_large, 'FCN-Resnet101' : fcn_resnet101, }
        self.Ml_TrainedModels = {'FCN-Resnet101' : 'FCN-Resnet101_dataset_897.pth', 'DeepLabV3-MobileNet-V3-Large' : 'DeepLabV3-MobileNet-V3-Large_dataset_897.pth', 'Lraspp-MobileNet-V3-Large' : 'Lraspp-MobileNet-V3-Large_dataset_897.pth'}
        self.Ml_TrainedModelURLs = {'FCN-Resnet101' : 'https://drive.google.com/file/d/1GFhZq4qzGj-PXiZDlZf66goDAa_SzGMw/view?usp=share_link', 'DeepLabV3-MobileNet-V3-Large' : 'https://drive.google.com/file/d/1GEroYQmlq9HoxT-id8PyAaXLfg706GzE/view?usp=share_link', 'Lraspp-MobileNet-V3-Large' : 'https://drive.google.com/file/d/1GGO-SO9eMQdwxhA-4gYdoI0-5AmE3ryp/view?usp=share_link'}
        self.Ml_TrainedModelIDs = {'FCN-Resnet101' : '1GFhZq4qzGj-PXiZDlZf66goDAa_SzGMw', 'DeepLabV3-MobileNet-V3-Large' : '1GEroYQmlq9HoxT-id8PyAaXLfg706GzE', 'Lraspp-MobileNet-V3-Large' : '1GGO-SO9eMQdwxhA-4gYdoI0-5AmE3ryp'}
        self.parameters = {'InputPath':'', 'InputType':'', 'InputName':'', 'ModelInputSize':256, 'OutputPath':'', 'CSVPath':'', 'TrainedDataset':'URDE_dataset_100', 'Model':0, 'ModelName':'', 'ModelPath':'', 'ModelURL':'', 'TrainedModel':'', 'ModelID':''}
        

        self.threadpool = QThreadPool()

        self.initUI()
        self.showImages(Image.open(self.DIAGRAM_PATH), [], 'Process')


    def show_view(self, image):
        self.showImages(image, [], 'Identifying ..')


    def Run(self):
        worker = test.Worker(self.parameters)
        worker.Network_inst.result.connect(self.show_view)
        worker.Network_inst.cycleStart.connect(self.clearView)
        worker.Network_inst.progress.connect(self.setProgress)
        worker.Network_inst.message.connect(self.Notice)
        self.threadpool.start(worker)

    def initUI(self):
        #Layouts
        self.main_layout = QVBoxLayout()
        self.control_btn_layout = QHBoxLayout()
        self.imgView_layout = QStackedLayout()

        #create components for layout
        self.createUIcomponents()

        #set margins (left,top,right,bottom)
        self.main_layout.setContentsMargins(1,1,1,1)
        self.imgView_layout.setContentsMargins(1,1,1,1)
        self.control_btn_layout.setContentsMargins(1,1,1,1)

        #layout Widgets
        self.main_widget = QWidget()
        self.control_btn_widget = QWidget()
        self.imgView_widget = QWidget()

        #Add component widgets to button layout
        self.control_btn_layout.addWidget(self.OpenVideo_btn)
        self.control_btn_layout.addWidget(self.OpenImage_btn)
        self.control_btn_layout.addWidget(self.Model_label)
        self.control_btn_layout.addWidget(self.Model_cbx)
        self.control_btn_layout.addWidget(self.about_btn)

        #Set widget
        self.control_btn_widget.setLayout(self.control_btn_layout)

        #Set widget
        self.imgView_widget.setLayout(self.imgView_layout)


        #Add widjets to main layout
        self.main_layout.addWidget(self.control_btn_widget)
        self.main_layout.addWidget(self.InputType_label)
        self.main_layout.addWidget(self.InputPath_label)
        self.main_layout.addWidget(self.ModelPath_label)
        self.main_layout.addWidget(self.OutputPath_label)
        self.main_layout.addWidget(self.imgView_widget)
        self.main_layout.addWidget(self.Run_btn)
        self.main_layout.addWidget(self.pbar)

        #Set widget
        #self.main_widget.setLayout(self.main_layout)

        self.setLayout(self.main_layout)

        

    def createUIcomponents(self):
        #Controls
        self.OpenImage_btn = QPushButton("Load Image")
        self.OpenImage_btn.setObjectName("LOADIMAGE")
        self.OpenImage_btn.clicked.connect(self.OpenImageFile)


        self.OpenVideo_btn = QPushButton("Load Video")
        self.OpenVideo_btn.setObjectName("LOADVIDEO")
        self.OpenVideo_btn.clicked.connect(self.OpenVideoFile)

        self.Run_btn = QPushButton("RUN")
        self.Run_btn.setObjectName("RUN")
        self.Run_btn.clicked.connect(self.Run)

        self.about_btn = QPushButton("About")
        self.about_btn.setObjectName("ABOUT")
        self.about_btn.clicked.connect(self.aboutpressed)

        self.Model_label = QLabel("Select ML Model")
        self.Model_cbx = QComboBox()
        self.Model_cbx.setObjectName('MODELCOMBO')
        for key in self.Ml_ModelList.keys():
            self.Model_cbx.addItem(key)
        self.Model_cbx.activated.connect(self.updateUI)

        self.InputType_label = QLabel("File Type = {}".format(self.parameters['InputType']))
        self.InputType_label.setWordWrap(True)
        self.InputPath_label = QLabel("Input Path = {}".format(self.parameters['InputPath']))
        self.InputPath_label.setWordWrap(True)
        self.OutputPath_label = QLabel("Output Path = {}".format(self.parameters['OutputPath']))
        self.OutputPath_label.setWordWrap(True)
        self.ModelPath_label = QLabel("Model Path = {}".format(self.parameters['ModelPath']))
        self.ModelPath_label.setWordWrap(True)


        #Progress Bar
        self.pbar = QProgressBar()
        self.pbar.setObjectName('ProgressBar')
        self.pbar.setMaximum(100)


    def clearView(self):
        if self.imgView_layout is not None:
            while self.imgView_layout.count():
                item = self.imgView_layout.takeAt(0)
                widget = item.widget()
                if widget is not None:
                    widget.deleteLater()
                else:
                    self.clearView(item.layout())  #recursive
    
    def showImages(self, Input_images, Output_Image, Method_title):

        fig = Figure(figsize=(12, 12), dpi=65, facecolor=(1, 1, 1), edgecolor=(0, 0, 0))
        fig.suptitle(Method_title, fontsize=10)

        if np.shape(Input_images)[0] > 0 and np.shape(Output_Image)[0] > 0:
            a=fig.add_subplot(1, 2, 1)
            a.imshow(Input_images)

            a=fig.add_subplot(1, 2, 2)
            a.imshow(Output_Image)   

        elif np.shape(Input_images)[0] > 0:
            a=fig.add_subplot(1, 1, 1)
            a.imshow(Input_images)
   

        fig.subplots_adjust(
                    left=0.05,
                    bottom=0.05, 
                    right=0.95, 
                    top=0.95, 
                    wspace=0.1, 
                    hspace=0.1)
        

        canvas = FigureCanvas(fig)
        self.imgView_layout.addWidget(canvas)

    def updateUI(self):
        self.InputType_label.setText("File Type = {}".format(self.parameters['InputType']))
        self.InputPath_label.setText("Input Path = {}".format(self.parameters['InputPath']))
        

        self.parameters['ModelName'] = self.Model_cbx.currentText()
        self.parameters['TrainedModel'] =  self.Ml_TrainedModels[self.parameters['ModelName']]
        MODEL_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), self.parameters['TrainedModel'])
        self.parameters['ModelPath'] = MODEL_PATH
        self.parameters['ModelURL'] = self.Ml_TrainedModelURLs[self.parameters['ModelName']]
        self.parameters['ModelID'] = self.Ml_TrainedModelIDs[self.parameters['ModelName']]

        self.ModelPath_label.setText("Model Path = {}".format(self.parameters['ModelPath']))

        OUTPUT_NAME = "output_" + self.parameters['ModelName'] + "_" + self.parameters['TrainedDataset'] + "_" + os.path.splitext(self.parameters['InputName'])[0]
        OUTPUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)) + self.Output_FOLDER, OUTPUT_NAME)
        self.parameters['OutputPath'] = OUTPUT_PATH
        self.OutputPath_label.setText("Output Path = {}".format(self.parameters['OutputPath']))
        self.parameters['Model'] = self.Ml_ModelList[self.parameters['ModelName']]

        CSV_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)) + self.Output_FOLDER, OUTPUT_NAME + '.csv')
        self.parameters['CSVPath'] = CSV_PATH

    def OpenVideoFile(self):
        FILE_NAME, FILE_PATH = self.getFile()
        self.parameters['InputName'] = FILE_NAME
        self.parameters['InputType'] = 'Video'
        self.parameters['InputPath'] = FILE_PATH
        self.updateUI()


    def OpenImageFile(self):
        FILE_NAME, FILE_PATH = self.getFile()
        self.parameters['InputName'] = FILE_NAME
        self.parameters['InputType'] = 'Image'
        self.parameters['InputPath'] = FILE_PATH
        self.updateUI()


    def getFile(self):
        dlgBox = QFileDialog()
        dlgBox.setFileMode(QFileDialog.AnyFile)
        FILE_PATH,_ = dlgBox.getOpenFileName(self, 'Open File')
        FILE_NAME = FILE_PATH.split('/')[-1]
        return FILE_NAME, FILE_PATH

    def setProgress(self,value):
        self.pbar.setValue(value)


    def aboutpressed(self):
        message = QMessageBox()
        message.setWindowTitle("ABOUT")
        link = "https://github.com/RajithaRanasinghe?tab=repositories"
        message.setText('<font color="blue">Written by</font><font color="red"> RGR </font><font color="green">!</font><br><a href="url">{}</a>'.format(link))

        message.show()
        message.exec_()

    def Notice(self, notice):
        message = QMessageBox()
        message.setWindowTitle("NOTICE")
        message.setText(str(notice))

        message.show()
        message.exec_()




if __name__ == "__main__":
    sys.argv.append('--no-sandbox')
    app = QApplication(sys.argv)
    app.setStyle('Fusion')
    
    with open(os.path.join(os.path.dirname(os.path.abspath(__file__)),"style.qss"), "r") as f:
        _style = f.read()
        app.setStyleSheet(_style)
    
    QApplication.processEvents()
    widget = DustIdentifier()
    app.setApplicationName("Dust Cloud Identifier")
    widget.show()

    app.exec_()
    sys.exit()