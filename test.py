from PySide2.QtCore import QRunnable, Slot, QObject, Signal
from torchvision.transforms import transforms
import requests
import torch
import cv2
from PIL import Image
import numpy as np
import csv




class Worker(QRunnable):


    def __init__(self, parameters):
        super(Worker, self).__init__()
        # Store constructor arguments (re-used for processing)
        self.Network_inst = Network(parameters)

    @Slot()  # QtCore.Slot
    def run(self):
         self.Network_inst.run()


class Network(QObject):
    progress = Signal(float)
    result = Signal(object)
    cycleStart = Signal()
    message = Signal(str)

    def __init__(self, parameters):
        super(Network, self).__init__()
        self.MyParameters = parameters


    def run(self):
        try:
            model = self.MyParameters['Model'](weights=None, progress=True, num_classes=1, aux_loss=None)
            
        except:
            model = self.MyParameters['Model'](pretrained=False, progress=True, num_classes=1, aux_loss=None)
            
        
        
        model.eval()
        try:
            model.cuda()
        except:
            model.cpu()


        self.model_name = self.MyParameters['ModelName']
        self.dataset_name = self.MyParameters['TrainedDataset']
        self.model_path = self.MyParameters['ModelPath']
        self.input_name = self.MyParameters['InputName']
        self.model_url = self.MyParameters['ModelURL']
        self.trainedModel_name = self.MyParameters['TrainedModel']
        self.output_path = self.MyParameters['OutputPath']
        self.input_type = self.MyParameters['InputType']
        self.input_path = self.MyParameters['InputPath']
        self.csv_path = self.MyParameters['CSVPath']
        self.model_id = self.MyParameters['ModelID']
        self.model_available = False


        csv_header = ['File Name', 'Model Name', 'Frame', 'Dust Pixel Ratio', 'PM30', 'PM10', 'PM4', 'PM2.5', 'PM1']

        print(self.csv_path)
        csv_file = open(self.csv_path, 'w', newline='')
        writer = csv.writer(csv_file)
        writer.writerow(csv_header)


        if torch.cuda.is_available():

            model.load_state_dict(torch.load(self.model_path))
        else:

            model.load_state_dict(torch.load(self.model_path, map_location='cpu'))

        self.model_available = True

        try:
            model.load_state_dict(torch.load(self.model_path))
            self.model_available = True
        except:
            self.message.emit('Model Needs to be Downloaded')
            #Downloading seems not working
            '''
            response = requests.get(self.model_url, stream=True)
            with open(self.trainedModel_name, "wb") as f:
                for chunk in response.iter_content(chunk_size= 10):
                    f.write(chunk)

            #self.download_file_from_google_drive(self.model_id, '/Output')
            model.load_state_dict(torch.load(self.model_path))
            self.model_available = True
            '''
        

        if self.input_type == 'Video' and self.model_available:

            vs = cv2.VideoCapture(self.input_path)
            _, frame = vs.read()
            H, W, _ = frame.shape
            vs.release()

            fourcc = cv2.VideoWriter_fourcc('m','p','4','v')
            out = cv2.VideoWriter(self.output_path, fourcc, 24, (W, H), True)

            cap = cv2.VideoCapture(self.input_path)
            totalframecount= int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            idx = 0
            img_transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

            dust_array =[]

            while True:
                ret, frame = cap.read()
                if ret == False:
                    cap.release()
                    out.release()
                    break

                font = cv2.FONT_HERSHEY_SIMPLEX
                H, W, _ = frame.shape
                ori_frame = frame
                ori_frame1 = Image.fromarray(ori_frame)
                frame = img_transform(ori_frame1)
                input = frame.unsqueeze(0).float()
                input = input.cuda()
                mask = torch.sigmoid(model(input)['out'])
                mask = mask.squeeze(0).squeeze(0) 

                #Dust Probability clip   
                #mask = mask > 0.99


                mask = mask.detach().cpu().numpy()       
                mask = mask.astype(np.float32)
                mask = cv2.resize(mask, (W, H))
                mask = np.expand_dims(mask, axis=-1)

                
                dust_array.append(np.sum(mask))

                invmask = mask.copy()
                invmask = abs(mask - 1)
                combine_frame = ori_frame

                combine_frame[:,:,1] = combine_frame[:,:,1] * invmask[:,:,0]
                combine_frame = combine_frame.astype(np.uint8)
                total_pixels = (combine_frame.shape[0] * combine_frame.shape[1])

                if len(dust_array) > 30:
                    dust_pixel = (np.sum(dust_array[-31:-1]))/30
                else:
                    dust_pixel = (np.sum(dust_array))/len(dust_array)

                #print('total = {:.1f} dust = {:.1f}'.format(total_pixels,dust_pixel))
                dustRatio = (dust_pixel/total_pixels)*100
                if dustRatio > 0.09:
                    pm30 = 0.675 * (dustRatio) + 2.7967
                    pm10 = 0.6764 * (dustRatio) + 2.0064
                    pm4 = 0.5385 * (dustRatio) + 1.1539
                    pm2_5 = 0.4709 * (dustRatio) + 0.9569
                    pm1 = 0.4501 * (dustRatio) + 0.9037
                else:
                    pm30 = 0
                    pm10 = 0
                    pm4 = 0
                    pm2_5 = 0
                    pm1 = 0


                csv_data = [self.input_name, self.model_name, idx, dustRatio, pm30, pm10, pm4, pm2_5, pm1]
                writer.writerow(csv_data)

                dust_data = ['PM30 = {:.1f}'.format(pm30), 'PM10 = {:.1f}'.format(pm10), 'PM4 = {:.1f}'.format(pm4),'PM2.5 = {:.1f}'.format(pm2_5),'PM1 = {:.1f}'.format(pm1)]
                
                i = 0
                for data in dust_data:
                    cv2.putText(combine_frame, 
                        data,
                        (50, 50 +i), 
                        font, 1, 
                        (10, 10, 255-i),
                        4, 
                        cv2.LINE_4)
                    i+= 50
                
                model_data = ['Model = {}'.format(self.model_name), self.dataset_name, 'Dust Pixel Ratio% = {:.1f}'.format(dustRatio)]
                
                for data in model_data:
                    cv2.putText(combine_frame, 
                        data,
                        (50, 50 +i), 
                        font, 1, 
                        (255, 0, 255),
                        4, 
                        cv2.LINE_4)
                    i+= 50

                self.cycleStart.emit()
                self.result.emit(cv2.cvtColor(combine_frame, cv2.COLOR_BGR2RGB))
                idx += 1
                pgr = (idx / totalframecount) * 100
                self.progress.emit(pgr)


                out.write(combine_frame)
            print('All Done')
            cap.release()


        if self.input_type == 'Image' and self.model_available:
            totalframecount = 1
            idx = 0
            font = cv2.FONT_HERSHEY_SIMPLEX
            img_transform = transforms.Compose([
                transforms.Resize((256,256)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])

            dust_array =[]

            ori_frame = Image.open(self.input_path)
            (H, W, _) = np.shape(ori_frame)       
            frame = img_transform(ori_frame)
            input = frame.unsqueeze(0).float()
            input = input.cuda()
            mask = torch.sigmoid(model(input)['out'])
            mask = mask.squeeze(0).squeeze(0) 

            #Dust Probability clip   
            #mask = mask > 0.99

            mask = mask.detach().cpu().numpy()       
            mask = mask.astype(np.float32)
            mask = cv2.resize(mask, (W, H))
            mask = np.expand_dims(mask, axis=-1)
          
            dust_array.append(np.sum(mask))

            invmask = mask.copy()
            invmask = abs(mask - 1)
            combine_frame = np.asanyarray(ori_frame).copy()

            print(type(ori_frame))

            combine_frame[:,:,1] = combine_frame[:,:,1] * invmask[:,:,0]
            combine_frame = combine_frame.astype(np.uint8)
            total_pixels = (combine_frame.shape[0] * combine_frame.shape[1])

            if len(dust_array) > 30:
                dust_pixel = (np.sum(dust_array[-31:-1]))/30
            else:
                dust_pixel = (np.sum(dust_array))/len(dust_array)

            #print('total = {:.1f} dust = {:.1f}'.format(total_pixels,dust_pixel))
            dustRatio = (dust_pixel/total_pixels)*100
            if dustRatio > 0.09:
                pm30 = 0.675 * (dustRatio) + 2.7967
                pm10 = 0.6764 * (dustRatio) + 2.0064
                pm4 = 0.5385 * (dustRatio) + 1.1539
                pm2_5 = 0.4709 * (dustRatio) + 0.9569
                pm1 = 0.4501 * (dustRatio) + 0.9037
            else:
                pm30 = 0
                pm10 = 0
                pm4 = 0
                pm2_5 = 0
                pm1 = 0


            csv_data = [self.input_name, self.model_name, idx, dustRatio, pm30, pm10, pm4, pm2_5, pm1]
            writer.writerow(csv_data)

            dust_data = ['PM30 = {:.1f}'.format(pm30), 'PM10 = {:.1f}'.format(pm10), 'PM4 = {:.1f}'.format(pm4),'PM2.5 = {:.1f}'.format(pm2_5),'PM1 = {:.1f}'.format(pm1)]

            '''           
            i = 0
            for data in dust_data:
                cv2.putText(combine_frame, 
                    data,
                    (480, 50 +i), 
                    font, 1, 
                    (10, 10, 255-i),
                    4, 
                    cv2.LINE_4)
                i+= 50
            
            model_data = ['Model = {}'.format(self.model_name), self.dataset_name, 'Dust Pixel Ratio% = {:.1f}'.format(dustRatio)]
            
            i = 0
            for data in model_data:
                cv2.putText(combine_frame, 
                    data,
                    (50, 50 +i), 
                    font, 1, 
                    (255, 0, 255),
                    4, 
                    cv2.LINE_4)
                i+= 50

            ''' 

            self.cycleStart.emit()
            self.result.emit(combine_frame)
            idx += 1
            pgr = (idx / totalframecount) * 100
            self.progress.emit(pgr)

            im = Image.fromarray(combine_frame)
            im.save(self.output_path)

            print('All Done')



        csv_file.close()


    def download_file_from_google_drive(self, id, destination):
        URL = "https://docs.google.com/uc?export=download"

        session = requests.Session()

        response = session.get(URL, params = { 'id' : id }, stream = True)
        token = self.get_confirm_token(response)

        if token:
            params = { 'id' : id, 'confirm' : token }
            response = session.get(URL, params = params, stream = True)

        self.save_response_content(response, destination)    

    def get_confirm_token(self, response):
        for key, value in response.cookies.items():
            if key.startswith('download_warning'):
                return value

        return None

    def save_response_content(self, response, destination):
        CHUNK_SIZE = 32768

        with open(destination, "wb") as f:
            for chunk in response.iter_content(CHUNK_SIZE):
                if chunk: # filter out keep-alive new chunks
                    f.write(chunk)


        

        