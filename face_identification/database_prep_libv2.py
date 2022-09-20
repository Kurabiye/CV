from base64 import decode
import cv2
import numpy as np
from deepface import DeepFace
import os
import face_mesh_libv1 as face
from math import hypot
import threading

class DatabasePrep():
    def __init__(self,path):
        self.embedding_arr = []
        self.path = path
        self.isExist = os.path.exists(self.path)

        splitted = self.path.split("/")
        self.dataset = self.replaceStr(splitted[0])
        self.company = self.replaceStr(splitted[1])
        self.id = self.replaceStr(splitted[2])
        self.company = self.company.casefold()
        #self.company = self.company.capitalize()
        self.id = self.id.split("_")
        self.name = self.id[0].casefold()
        #self.name = self.id[0].capitalize()
        self.surname = self.id[1].casefold()
        #self.surname = self.id[1].capitalize()
        self.name = self.name + "_" + self.surname

        #self.cam = cv2.VideoCapture(0)
        self.faceMesh = face.FaceMesh()
        self.angle = 45
        self.photoCount = int((90/self.angle))
        self.x = 0

        if not self.isExist:
            os.chdir(self.dataset)
            path = self.company
            if not os.path.exists(path):
                os.mkdir(path)
            os.chdir(path)
            path = self.name
            if not os.path.exists(path):
                os.mkdir(path)
            os.chdir(path)
            self.path = os.getcwd()
        else:
            path = f"{self.dataset}/{self.company}/{self.name}"
            os.chdir(path)
            self.path = os.getcwd()
            image_arr = [i for i in os.listdir() if i.endswith(".jpg")]
            for i in range(len(image_arr)):
                img = image_arr[i]
                os.remove(img)
                print(f"file deleted: {img}")

    def replaceStr(self,old):
        string = old
        letters_from = ["Ğ","Ü","Ş","İ","Ö","Ç","ğ","ü","ş","ı","ö","ç"]
        letters_to = ["G","U","S","I","O","C","g","u","s","i","o","c"]
        for i in range (len(letters_from)):
            if letters_from[i] in old:
                string = string.strip().replace(letters_from[i],letters_to[i])
        new = string

        return new
    
    def area2(self):
        self.x = 0
        print("area 2 begin")
        #for i in (n+1 for n in range(self.photoCount)):
        for i in range(self.photoCount):
            path = self.path + f"/img_{i+1}.jpg"
            if os.path.exists(path):
                os.remove(path)
                print(f"file deleted: img_{i+1}.jpg")
                
        while True:
            #print("while begin")
            self.values, self.image = self.faceMesh.drawing()
            self.alpha = int(self.values[0])
            xcoord,ycoord = self.values[1][0],self.values[1][1]
            x0,y0 = self.image.shape[1]/2,self.image.shape[0]/2
            hypotenuse = hypot(xcoord-x0,ycoord-y0)
            cheek_size = self.values[2]
            optimum_circle = cheek_size
            small_circle = optimum_circle - (cheek_size/4)
            large_circle = optimum_circle + (cheek_size/4)
            #ret,self.image = self.cam.read()
            #self.image = cv2.flip(self.image,1)
            path = self.path + f"/img_{self.x+1}.jpg"
            if self.alpha < self.angle*(self.x+1) and self.alpha >= self.angle*(self.x):
                #print("alpha control")
                if not hypotenuse < small_circle and not hypotenuse > large_circle:
                    if xcoord < x0 and ycoord < y0:
                        self.x += 1
                        #print("tolerance cloud control")
                        cv2.imwrite(path,self.image)
                        print(f"image saved: {path}")
                        if (self.x == self.photoCount) or (xcoord > x0) or (ycoord > y0):
                            #print("while break")
                            break
    
    def area3(self):
        print("area 3 begin")
        self.x = self.photoCount
        #for i in (n+self.photoCount+1 for n in range(self.photoCount*2)):
        for i in range(self.photoCount):
            path = self.path + f"/img_{i+self.photoCount+1}.jpg"
            if os.path.exists(path):
                os.remove(path)
                print(f"file deleted: img_{i}.jpg")

        while True:
            self.values, self.image = self.faceMesh.drawing()
            self.alpha = int(self.values[0])
            xcoord,ycoord = self.values[1][0],self.values[1][1]
            x0,y0 = self.image.shape[1]/2,self.image.shape[0]/2
            hypotenuse = hypot(xcoord-x0,ycoord-y0)
            cheek_size = self.values[2]
            optimum_circle = cheek_size
            small_circle = optimum_circle - (cheek_size/4)
            large_circle = optimum_circle + (cheek_size/4)
            path = self.path + f"/img_{self.x+1}.jpg"
            #print(f"hypot: {hypotenuse}, small circle: {small_circle}, large circle: {large_circle}")
            if self.alpha < self.angle*((self.x+1)-self.photoCount) and self.alpha >= self.angle*(self.x-self.photoCount):
                #print("alpha control")
                if not hypotenuse < small_circle and not hypotenuse > large_circle:
                    if xcoord <= x0 and ycoord >= y0:
                        self.x += 1
                        cv2.imwrite(path,self.image)
                        print(f"image saved: {path}")
                        if (self.x == (self.photoCount*2)) or (xcoord > x0) or (ycoord < y0):
                            break

    def area4(self):
        print("area 4 begin")
        self.x = (self.photoCount*2)
        #for i in (n+(self.photoCount*2)+1 for n in range(self.photoCount*3)):
        for i in range(self.photoCount):
            path = self.path + f"/img_{i+(self.photoCount*2)+1}.jpg"
            if os.path.exists(path):
                os.remove(path)
                print(f"file deleted: img_{i+(self.photoCount*2)+1}.jpg")

        while True:
            self.values, self.image = self.faceMesh.drawing()
            self.alpha = int(self.values[0])
            xcoord,ycoord = self.values[1][0],self.values[1][1]
            x0,y0 = self.image.shape[1]/2,self.image.shape[0]/2
            hypotenuse = hypot(xcoord-x0,ycoord-y0)
            cheek_size = self.values[2]
            optimum_circle = cheek_size
            small_circle = optimum_circle - (cheek_size/4)
            large_circle = optimum_circle + (cheek_size/4)
            path = self.path + f"/img_{self.x+1}.jpg"
            if self.alpha < self.angle*(self.x+1-(self.photoCount*2)) and self.alpha >= self.angle*(self.x-self.photoCount*2):
                if not hypotenuse < small_circle and not hypotenuse > large_circle:
                    if xcoord >= x0 and ycoord >= y0:
                        self.x += 1
                        cv2.imwrite(path,self.image)
                        print(f"image saved: {path}")
                        if (self.x == (self.photoCount*3)) or (xcoord < x0) or (ycoord < y0):
                            break
    
    def area1(self):
        print("area 1 begin")
        self.x = (self.photoCount*3)
        os.chdir(self.path)
        #image_arr = [i for i in os.listdir() if i.endswith(".jpg")]
        #for i in (n+(self.photoCount*3)+1 for n in range(self.photoCount*4)):
        for i in range(self.photoCount):
            path = f"/img_{i+(self.photoCount*3)+1}.jpg"
            if os.path.exists(path):
                os.remove(path)
                print(f"file deleted: img_{i}.jpg")

        while True:
            self.values, self.image = self.faceMesh.drawing()
            self.alpha = int(self.values[0])
            xcoord,ycoord = self.values[1][0],self.values[1][1]
            x0,y0 = self.image.shape[1]/2,self.image.shape[0]/2
            hypotenuse = hypot(xcoord-x0,ycoord-y0)
            cheek_size = self.values[2]
            optimum_circle = cheek_size
            small_circle = int(optimum_circle - (cheek_size/4))
            large_circle = int(optimum_circle + (cheek_size/4))
            path = self.path + f"/img_{self.x+1}.jpg"
            if self.alpha < self.angle*(self.x+1-(self.photoCount*3)) and self.alpha >= self.angle*(self.x-self.photoCount*3):
                if not hypotenuse < small_circle and not hypotenuse > large_circle:
                    if xcoord >= x0 and ycoord <= y0:
                        self.x += 1
                        cv2.imwrite(path,self.image)
                        print(f"image saved: {path}")
                        if (self.x == (self.photoCount*4)) or (xcoord < 0) or (ycoord > y0):
                            break
    
    def middle(self):
        self.x = (self.photoCount*4)+1
        os.chdir(self.path)
        path = self.path + f"/img_{self.x}.jpg"
        if os.path.exists(path):
            os.remove(path)
            print(f"file deleted: img_{self.x}.jpg")
        self.values, self.image = self.faceMesh.drawing()
        if self.values[4] == 1:
            cv2.imwrite(path,self.image)
            print(f"image saved: {path}")


    def embedding(self):
        print("initializing embedding")
        os.chdir(self.path)
        self.image_arr = [i for i in os.listdir() if i.endswith(".jpg")]
        x = 0
        while True:
            x += 1
            path = self.path + f"/img_{x}.jpg"
            embedding = DeepFace.represent(img_path = path, model_name = "VGG-Face",detector_backend = 'retinaface')
            self.embedding_arr.append(embedding)
            print(f"img_{x}.jpg: embedded")
            #print(x)
            if x == len(self.image_arr):
                self.numpy_save()
                break
    
    def numpy_save(self):
        os.chdir(self.path)
        #print(self.embedding_arr[1])
        #print(self.embedding_arr)
        path = self.path + "/vectors"
        isExists = os.path.exists(path+".npy")
        if isExists:
            self.numpy_arr = np.load(path+".npy")
            print(f"{len(self.numpy_arr)} vectors loaded.")
            #self.numpy_arr = np.append(self.numpy_arr,[self.embedding_arr])
            numpy_vector = np.array(self.embedding_arr)
            self.numpy_arr = np.concatenate((self.numpy_arr,numpy_vector))
        else:
            self.numpy_arr = np.array(self.embedding_arr)
        np.save(path,self.numpy_arr)
        print(f"{len(self.embedding_arr)} images saved as vectors. Dataset has {len(self.numpy_arr)} vectors")
        #image_arr = [i for i in os.listdir() if i.endswith(".jpg")]
        for i in range (len(self.image_arr)):
            #path = self.path + f"img_{i+1}.jpg"
            img = self.image_arr[i]
            if os.path.exists(img):
                os.remove(img)
                print(f"image deleted: img_{i+1}.jpg")
    
    def photoShoot(self):
        values, image = self.faceMesh.drawing()
        coords = values[1]
        xCoord,yCoords = coords[0],coords[1]
        width,height = image.shape[1]/2,image.shape[0]/2
        order = []
        fault_active = False
        green_signal = values[4]
        #embedding_thread = threading.Thread(target=self.embedding,daemon=True)
        if green_signal == 1:
            self.middle()
            if xCoord-width < 0 and yCoords-height < 0:
                order = [self.area2,self.area3,self.area4,self.area1]
            elif xCoord-width < 0 and yCoords-height >= 0:
                order = [self.area3,self.area4,self.area1,self.area2]
            elif xCoord-width >= 0 and yCoords-height >= 0:
                order = [self.area4,self.area1,self.area2,self.area3]
            elif xCoord-width >= 0 and yCoords-height < 0:
                order = [self.area1,self.area2,self.area3,self.area4]

            for i in range(len(order)):
                area = order[i]
                area()

        elif green_signal == 0:
            print("lutfen burnunuzu yesil noktaya getirin")

        fault = self.control()
        if 1 in fault:
            fault_active = True
        while fault_active:
            fault = self.control()
            if fault[0] == 1:
                self.area1()
            if fault[1] == 1:
                self.area2()
            if fault[2] == 1:
                self.area3()
            if fault[3] == 1:
                self.area4()
            if fault[4] == 1:
                self.middle()
            if fault[0] == 0 and fault[1] == 0 and fault[2] == 0 and fault[3] == 0 and fault[4] == 0:
                self.embedding()
                #embedding_thread.start()
                fault_active = False
                break
        else:
            self.embedding()
            #embedding_thread.start()
        
    def control(self):
        #os.chdir(self.path)
        os.chdir(self.path)
        #self.image_arr = [i for i in os.listdir() if i.endswith(".jpg")]
        fault = [0,0,0,0,0]

        #if not len(self.image_arr) == self.photoCount*4:
        for i in (n+1 for n in range((self.photoCount*4)+1)):
            path = f"img_{i}.jpg"
            if not os.path.exists(path):
                #print(f"'img_{i+1}' is not exists")
                #add error handlings down below
                print(f"image can not found: img_{i}.jpg")
                if i <= self.photoCount:
                    print("area 2 is not accurate")
                    fault[1] = 1
                elif i <= self.photoCount*2 and i > self.photoCount:
                    print("area 3 is not accurate")
                    fault[2] = 1
                elif i <= self.photoCount*3 and i > self.photoCount*2:
                    print("area 4 is not accurate")
                    fault[3] = 1
                elif i <= self.photoCount*4 and i > self.photoCount*3:
                    print("area 1 is not accurate")
                    fault[0] = 1
                elif i == (self.photoCount*4)+1:
                    print("middle is not accurate")
                    fault[4] = 1
        #elif len(self.image_arr) == self.photoCount*4:
            #fault = [0,0,0,0]

        return fault


if __name__ == "__main__":
    pass