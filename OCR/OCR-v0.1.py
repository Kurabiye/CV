from paddleocr import PaddleOCR
import os
import cv2
from pdf2image import convert_from_path
import tempfile 
from shutil import rmtree


class OCR():
    def __init__(self,pdf):
        self.pdf = pdf
        self.pdf_name = pdf.split("/")[-1].split(".")[0]
        self.doc = {}
        self.doc['pdf_name'] = {}
        self.doc['page'] = {}
        self.doc['text'] = {}
        self.ocr_model = PaddleOCR(use_angle_cls=False,lang='tr')

    def pdf2img(self,pdf):
        try:
            pages = convert_from_path(pdf)
            tmp_path = os.path.join(tempfile.mkdtemp())
            print(os.path.dirname(tmp_path))
            for idx, page in enumerate(pages):
                im_path = tmp_path + f"/{self.pdf_name}-page{idx+1}.jpg"
                page.save(im_path, 'JPEG')
                print("pdf save to file:",im_path)
        except Exception as e:
            print(e)

        return tmp_path

    def ocr(self):
        img_path = self.pdf2img(self.pdf)
        img_arr = [i for i in os.listdir(img_path) if i.endswith(".jpg")]
        name = self.pdf_name
        self.doc["pdf_name"] = name

        for idx, img in enumerate(img_arr):
            img = cv2.imread(img_path+"/"+img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            self.doc["page"] = idx+1

            try:
                result = self.ocr_model.ocr(img,cls=False)
                print(f"{img_arr[idx]} scanned!")
                for idx, res in enumerate(result):
                    text = res[1][0]
                    self.doc["text"][idx] = text
            
            except Exception as e:
                print(e)

        print(self.doc)
        rmtree(img_path, ignore_errors=True)

if __name__ == "__main__":
    ocr = OCR(os.path.join('pdf','mqtt_sim.pdf'))
    ocr.ocr()