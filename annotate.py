import cv2
import numpy as np
import os


class Segmenter:
    def __init__(self, filename):
        cv2.namedWindow("result")
        self.filename = filename
        self.image = cv2.imread(filename)
        self.image_disp = np.copy(self.image)
        self.model = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.model.fill(cv2.GC_PR_BGD)

        self.mask = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        self.mask.fill(0)

        cv2.imshow("result", self.image)
        self.lmb_drawing = False
        self.rmb_drawing = False
        self.bg_model = np.zeros((1, 65), np.float64)
        self.fg_model = np.zeros((1, 65), np.float64)
        self.rect = (50, 50, 450, 290)

    def on_mouse(self, event, x, y, flag, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            self.lmb_drawing = True
        elif event == cv2.EVENT_LBUTTONUP:
            self.lmb_drawing = False

        if event == cv2.EVENT_RBUTTONDOWN:
            self.rmb_drawing = True
        elif event == cv2.EVENT_RBUTTONUP:
            self.rmb_drawing = False

        elif event == cv2.EVENT_MOUSEMOVE:
            if self.lmb_drawing == True:
                cv2.circle(self.image, (x, y), 1, (0, 0, 255), -1)
                cv2.circle(self.model, (x, y), 1, cv2.GC_FGD, -1)
                cv2.imshow("result", self.image)
            elif self.rmb_drawing == True:
                cv2.circle(self.image, (x, y), 1, (255, 0, 0), -1)
                cv2.circle(self.model, (x, y), 1, cv2.GC_BGD, -1)
                cv2.imshow("result", self.image)

        elif event == cv2.EVENT_MBUTTONDOWN:
            cv2.grabCut(self.image_disp, self.model, self.rect, self.bg_model, self.fg_model, 1, cv2.GC_INIT_WITH_MASK)
            self.mask = np.copy(self.model)
            self.mask = np.piecewise(self.mask, [self.mask == cv2.GC_FGD, self.mask == cv2.GC_PR_FGD], [255, 255, 0])
            cv2.imshow("mask", self.mask.astype(np.uint8))
            head, tail = os.path.split(self.filename)
            print(path + "/mask/" + tail)
            cv2.imwrite(path + "/mask/" + tail, self.mask.astype(np.uint8))

    def callback(self):
        cv2.setMouseCallback("result", self.on_mouse, param=None)
        while True:
            if cv2.waitKey(10) == 27:
                break


path = "data/"

if __name__ == '__main__':
    sg = Segmenter(path + "/img/" + "00100.jpg")
    sg.callback()
