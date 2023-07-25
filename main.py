import random

import numpy as np
from matplotlib import pyplot as plt
import cv2
from search import LocationSearch
from stream import StreamController
import time
import config


def track(choice):
    searcher = LocationSearch()
    stream = StreamController(config.FILENAME_IN, config.FILENAME_OUT)

    total_frames = 0
    rnd = random.randint(0,2)
    while True:
        #t = time.time()
        frame = stream.get_frame()
        if frame is None:
            break

        if total_frames % 5 == 0:
            source_img = frame
            if choice == 0:
                source_img = frame
            elif choice == 1:
                source_img = cv2.blur(frame, (4, 4))
            elif choice == 2:
                source_img = cv2.GaussianBlur(frame,(5,5),0)
            elif choice == 3:
                angles = [90, 180, 270]
                h, w = frame.shape[:2]
                center = (w/2,h/2)
                M = cv2.getRotationMatrix2D(center, angles[rnd], 1.0)
                source_img = cv2.warpAffine(frame, M, (h, w))
            elif choice == 4:
                sharp_filter = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
                source_img = cv2.filter2D(frame, ddepth=-1, kernel=sharp_filter)
            elif choice == 5:
                morph_kernel = np.ones((3, 3))
                source_img = cv2.dilate(frame, kernel= morph_kernel, iterations=1)
            elif choice == 6:
                morph_kernel = np.ones((3, 3))
                source_img = cv2.erode(frame, kernel= morph_kernel, iterations=1)
            elif choice == 7:
                clahe = cv2.createCLAHE(clipLimit=3., tileGridSize=(8, 8))

                lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)  # convert from BGR to LAB color space
                l, a, b = cv2.split(lab)  # split on 3 different channels

                l2 = clahe.apply(l)  # apply CLAHE to the L-channel

                lab = cv2.merge((l2, a, b))  # merge channels
                source_img = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)  # convert from LAB to BGR
            target_img = plt.imread(config.FILENAME_MAP)
            h, w = target_img.shape[:2]
            target_img = cv2.resize(target_img, (int(0.75 * w), int(0.75 * h)))
            try:
                output_image = searcher.search(source_img, target_img)
            except Exception as e:
                searcher.is_focus = False
                pass
        #print(f'{time.time() - t}')
        output_image = cv2.resize(output_image, (1920,1080))
        stream.write_and_show(output_image)

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

        total_frames += 1

    stream.destroy()

def chose_filter():
    choice = -1
    while choice < 0 or choice > 7:
        print('0 - nothing')
        print('1 - blur')
        print('2 - GaussianBlur')
        print('3 - rotation')
        print('4 - sharpen')
        print('5 - dilate')
        print('6 - erode')
        print('7 - contrast')
        choice = int(input())
    return choice

if __name__ == "__main__":
    track(chose_filter())

