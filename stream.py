import cv2
import config


class StreamController:
    def __init__(self, filename_in=config.FILENAME_IN, filename_out=config.FILENAME_OUT):
        self.vs = cv2.VideoCapture(filename_in)
        fourcc = cv2.VideoWriter_fourcc(*"XVID")
        self.writer = cv2.VideoWriter(filename_out, fourcc, config.FPS_OUT, (config.OUT_WIDTH, config.OUT_HEIGHT), True)

    def write_and_show(self, img):
        result_frame = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        self.writer.write(result_frame)
        cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
        cv2.imshow("Frame", result_frame)

    def get_frame(self):
        frame = self.vs.read()
        if frame[1] is None:
            return None
        rgb_frame = cv2.cvtColor(frame[1], cv2.COLOR_BGR2RGB)
        h, w = rgb_frame.shape[:2]
        source_img = cv2.resize(rgb_frame, (int(0.5 * w), int(0.5 * h)))
        return source_img

    def destroy(self):
        self.vs.release()
        self.writer.release()
        cv2.destroyAllWindows()
