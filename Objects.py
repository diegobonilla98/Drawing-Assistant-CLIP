from pydub import AudioSegment
from pydub.playback import play
from threading import Thread
import numpy as np
import cv2
from gtts import gTTS


class PlayAudio:
    def __init__(self, sound_filename):
        if sound_filename is not None:
            self.sound = AudioSegment.from_mp3(sound_filename)
            self.message = None
        else:
            self.message = ""

    def play(self, is_parallel=True):
        if is_parallel:
            Thread(target=self._play_sound, args=()).start()
        else:
            self._play_sound()

    def _play_sound(self):
        if self.message is not None:
            tts = gTTS(self.message, lang='en')
            tts.save('msg.mp3')
            self.sound = AudioSegment.from_mp3('msg.mp3')
        play(self.sound)


def four_point_transform(pts):
    def order_points(pts):
        rect = np.zeros((4, 2), dtype="float32")
        s = pts.sum(axis=1)
        rect[0] = pts[np.argmin(s)]
        rect[2] = pts[np.argmax(s)]
        diff = np.diff(pts, axis=1)
        rect[1] = pts[np.argmin(diff)]
        rect[3] = pts[np.argmax(diff)]
        return rect
    rect = order_points(pts)
    (tl, tr, br, bl) = rect
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    M = cv2.getPerspectiveTransform(rect, dst)
    return M, (maxWidth, maxHeight)
