import time
import torch
import clip
from PIL import Image
import cv2
import numpy as np
from Objects import four_point_transform, PlayAudio


device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-L/14", device=device)

probs = None
last_processed = time.time()
just_started = 0
text_d = None


def process_clip(img):
    global probs, last_processed, just_started
    img = preprocess(Image.fromarray(img[:, :, ::-1])).unsqueeze(0).to(device)
    logits_per_image, logits_per_text = model(img, text)
    probs = logits_per_image.softmax(dim=-1).cpu().numpy()
    last_processed = time.time()
    if just_started == 0:
        text_d = "You can start drawing now"
        message.message = text_d
        message.play()
        just_started = 1


aux_classes = ['apple', 'aquarium fish', 'baby', 'bear', 'beaver', 'bed', 'bee', 'beetle', 'bicycle', 'bird', 'bottle', 'bowl', 'boy', 'bridge', 'bus', 'butterfly', 'camel', 'can', 'car', 'castle', 'cat', 'caterpillar', 'cattle', 'chair', 'chimpanzee', 'clock', 'cloud', 'cockroach', 'couch', 'crab', 'crocodile', 'cup', 'dinosaur', 'dog', 'dolphin', 'elephant', 'face', 'flatfish', 'flower', 'forest', 'fox', 'girl', 'guitar', 'hamster', 'horse', 'house', 'kangaroo', 'keyboard', 'lamp', 'lawn mower', 'leopard', 'lion', 'lizard', 'lobster', 'logo', 'man', 'maple tree', 'mask', 'motorcycle', 'mountain', 'mouse', 'mushroom', 'oak tree', 'orange', 'orchid', 'otter', 'palm tree', 'pear', 'person', 'pickup truck', 'pine tree', 'plain', 'plate', 'poppy', 'porcupine', 'possum', 'rabbit', 'raccoon', 'ray', 'rectangle', 'road', 'rocket', 'rose', 'sea', 'seal', 'shark', 'shrew', 'skunk', 'skyscraper', 'snail', 'snake', 'spider', 'squirrel', 'streetcar', 'sunflower', 'superhero', 'sweet pepper', 'table', 'tank', 'telephone', 'television', 'tiger', 'tractor', 'train', 'trout', 'tulip', 'turtle', 'wardrobe', 'whale', 'willow tree', 'wolf', 'woman', 'worm']
# interest_class = input("What do you want to draw? ")
# if interest_class.lower() in aux_classes:
#     aux_classes.remove(interest_class)
text_descr = ["a clean white board"] + ["a drawing of " + c for c in aux_classes]  # + [f"a drawing of {interest_class}"]
# ["a clean white board", "a person drawing"] +

text = clip.tokenize(text_descr).to(device)
text_features = model.encode_text(text)

cam = cv2.VideoCapture(1)
M, new_size = four_point_transform(np.array([[16, 71], [628, 50], [589, 417], [74, 425]], "float32"))

is_success = False
success = PlayAudio("./success.mp3")
message = PlayAudio(None)


message.message = "You are going to draw one of these things and I'm going to guess it. Press any key whenever you're ready."
message.play(True)
cls = cv2.imread('classes.png')
r = 700 / cls.shape[1]
cls = cv2.resize(cls, None, fx=r, fy=r)
cv2.imshow("Classes", cls)
cv2.waitKey()
cv2.destroyWindow("Classes")

message.message = "Let's see if I can guess your drawing"
message.play(False)

last_max_class = ""

with torch.no_grad():
    while True:
        ret, frame = cam.read()
        frame = cv2.rotate(frame, cv2.ROTATE_180)
        warped = cv2.warpPerspective(frame, M, new_size)

        if (probs is None or time.time() - last_processed > 3) and not is_success:
            process_clip(warped)
            max_class = np.argmax(probs[0])
            if just_started == 2 and max_class > 0:
                if last_max_class != max_class:
                    if probs[0][max_class] > 0.7:
                        text_d = f"I see {text_descr[max_class]}"
                    else:
                        text_d = f"I think it's {text_descr[max_class]} but I'm not sure..."
                    message.message = text_d
                    message.play()
                    last_max_class = max_class

        frame = cv2.pyrUp(frame)

        if just_started == 1 and max_class == 0:
            cv2.putText(frame, f"Start drawing!", (20, 40), cv2.FONT_HERSHEY_COMPLEX, frame.shape[1] / 800, (252, 184, 73), 2, cv2.LINE_AA)
        if just_started == 1 and max_class != 0:
            just_started = 2
        if just_started == 2:
            if max_class > 0:
                cv2.putText(frame, f"I see {text_descr[max_class]} ({int(probs[0][max_class] * 100.)}%)", (20, 40), cv2.FONT_HERSHEY_COMPLEX, frame.shape[1] / 800, (252, 184, 73), 2, cv2.LINE_AA)

        if max_class > 0 and probs[0][max_class] > 0.75:
            if not is_success:
                # success.play()
                is_success = True
                time.sleep(1.)
                message.message = f"Did I guessed your drawing?"
                message.play()
        # if just_started == 2 and max_class == len(text_descr) - 1 and probs[0][max_class] > 0.89:
        #     if not is_success:
        #         success.play()
        #         is_success = True
        #         time.sleep(1.)
        #         message.message = f"Yes! I guessed your drawing of {interest_class}. Thanks for playing!"
        #         message.play()

        cv2.imshow("Result", frame)
        # cv2.imshow("Warped", warped)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

cv2.destroyAllWindows()
cam.release()
