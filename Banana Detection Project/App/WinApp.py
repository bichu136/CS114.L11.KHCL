import cv2
import numpy as np
import tkinter as tk
from tkinter import *
from tkinter import filedialog
from PIL import Image, ImageTk
import model


def custom_resize(img):
    n = 1
    flag = True
    height, width = img.shape[:2]
    while flag:
        if height/n > scr_height or width/n > scr_width/2:
            n += 1
        else:
            flag = False

    img = cv2.resize(img, (int(width/n), int(height/n)))
    return img


def image_process():
    im_tk = tk.Tk()
    im_tk.withdraw()

    path = filedialog.askopenfilename()
    result = model.predict(model.load_image(path))
    cv2.imshow('Result', custom_resize(result))
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def video_process():
    vid_tk = tk.Tk()
    vid_tk.withdraw()
    path = filedialog.askopenfilename()
    cap = cv2.VideoCapture(path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    while cap.isOpened():
        ret, frame = cap.read()
        if ret is True:
            # start = time.time()
            new_frame = model.predict_live(frame)
            # print(time.time() - start)
            cv2.imshow('frame', custom_resize(new_frame))
            if cv2.waitKey(int(100/fps)) & 0xFF == ord('q'):
                break
        else:
            break
    cap.release()
    cv2.destroyAllWindows()
    print('Done')


def live_process():

    video = cv2.VideoCapture(0)
    while True:
        ret, frame = video.read()
        result = model.predict_live(frame)
        cv2.imshow('Result', custom_resize(result))
        k = cv2.waitKey(1) & 0xFF
        if k == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


def close_window():
    root.destroy()


# file_img = 'image_video/human_1.jpg'
# file_bgr = 'image_video/BG_1.jpg'
# file_video = 'image_video/Green_effect_1.mp4'


root = tk.Tk()
root.geometry('720x360')

scr_width = root.winfo_screenwidth()
scr_height = root.winfo_screenheight()

subBG = Image.open('BG.jpg')
subBG.resize((720, 360))
subBG = ImageTk.PhotoImage(subBG)

frameMain = Frame(root)
frameMain.pack()

varMain = StringVar()
labelMain = Label(frameMain, textvariable=varMain, relief=RAISED, height=100, width=720, padx=20, pady=20)
labelMain.config(font=("Courier", 16, "bold"))
labelMain.config(image=subBG, compound='center')
varMain.set("Welcome to our project\nBanana Detection")

varSub = StringVar()
labelSub = Label(frameMain, textvariable=varSub, relief=RAISED, bd=0)
labelSub.config(font=("Courier", 12), pady=10)
varSub.set("Processing with: ")

labelMain.pack()
labelSub.pack()

frameSub = Frame(root)
frameSub.pack()

frameBottom = Frame(root)
frameBottom.pack(side=BOTTOM, fill='x')


A = Button(frameSub, text='Image', command = lambda : image_process(), padx=10, pady=10)
A.config(font=("Courier", 12))
list_refPt = []
B = Button(frameSub, text='Video', command= lambda :video_process(), padx=10, pady=10)
B.config(font=("Courier", 12))
C = Button(frameSub, text='Live', command= lambda : live_process(), padx=10, pady=10)
C.config(font=("Courier", 12))
D = Button(frameBottom, text='End', command=close_window, padx=10, pady=10)
D.config(font=("Courier", 12))

A.grid(column=0, row=0, padx=10, pady=20)
B.grid(column=1, row=0, padx=10, pady=20)
C.grid(column=3, row=0, padx=10, pady=20)
D.pack(side=RIGHT)
root.mainloop()



