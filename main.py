import cv2
import numpy as np
import imutils
import easyocr
import tkinter as tk
from tkinter import *
from tkinter import filedialog, ttk
from PIL import ImageTk, Image

global photo_original
global photo_gray

root = Tk()
root.title('License Plate Detector (PMSCS-600), submitted by Wahid')
root.resizable(width=True, height=True)

frame = tk.Frame(root, bg='#e6e6ff', padx=20, pady=20)
frame.place_configure(relwidth=1, relheight=1)

canvas = tk.Canvas(root, heigh=700, width=1000, bg='#e6e6ff')
canvas.pack()


def detectlicense(image):
    global photo_gray

    img = cv2.imread(image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Show gray image
    img_gray_converted = cv2.cvtColor(img_gray, cv2.COLOR_BGR2RGB)
    pil_img_gray = Image.fromarray(img_gray_converted)
    pil_img_gray = pil_img_gray.resize((400, 400))
    photo_gray = ImageTk.PhotoImage(pil_img_gray)
    label_gray = tk.Label(frame, image=photo_gray)
    label_gray.pack(pady=10)

    #bfilter = cv2.bilateralFilter(gray, 11, 17, 17)  # Noise reduction
    #edged = cv2.Canny(bfilter, 30, 200)  # Edge detection


def btnclick():
    global photo_original

    filename = filedialog.askopenfilename(
        initialdir="/Projects/JU/train data",
        title='Select Image',
        filetypes=([('image', '*.jpg'), ('image', '*.jpeg'), ('image', '*.png'), ('all files', '*.*')]))
    image = Image.open(filename)
    image = image.resize((400, 400))
    photo_original = ImageTk.PhotoImage(image)
    label = tk.Label(frame, image=photo_original)
    label.pack(side='top', anchor='nw', pady=10)
    detectlicense(filename)


openFile = tk.Button(frame, text="Open File", padx=10, pady=5, fg='white', bg='black', command=btnclick)
openFile.pack(side='top', anchor='nw')

root.mainloop()
