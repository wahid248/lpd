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
global photo_filtered
global photo_contour
global photo_cropped

root = Tk()
root.title('License Plate Detector (PMSCS-600), submitted by Wahid')
root.resizable(width=True, height=True)

# Main frame
main_frame = Frame(root, bg='#e6e6ff', padx=20, pady=20)
main_frame.pack(fill=BOTH, expand=1)

# Canvas
canvas = Canvas(main_frame, height=700, width=1000)
canvas.pack(side=LEFT, fill=BOTH, expand=1)

# Scrollbar
scrollbar = ttk.Scrollbar(main_frame, orient=VERTICAL, command=canvas.yview)
scrollbar.pack(side=RIGHT, fill=Y)

# Configure canvas
canvas.configure(yscrollcommand=scrollbar)
canvas.bind('<Configure>', lambda e: canvas.configure(scrollregion=canvas.bbox('all')))

# 2nd frame
frame2 = Frame(canvas)

# Add 2nd frame to a window in the canvas
canvas.create_window((0, 0), window=frame2, anchor='nw')


def detectlicense(image):
    global photo_gray
    global photo_filtered
    global photo_contour
    global photo_cropped

    img = cv2.imread(image)
    img_gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # convert image to gray
    img_gray_converted = cv2.cvtColor(img_gray, cv2.COLOR_BGR2RGB)
    pil_img_gray = Image.fromarray(img_gray_converted)
    pil_img_gray = pil_img_gray.resize((400, 400))
    photo_gray = ImageTk.PhotoImage(pil_img_gray)
    label_gray = tk.Label(frame2, image=photo_gray)
    label_gray.pack(pady=10)

    # filter image and perform edge detection
    bfilter = cv2.bilateralFilter(img_gray, 11, 17, 17)  # Noise reduction
    edged = cv2.Canny(bfilter, 30, 200)  # Edge detection
    img_filtered_converted = cv2.cvtColor(edged, cv2.COLOR_BGR2RGB)
    pil_img_filtered = Image.fromarray(img_filtered_converted)
    pil_img_filtered = pil_img_filtered.resize((400, 400))
    photo_filtered = ImageTk.PhotoImage(pil_img_filtered)
    label_filtered = tk.Label(frame2, image=photo_filtered)
    label_filtered.pack(pady=10)

    # Find contours and apply mask
    keypoints = cv2.findContours(edged.copy(), cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(keypoints)
    contours = sorted(contours, key=cv2.contourArea, reverse=True)[:10]

    location = None
    for contour in contours:
        approx = cv2.approxPolyDP(contour, 15, True)
        if len(approx) == 4:
            location = approx
            break

    mask = np.zeros(img_gray.shape, np.uint8)
    img_contour = cv2.drawContours(mask, [location], 0, 255, -1)
    img_contour = cv2.bitwise_and(img, img, mask=mask)

    img_contour_converted = cv2.cvtColor(img_contour, cv2.COLOR_BGR2RGB)
    pil_img_contour = Image.fromarray(img_contour_converted)
    pil_img_contour = pil_img_contour.resize((400, 400))
    photo_contour = ImageTk.PhotoImage(pil_img_contour)
    label_contour = tk.Label(frame2, image=photo_contour)
    label_contour.pack(pady=10)

    # crop the license plate
    (x, y) = np.where(mask == 255)
    (x1, y1) = (np.min(x), np.min(y))
    (x2, y2) = (np.max(x), np.max(y))
    img_cropped = img_gray[x1:x2+1, y1:y2+1]

    img_cropped_converted = cv2.cvtColor(img_cropped, cv2.COLOR_BGR2RGB)
    pil_img_cropped = Image.fromarray(img_cropped_converted)
    pil_img_cropped = pil_img_cropped.resize((100, 50))
    photo_cropped = ImageTk.PhotoImage(pil_img_cropped)
    label_cropped = tk.Label(frame2, image=photo_cropped)
    label_cropped.pack(pady=10)

    # Apply OCR
    reader = easyocr.Reader(['en'])
    result = reader.readtext(img_cropped)
    print(result)

def btnclick():
    global photo_original

    filename = filedialog.askopenfilename(
        initialdir="/Projects/JU/train data",
        title='Select Image',
        filetypes=([('image', '*.jpg'), ('image', '*.jpeg'), ('image', '*.png'), ('all files', '*.*')]))
    image = Image.open(filename)
    image = image.resize((400, 400))
    photo_original = ImageTk.PhotoImage(image)
    label = tk.Label(frame2, image=photo_original)
    label.pack(side='top', anchor='nw', pady=10)
    detectlicense(filename)


openFile = tk.Button(frame2, text="Open File", padx=10, pady=5, fg='white', bg='black', command=btnclick)
openFile.pack(side='top', anchor='nw')

root.mainloop()
