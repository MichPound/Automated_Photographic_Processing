# Library imports.
import tkinter as tk
from tkinter import *  
from tkinter import ttk
from tkinter import filedialog as fd
from tkinter.messagebox import showinfo
import numpy as np
import cv2
import os
import re
import time
import pandas as pd
import tensorflow as tf
import csv


# Imports cropping scripts.
import crop_frame
import crop_art

# Imports OCR functions.
from ocr import advanced_ocr
from ocr import simple_ocr

# Imports scaled shot function.
from scaled_shot import scaled

# Loads in CNN model.
model = tf.keras.models.load_model("19-04-22-1127")


os.system('cls' if os.name == 'nt' else 'clear')
print("")
print("#################################")
print("Automated Photographic Processing")
print("#################################")
print("")

# GUI setup.
root = tk.Tk()
root.title('A.P.P.')
root.resizable(True, True)
root.geometry('550x300')

# Initialses varables.
input = ""
output = ""
measurements = ""
recognition = True
flag = False

# Initialise input label.
label1 = Label(text = input)
label1.place(x=18, y=50)

# Initialise output label.
label2 = Label(text = output)
label2.place(x=18, y=110)

# Initialise measurements label.
label3 = Label(text = measurements)
label3.place(x=18, y=170)

# Adding image order label.
image_order_label = Label(text = "Image Order")
image_order_label.place(x=375, y=50)

# Initilaise the processing time label.
processing_time = Label(text = "")
processing_time.place(x=375, y=245)

def select_input():
    # Selecting input directory.
    global input
    input = fd.askdirectory(
        title='Select Input Directory',
        initialdir='/home/michael/fyp/main/',)

    # Adding input label to GUI.
    label1 = Label(text = input)
    label1.place(x=18, y=50)

def select_output():
    # Selecting output directory.
    global output
    output = fd.askdirectory(
        title='Select Output Directory',
        initialdir='/home/michael/fyp/main/',)

    # Adding output label to GUI.
    label2 = Label(text = output)
    label2.place(x=18, y=110)

def select_measurements():
    # Limiting to CSV files
    filetypes = (
        ('CSV file', '*.csv'),
    )

    # Selecting measurements file.
    global measurements
    measurements = fd.askopenfilename(
        title='Select Measurements File',
        initialdir='/home/michael/fyp/main/',
        filetypes=filetypes)

    # Adding measurements label to GUI.
    label3 = Label(text = measurements)
    label3.place(x=18, y=170)

# Main process function.
def process():

    # Error handling.

    # Checking if input and output directories are selected.
    if input == "" or output == "":
        showinfo("Error", "Please select input and output directories.")
        return
    
    # Checking directories exist.
    if os.path.isdir(input) == False or os.path.isdir(output) == False:
        showinfo("Error", "Directory does not exist.")
        return
    
    # Checking if measurements file is selected.
    if check_3.get() == 1 and measurements == "":
        showinfo("Error", "Please select measurements file.")
        return

    # Checking if measurements file exists.
    if check_3.get() == 1 and os.path.isfile(measurements) == False:
        showinfo("Error", "Measurements file does not exist.")
        return

    # Checking if name by flah is selected.
    if check_3.get() == 1 and check_2.get() != 1:
        showinfo("Error", "Name by Flag must be enabled for scaled shots.")
        return

    # Checking each order number is selected only once.
    if variable1.get() == variable2.get() or variable1.get() == variable3.get() or variable2.get() == variable3.get():
        showinfo("Error", "Please select only one option per number.")
        return

    # Checking correct order is seected.
    if check_2.get() != 1:
        if (int(variable1.get()) != 1 and int(variable1.get()) != 2) or (int(variable2.get()) != 1 and int(variable2.get()) != 2):
            showinfo("Error", "Please select option number in increasing order.")
            return

    # If scaled shots selected load in background images.
    if check_3.get() == 1:
        # Read in measurements CSV.
        measurementsPD = pd.read_csv(measurements)

        # Loading background images.
        dark_large = cv2.imread("28_3500_12.jpg")
        dark_small = cv2.imread("50_005c.jpg")
        light_large = cv2.imread("28_4500_17.jpg")
        light_small = cv2.imread("50_001.jpg")

    # Starts procesing timer.
    start = time.time()

    print("")

    # Initialises variables.
    file_number = 0
    front_file = ""
    back_file = ""
    ocr_file = ""
    file_count = 0

    # Loads in text detection model.
    net = cv2.dnn.readNet("frozen_east_text_detection.pb")

    # Initislise manual review list.
    manual_review_list = []

    # Loops through each file in input directory.
    for file in sorted(os.listdir(input)):

        # Incrememnts file number.
        file_number += 1
        filename = ""

        # Checks if current file is set to be front image.
        if file_number == int(variable1.get()):
            front_file = file

        # Checks if current file is set to be back image.
        elif file_number == int(variable2.get()):
            back_file = file

        # Checks if current file is set to be OCR image.
        elif check_2.get() == 1:
            if variable3.get() != "-" and file_number == int(variable3.get()):
                ocr_file = file
            else:
                showinfo("Error", "Please select 1, 2 or 3 for flag option.")
                return

        # Checks if all images of current lot have been looped over.
        if  (check_2.get() == 1 and file_number == 3) or (check_2.get() != 1 and file_number == 2):
            
            # Resets file number for looping.
            file_number = 0

            # Increments file count fr naming without OCR.
            file_count += 1

            # OCR extraction for lot numbers.
            if check_2.get() == 1:
                filename = ocr_extract(input + "/" + ocr_file, net)

                # Cleans lot number to be only numerical characters.
                filename = re.sub(r'[^0-9]', '', filename)

                if filename == "": filename = "-" + str(file_count)
            else:
                filename = str(file_count)

            print("Lot Number: ", filename)

            cropped = frame_crop(front_file, filename, False)

            # If exception was caught add to manual review list.
            if cropped == False:

                manual_review_list.append([filename, front_file, "Front Frame Crop"])

                print("")
                continue

            # Detecting average colour of frame edges.
            colour = colour_detect(cropped)

             # Set up and prediction of CNN model.
            #####################################
            CATEGORIES = ['type_one', 'type_two']

            prediction = model.predict([[prepare(cropped)]])

            predicted_class = np.argmax(prediction)

            print("Frame Classification: ", CATEGORIES[predicted_class])
            #####################################

            # Creating artwork crop.
            if (CATEGORIES[predicted_class] == 'type_one') and (colour == "Lighter"):
                art_cropped = art_crop(cropped, filename, True)
            elif (CATEGORIES[predicted_class] == 'type_one') and (colour == "Darker"):
                art_cropped = dark_art_crop(cropped, filename, True)
            elif CATEGORIES[predicted_class] == 'type_two' and (colour == "Lighter"):
                art_cropped = art_crop(cropped, filename, False)
            elif CATEGORIES[predicted_class] == 'type_two' and (colour == "Darker"):
                art_cropped = dark_art_crop(cropped, filename, False)
            else:
                art_cropped == False

            # If exception was caught add to manual review list.
            if art_cropped == False:
                    
                manual_review_list.append([filename, front_file, "Artwork Crop"])

                print("")
                continue

            # Back Image.
            back = frame_crop(back_file, filename, True)

            # If exception was caught add to manual review list.
            if back == False:

                manual_review_list.append([filename, back_file, "Back Frame Crop"])
                print("")
                continue

            # Checks if back text extraction is enabled.
            if check_1.get() == 1:
                temp_text = simple_ocr(back)

                # Cleans any text found.
                temp_text = re.sub(' +', ' ', temp_text)
                temp_text = re.sub('\r', '', temp_text)
                temp_text = re.sub('\t', '', temp_text)

                # If text is found, create text file.
                if temp_text != "":
                    text_file = open(output + "/" + filename + ".txt", "w")
                    text_file.write(temp_text)
                    text_file.close()

            # Checks if scaled images is selected.
            if check_3.get() == 1:
                # Scaled shot here
                ######################################################
                measure = measurementsPD[measurementsPD['lot'] == int(filename)]
                
                # If measurement found create scaled image.
                if not measure.empty:
                    # Retrieveing frame measurements.
                    frame_width = int( measure['width'].iloc[0])

                    # Initialise background image.
                    background = dark_large

                    # Selecting correct scaled background.
                    if (colour == "Lighter") & (frame_width > 120):
                        background = dark_large
                        background_width = 500
                    elif (colour == "Lighter") & (frame_width <= 120):
                        background = dark_small
                        background_width = 180
                    elif (colour == "Darker") & (frame_width > 120):
                        background = light_large
                        background_width = 450
                    elif (colour == "Darker") & (frame_width <= 120):
                        background = light_small
                        background_width = 175
                    
                    # Creating scaled image.
                    scale_img = scaled(cropped, background, background_width, frame_width, output, filename)

                    if scale_img == False:

                        manual_review_list.append([filename, front_file, "Frame Scaling"])

                        print("")
                        continue
                    
                ######################################################

            # Cleans up terminal output.
            print(" ")

    # If manual review list is not empty, create manual review CSV.
    if manual_review_list != []:

        print("Creating Manual Review CSV")

        header = ["Lot", "File", "Issue"]

        with open(output + "/manual_review.csv", "w") as csvfile:
            
            writer = csv.writer(csvfile)

            writer.writerow(header)

            writer.writerows(manual_review_list)

    # Ends procesing timer.
    end = time.time()
    print(end - start)

    # Adds GUI processing time.
    processing_time = Label(text = str(int(end - start)) + " seconds")
    processing_time.place(x=375, y=245)

# Prepares image for model prediction.
def prepare(filepath):
    IMG_SIZE = 1000
    img_array = cv2.imread(filepath)
    new_array = cv2.resize(img_array, (IMG_SIZE, IMG_SIZE))
    return new_array.reshape(1, IMG_SIZE, IMG_SIZE, 3)

# Front and back frame crops.
def frame_crop(file, filename, back):
    crop = crop_frame.frame_crop(input + "/" + file, output + "/" + filename + ".jpg", back)
    return crop

# Artwork frame crops.
def art_crop(cropped, filename, type_one):
    crop = crop_art.art_crop(cropped, output + "/" + filename + ".jpg", type_one)
    return crop

# Artwork dark frame crops.
def dark_art_crop(cropped, filename, type_one):
    crop = crop_art.dark_art_crop(cropped, output + "/" + filename + ".jpg", type_one)
    return crop

# Creates OCR text file.
def ocr_extract(file, net):
    text = advanced_ocr(file, net)
    return text

# Used for detecting the colour average of frames.
def colour_detect(image):
    image = cv2.imread(image)

    W, H, C = image.shape

    # List of different points around the frame.
    dimensions = [
        [20, 20],
        [W/2, 20],
        [W-20, 20],
        [20, H/2],
        [W-20, H/2],
        [20, H-20],
        [W/2, H-20],
        [W-20, H-20]
    ]

    # Initialise list of colours.
    colors = []

    # Loops through all points.
    for x, y in dimensions:
        colors.append(image[int(x), int(y)])

    # Calculates average colour of points.
    average = np.average(colors, axis=0)
    average = average.astype(int)

    # Checks if average colour is in range.
    lower = np.array([0, 0, 0])
    upper = np.array([100, 100, 100])
    mask = cv2.inRange(average, lower, upper)

    # Returns measure of frame colour.
    if np.array_equal(mask, np.array([[0], [0], [0]]), equal_nan=False):
        print("Frame Colour: Lighter")
        return "Lighter"
    else:
        print("Frame Colour: Darker")
        return "Darker"

# Input directory selection button.
input_button = ttk.Button(
    root,
    text='Select Input Directory',
    command=select_input
)
input_button.place(x=18, y=20)

# Output directory selection button.
output_button = ttk.Button(
    root,
    text='Select Output Directory',
    command=select_output
)
output_button.place(x=18, y=80)

# Measurement file selection button.
measurement_button = ttk.Button(
    root,
    text='Select Measurements File',
    command=select_measurements
)
measurement_button.place(x=18, y=140)

# Initiate cropping button.
process_button = ttk.Button(
    root,
    text='Process',
    command=process
)
process_button.place(x=375, y=215)

# Menu option numbers.
choices = ["1", "2", "3"]

# Front option menu.
variable1 = StringVar(root)
variable1.set("1")
w1 = OptionMenu(root, variable1, *choices)
w1.place(x=375, y=100)
option1 = Label(text = "Front")
option1.place(x=450, y=100)

# Back option menu.
variable2 = StringVar(root)
variable2.set("2")
w2 = OptionMenu(root, variable2, *choices)
w2.place(x=375, y=125)
option2 = Label(text = "Back")
option2.place(x=450, y=125)

# Flag option menu.
variable3 = StringVar(root)
variable3.set("-")
w3 = OptionMenu(root, variable3, *choices)
w3.place(x=375, y=150)
option3 = Label(text = "Flag")
option3.place(x=450, y=150)

# Checkbutton for frame back text extraction.
check_1 = IntVar()
recognition_check = Checkbutton(root, text="Artist's Notes", variable=check_1)
recognition_check.place(x=18, y=215)

# Checkbutton for lot number extraction.
check_2 = IntVar()
flag_check = Checkbutton(root, text="Name by Flag", variable=check_2)
flag_check.place(x=18, y=245)

# Checkbutton for scaled shots.
check_3 = IntVar()
crop_check = Checkbutton(root, text="Scaled shots", variable=check_3)
crop_check.place(x=150, y=215)

root.mainloop()