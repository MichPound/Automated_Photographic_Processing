import pytesseract
import cv2
import numpy as np
import imutils
from imutils.object_detection import non_max_suppression

def simple_ocr(filename):
    # Read in image.
    img = cv2.imread(filename)

    # Image pre-processing.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel)
    img = cv2.erode(img, kernel)
    img = cv2.medianBlur(img, 1)

    # Apply OCR and return text.
    return(pytesseract.image_to_string(img))


def advanced_ocr(filename, net):
    # Read in image.
    img = cv2.imread(filename)

    # Creates copy of ogiinal image.
    original = img.copy()

    # Getting original image dimensions.
    original_width, original_height, channels = img.shape

    # Resize image to a smaller size for network.
    img = imutils.resize(img, height=960, width=960)
    W, H, channels = img.shape
    new_height = original_width / H
    new_width = original_height / W

    # Image pre-processing.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    kernel = np.ones((1, 1), np.uint8)
    img = cv2.dilate(img, kernel)
    img = cv2.erode(img, kernel)
    img = cv2.medianBlur(img, 1)
    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)

    # Create blob from image.
    blob = cv2.dnn.blobFromImage(img, 1, (W, H), (115, 117, 110), swapRB=True, crop=False)

    # Set output layers for pretrained network.
    layers = [
        "feature_fusion/Conv_7/Sigmoid",
        "feature_fusion/concat_3"
        ]

    # Feed blob into network.
    net.setInput(blob)
    (scores, geometry) = net.forward(layers)
    (no_rows, no_cols) = scores.shape[2:4]
    rects = []

    # Loop through each row.
    for y in range(0, no_rows):

        # Getting confidence scores.
        score = scores[0, 0, y]

        # Geometrics for possible boxes.
        x_0 = geometry[0, 0, y]
        x_1 = geometry[0, 1, y]
        x_2 = geometry[0, 2, y]
        x_3 = geometry[0, 3, y]
        angles = geometry[0, 4, y]

        # Loop through each column.
        for x in range(0, no_cols):
            
            # Checks confidence is greater than 80%.
            if score[x] < 0.8:
                continue

            # Re-adjusts coordinates to image as it maps 4 times smaller.
            (x_offset, y_offset) = (x * 4.0, y * 4.0)

            # extract the rotation angle for the prediction and then
            # compute the sin and cosine
            angle = angles[x]
            cos = np.cos(angle)
            sin = np.sin(angle)

            # Using geometry to build the width and height of text box.
            h = x_0[x] + x_2[x]
            w = x_1[x] + x_3[x]

            # Building main coordinates of text box.
            bottom_x = int(x_offset + (cos * x_1[x]) + (sin * x_2[x]))
            bottom_y = int(y_offset - (sin * x_1[x]) + (cos * x_2[x]))
            top_x = int(bottom_x - w)
            top_y = int(bottom_y - h)

            # Add text box to list of rectangles.
            rects.append((top_x, top_y, bottom_x, bottom_y))

    # Creating a list of all text boxes.
    # Uses NMS to handle overlapping text boxes.
    boxes = non_max_suppression(np.array(rects))

    # Limited to first 250 boxes.
    boxes = boxes[0:250]

    # Initialise list for all extracted text.
    all_text = []

    # Loops over all boxes of detected text.
    for (top_x, top_y, bottom_x, bottom_y) in boxes:

        # Re-scaling coordinates.
        top_x = int(top_x * new_width)
        top_y = int(top_y * new_height)
        bottom_x = int(bottom_x * new_width)
        bottom_y = int(bottom_y * new_height)

        # Initialise clean text.
        cleaned_text = ""
        # Initialise padding.
        padding = 5

        # While tesseract returns blank text increase padding by 5 pixels until text extracted.
        while cleaned_text == "":

            # Limits max padding size.
            if padding == 50: break

            # Limits max number of extracted text.
            if len(all_text) >= 4: break

            # Cropping text box out of original image.
            square_text = original[top_y - padding:bottom_y + padding, top_x - padding:bottom_x + padding]

            if square_text.shape[0] == 0 or square_text.shape[1] == 0:
                break

            # Pre-processing.
            square_text = cv2.cvtColor(square_text, cv2.COLOR_BGR2GRAY)
            kernel = np.ones((1, 1), np.uint8)
            square_text = cv2.dilate(square_text, kernel)
            square_text = cv2.erode(square_text, kernel)
            square_text = cv2.medianBlur(square_text, 1)

            # increase padding by 5 pixels.
            padding = padding + 5

            # Extracting text.
            cleaned_text = str(pytesseract.image_to_string(square_text)).strip()

        # Building list of all text extracted.
        all_text.append(cleaned_text)

    # Return specific lot number text box.
    if len(all_text) < 4:
        return ""
    else:
        return all_text[3]