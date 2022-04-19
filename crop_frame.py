import cv2
import numpy as np

def frame_crop(input, output, back):
    # Read in cropped frame image.
    img = cv2.imread(input)
    original = img

    # Image pre-processing to imrpove clarity.
    kernel = np.ones((5, 5), np.uint8)
    img = cv2.dilate(img, kernel, iterations=1)
    img = cv2.GaussianBlur(img,(1,1),0)
    kernel = np.ones((5,5),np.float32)/25
    img = cv2.filter2D(img,-1,kernel)

    # Building colour range.
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    col_range = cv2.inRange(hsv, (36, 25, 25), (69, 255,255))

    # Finding all contours in processed image.
    contours, hierarchy = cv2.findContours(col_range, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialise 2 largest contours.
    select1 = (0, 0, 0, 0)

    # Finding the largest contour.
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        contour_area = w * h
        select1_area = select1[2] * select1[3]

        if contour_area > select1_area:
            select1 = x, y, w, h

    x, y, w, h = select1

    # Adjustment value for cleaning up the cropped image edges.
    adjustment = 7

    # Crop the image to the largest contour.
    # Check if back flag is set and adjust file naming.
    try:
        output = output.split('.')
        if back == True:
            cv2.imwrite(output[0] + "_3." + output[1], original[y+adjustment:y + (h-adjustment), x+adjustment:x + (w-adjustment)])
            return (output[0] + "_3." + output[1])
        else:
            cv2.imwrite(output[0] + "_1." + output[1], original[y+adjustment:y + (h-adjustment), x+adjustment:x + (w-adjustment)])
            return (output[0] + "_1." + output[1])   
    except Exception as e:
        print(e)
