import cv2

def art_crop(input, output, type_one):
    # Read in cropped frame image.
    img = cv2.imread(input)
    # Creae copy of original.
    original = img

    # Image pre-processing for reducing image detail.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (9, 9), 0)
    img = cv2.threshold(img, 180, 255, cv2.THRESH_BINARY)[1]
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 2)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Building colour threshold.
    val, col_threshold = cv2.threshold(img, thresh=100, maxval=255, type=cv2.THRESH_BINARY_INV)

    # Finding all contours in processed image.
    contours, hierarchy = cv2.findContours(col_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialise 3 largest contours.
    select1 = (0, 0, 0, 0)
    select2 = (0, 0, 0, 0)
    select3 = (0, 0, 0, 0)

    # Finding the 3 largest contours.
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        contour_area = w * h
        select1_area = select1[2] * select1[3]

        if contour_area > select1_area:
            select3 = select2
            select2 = select1
            select1 = x, y, w, h

    x, y, w, h = select1
    x2, y2, w2, h2 = select2
    x3, y3, w3, h3 = select3

    # Adjustment value for cleaning up the cropped image edges.
    adjustment = 10

    # Crop the image to the wanted contour.
    try:
        output = output.split('.')

        if type_one == True:
            cv2.imwrite(output[0] + "_2." + output[1], original[y2+adjustment:y2 + (h2-adjustment), x2+adjustment:x2 + (w2-adjustment)])
            return (output[0] + "_2." + output[1])
        else:
            cv2.imwrite(output[0] + "_2." + output[1], original[y3+adjustment:y3 + (h3-adjustment), x3+adjustment:x3 + (w3-adjustment)])
            return (output[0] + "_2." + output[1])


    except Exception as e:
        # print(e)
        print("Error encountered with artwork cropping.")
        return False


def dark_art_crop(input, output, type_one):
    # Read in cropped frame image.
    img = cv2.imread(input)
    # Creae copy of original.
    original = img

    # Image pre-processing for reducing image detail.
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.GaussianBlur(img, (13, 13), 0)
    img = cv2.threshold(img, 135, 255, cv2.THRESH_BINARY)[1]
    img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 43, 1)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (6, 6))
    img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)

    # Building colour threshold.
    val, col_threshold = cv2.threshold(img, thresh=100, maxval=255, type=cv2.THRESH_BINARY_INV)

    # Finding all contours in processed image.
    contours, hierarchy = cv2.findContours(col_threshold, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
    
    # Initialise 3 largest contours.
    select1 = (0, 0, 0, 0)
    select2 = (0, 0, 0, 0)
    select3 = (0, 0, 0, 0)

    # Finding the 3 largest contours.
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        contour_area = w * h
        select1_area = select1[2] * select1[3]

        if contour_area > select1_area:
            select3 = select2
            select2 = select1
            select1 = x, y, w, h

    x, y, w, h = select1
    x2, y2, w2, h2 = select2
    x3, y3, w3, h3 = select3

    # Adjustment value for cleaning up the cropped image edges.
    adjustment = 10

    # Crop the image to the wanted contour.
    try:
        output = output.split('.')

        if type_one == True:
            cv2.imwrite(output[0] + "_2." + output[1], original[y2+adjustment:y2 + (h2-adjustment), x2+adjustment:x2 + (w2-adjustment)])
            return (output[0] + "_2." + output[1])
        else:
            cv2.imwrite(output[0] + "_2." + output[1], original[y3+adjustment:y3 + (h3-adjustment), x3+adjustment:x3 + (w3-adjustment)])
            return (output[0] + "_2." + output[1])


    except Exception as e:
        # print(e)
        print("Error encountered with artwork cropping.")
        return False
