import cv2
import imutils
from PIL import Image  

def scaled(cropped, background, background_width, frame_width, output, filename):
    # Read in cropped image.
    frame = cv2.imread(cropped)

    # Find background resolution.
    backY, backX, channels = background.shape
    # Find background center point.
    midY, midX = backY/3, backX/2

    # Finding pixels per cm of background.
    pxpercm = backY/background_width

    # Rescaling the frame.
    # Finding the frames new scale width.
    scaled_width = frame_width * pxpercm
    # Rescaling frame using new width.
    frame = imutils.resize(frame, width = int(scaled_width))

    # Find new frame resolution.
    frameY, frameX, channels_f = frame.shape
    # Finding frame mid point.
    pasteY, pasteX =  midY - frameY/2, midX - frameX/2

    # Convert back to rgb.
    background = cv2.cvtColor(background, cv2.COLOR_BGR2RGB)
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Open images using PIL.
    background2 = Image.fromarray(background)
    frame2 = Image.fromarray(frame)

    # Paste frame onto background.
    background2.paste(frame2, (int(pasteX), int(pasteY)))

    # Save the new scaled photograph.
    background2.save(output + "/" + filename + "_4.jpg")