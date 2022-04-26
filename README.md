# Automated Photographic Processing (A.P.P.)

### Final Year Project
### Michael Pound, 20085540

_____________________________________________
_____________________________________________

This is an application that can be used to automate the photographic work done by professional photographers for artwork auction houses.

## GUI Overview:

![][gui]
_____________________________________________

## Application Functionality:

### Automated File Management
This application handles file saving by extracting the lot numbers using Optical Character Recognition, else a simple numeric naming standard can be used.
### Automated Artwork Frame Cropping
This is where the front and back of the frame are cropped out of the background wall they are hung on.
### Automated Artwork Extraction
This is where the artwork is cleanly cropped out of the frame.
### Framework Classification
This application has a trained CNN model for frame classification, this is used to assist in optimising the cropping accuracy.
### OCR Lot Number Extraction
Optical Character Recognition is used to extract lot numbers from images of artwork lot labels.
### OCR Artist Note Extraction
Optical Character Recognition is used to extract any text found on the back of the artwork frame.
## Scaled Photographs
This is where the cropped artwork is inserted into a suitable scaled environment.

_____________________________________________

## Technologies Used:
 + Python
 + Tkinter
 + Tesseract OCR
 + TensorFlow Keras
 + OpenCV
 + Pillow
 + Pandas
 + Imutils
 + NumPy
 + OS
 + RE

 [gui]: gui.png