# photo-id-ocr
Assignment for Signzy

Uses OpenCV 3.0.0 with Python 2.7.11
Note: findContours() function in OpenCV 3.0.0 has changed from OpenCV 2 and now returns a three tuple instead of 2. This function as well as others may break functionality if used with OpenCV 2.4.x etc.

### Usage
First run preprocess.py to get the required ROI as:
```
$ python preprocess.py -i <path-to-image>
```
Then run segment.py to obtain the text regions and face:
```
$ python segment.py -i <path-to-image>
```
