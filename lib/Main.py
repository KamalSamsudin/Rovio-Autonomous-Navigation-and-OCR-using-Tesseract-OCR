import cv2
import numpy as np
from lib import Rovio
import urllib
import time
from time import sleep
from PIL import Image
import pytesseract
import cgi
from skimage import io
from threading import Thread
import time
import winsound
import urllib2
from time import sleep

ip_of_rovio = '192.168.10.18'
save_file_path = 'D:/works/Third year/Intelligent Robotics/Rovio/'
pytesseract.pytesseract.tesseract_cmd = 'C:/Program Files (x86)/Tesseract-OCR/tesseract'
tessdata_dir_config = '--tessdata-dir "C:/Program Files (x86)/Tesseract-OCR/tessdata"'


class rovioControl(object):
    def __init__(self, url, username, password, port=80):
        self.rovio = Rovio(url, username=username, password=password,port=port)

    def main(self):
        #self.rovio.head_up()
        #frame = self.rovio.camera.get_frame()
        #print(frame)
        self.rovio.head_down()
        lower_flag = False
        human_flag = False
        current_time = lambda: int(round(time.time() * 1000))
        old_image = []
        start_time = 0
        while(True):
            check = main2()
            if(check == 1):
                continue

            self.rovio.head_down()
            battery,charging  = self.rovio.battery()
            start = time.time()
            #img = io.imread('http://192.168.10.18/Jpeg/CamImg0000.jpg')

            response = urllib.urlopen('http://192.168.10.18/Jpeg/CamImg.jpg')
            img_array = np.asarray(bytearray(response.read()), dtype=np.uint8)
            img = cv2.imdecode(img_array, 1)

            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            gaussian = cv2.GaussianBlur(img, (5, 5), 0)

            if(self.get_image_difference(np.asarray(gaussian,dtype="uint8"),np.asarray(old_image,dtype="uint8")) < 0.1):
                print('I think i am stuck')
                if(start_time == 0):
                    start_time = current_time()
            else:
                start_time = 0

            diff = current_time() - start_time
            if(start_time != 0 and diff > 200):
                self.rovio.backward()
                self.rovio.backward()
                self.rovio.backward()
                self.rovio.backward()
                self.rovio.backward()
                self.rovio.backward()
                self.rovio.backward()
                self.rovio.backward()
                print('Rotating since stuck')
                #self.dancer()
                self.rovio.rotate_right(angle=30)
                start_time = 0

            print('Delays :',time.time() - start)

            battery = 100 * battery / 130.
            bs = "Battery: %0.1f" % battery

            cv2.putText(gaussian, bs, (20, 20),cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0))
            if cv2.waitKey(1) & 0xFF == ord('q'):
                exit()
            cv2.imshow("Ahmad's Group", gaussian)
            gray = cv2.cvtColor(gaussian,cv2.COLOR_RGB2GRAY)
            th = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 115, 1)

            if(self.rovio.obstacle()):
                #self.rovio.backward()
                #self.rovio.backward()
                #self.rovio.backward()
                #self.rovio.backward()
                #self.rovio.backward()
                print('obstacle detected')
                #self.dancer()
                self.rovio.rotate_right(angle=45)
                continue

            #app.rovio.patrol()
            print('moving forward')
            self.rovio.forward()
            self.rovio.forward()
            self.rovio.forward()
            self.rovio.forward()
            self.rovio.forward()
            print('End of move')

            edge = cv2.Canny(gaussian,100,200)
            cv2.imshow('Edge',edge)
            old_image = gaussian
            sleep(25/1000)

    def get_image_difference(self,image_1, image_2):
        first_image_hist = cv2.calcHist([image_1], [0], None, [256], [0, 256])
        second_image_hist = cv2.calcHist([image_2], [0], None, [256], [0, 256])

        img_hist_diff = cv2.compareHist(first_image_hist, second_image_hist, cv2.HISTCMP_BHATTACHARYYA)
        img_template_probability_match = cv2.matchTemplate(first_image_hist, second_image_hist, cv2.TM_CCOEFF_NORMED)[0][0]
        img_template_diff = 1 - img_template_probability_match

        # taking only 10% of histogram diff, since it's less accurate than template method
        commutative_image_diff = (img_hist_diff / 10) + img_template_diff
        return commutative_image_diff

    def dancer(self):
        dance_sequence = [1, 2, 3, 4, 5]
        winsound.PlaySound('Fallout 4 - Intro Cinematic Theme Music (NO VOICE).mp3', winsound.SND_FILENAME)
        for x in dance_sequence:
            # x = random.randint(1,5)
            if x == 1:
                self.rovio.head_up()
                time.sleep(1)
            elif x == 2:
                self.rovio.rotate_left(angle=360, speed=2)
                time.sleep(3)
            elif x == 3:
                self.rovio.head_middle()
                self.rovio.right()
                self.rovio.right()
                self.rovio.right()
                time.sleep(0.5)
                self.rovio.left()
                self.rovio.left()
                self.rovio.left()
                time.sleep(1)
            elif x == 4:
                self.rovio.rotate_right(angle=360, speed=2)
                time.sleep(3)
            elif x == 5:
                self.rovio.head_down()
                #time.sleep(1)


# image processing and ocr
def ocr(image, num):
    rovio = Rovio('192.168.10.18', 'myname', '12345')
    # open image
    image = cv2.imread(image)

    # turn image to binary
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]

    # deskew text function
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]

    # the `cv2.minAreaRect` function returns values in the
    # range [-90, 0); as the rectangle rotates clockwise the
    # returned angle trends to 0 -- in this special case we
    # need to add 90 degrees to the angle
    if angle < -45:
        angle = -(90 + angle)

    # otherwise, just take the inverse of the angle to make it positive
    else:
        angle = -angle

    # rotate the image to deskew it
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)

    # draw the correction angle on the image so we can validate it
    cv2.putText(rotated, "Angle: {:.2f} degrees".format(angle),
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    # show the output image
    print("[INFO] angle: {:.3f}".format(angle))
    # end of text deskew function

    # save rotated image
    imag = 'D:/works/Third year/Intelligent Robotics/Rovio/thresh' + str(num) + '.jpg'
    cv2.imwrite('D:/works/Third year/Intelligent Robotics/Rovio/skewed' + str(num) + '.jpg', rotated)

    # turn image to binary and save image
    gray = cv2.cvtColor(rotated, cv2.COLOR_BGR2GRAY)
    thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    cv2.imwrite('D:/works/Third year/Intelligent Robotics/Rovio/thresh' + str(num) + '.jpg', thresh)

    # Simple image to string ocr, lang=eng
    text = pytesseract.image_to_string(Image.open(imag), lang='eng', config=tessdata_dir_config)

    # print output to console
    print(num,':\n',text)
    text = text.lower()
    val = 0
    if "dance" in text:
        val = 1
    if "rotate" in text:
        val = 2
    if "stop" in text:
        val = 99
    if "nod" in text:
        val = 3

    url = '192.168.10.18'
    user = 'myname'
    password = "12345"
    app = rovioControl(url, user, password)

    # if val value to run command
    if val == 1:
        #file.write(1)
        print "Order Received. I'm dancing till dropping..."

        app.dancer()
        return 1
    if val == 2:
        rovio.rotate_left(angle=360)
        return 1
    if val == 3:
        rovio.head_middle()
        rovio.head_middle()
        rovio.head_up()
        rovio.head_up()
        rovio.head_middle()
        rovio.head_middle()
        rovio.head_down()
        rovio.head_down()
        rovio.head_middle()
        rovio.head_middle()
        rovio.head_up()
        rovio.head_up()
        rovio.head_up()
        rovio.head_middle()
        rovio.head_middle()
        rovio.head_down()
        rovio.head_down()
        rovio.head_middle()
        rovio.head_middle()
        rovio.head_up()
        rovio.head_up()
        rovio.head_up()
        rovio.head_middle()
        rovio.head_middle()
        rovio.head_down()
        rovio.head_down()
        rovio.head_middle()
        rovio.head_middle()
        rovio.head_up()
        rovio.head_up()
        rovio.head_up()
        rovio.head_middle()
        rovio.head_middle()
        rovio.head_down()
        rovio.head_down()
        return 1
    if val == 99:
        rovio.head_down()
        exit()

    else:
        return 0

# function to save image from rovio camera per second
def downloadImage(pictureName):
    global ip_of_rovio
    global save_file_path
    url = 'http://'+ip_of_rovio+'/Jpeg/CamImg0000.jpg'
    request = urllib2.Request(url)
    pic = urllib2.urlopen(request)
    filePath = save_file_path + str(pictureName) + '.jpg'
    with open(filePath, 'wb') as localFile:
        localFile.write(pic.read())

    # send image for preprocess and ocr
    return ocr(filePath, pictureName)

# loop to call downloadImage function
def main2():
    pictureVal = 0
    if (pictureVal < 10):
        pictureVal += 1
    else:
        pictureVal = 0
    j = downloadImage(pictureVal)
    sleep(1)
    return j

if __name__ == "__main__":
    url = '192.168.10.18'
    user = 'myname'
    password = "12345"
    app = rovioControl(url, user, password)
    app.main()
