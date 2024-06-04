import cv2
import os
import numpy as np

vdo_name = "camera_0028"

vidcap = cv2.VideoCapture(vdo_name+'.mp4')
vidcap.set(cv2.CAP_PROP_POS_FRAMES, 0)
vidcap.set(cv2.CAP_PROP_BUFFERSIZE, 3);
success,image = vidcap.read()
count = 0
success = True
while success:
  success,image = vidcap.read()
  if success:
      img_path = vdo_name+'_det'
      isExist = os.path.exists(img_path)
      if not isExist:
          os.mkdir(img_path)

      cv2.imwrite(img_path+"/frame%d.jpg" % count, image)     # save frame as JPEG file
      if cv2.waitKey(10) == 27:                     # exit if Escape is hit
          break
      count += 1

det_map = {}
det = open(vdo_name+".txt")

for line in det:
    s = line.split(" ")
    frameId = int(float(s[0]))-2
    x = int(float(s[3]))
    y = int(float(s[4]))
    w = int(float(s[5]))
    h = int(float(s[6]))
    bbox = [x,y,w,h,int(float(s[2]))]
    if frameId not in det_map:
        det_map[frameId] = [bbox]
    else:
        det_map[frameId].append(bbox)
#import pdb;pdb.set_trace()
for fID in det_map.keys():



    fpath = "./"+vdo_name+"_det/"
    fpath1 = "./"+vdo_name+"_speed/"
    fname = "frame"+str(fID)

    isExist = os.path.exists(fpath1)
    if not isExist:
      os.mkdir(fpath1)

    img = cv2.imread(fpath+fname+".jpg")

    if fID>0:

        for box in det_map[fID]:
            x = box[0]
            y = box[1]
            w = box[2]
            h = box[3]
            cid = box[4]


            image = cv2.rectangle(img, (x,y), (x+w,y+h), (36,255,12), 2)
            cv2.putText(image, str(cid), (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (36,255,12), 2)
        cv2.imwrite(fpath1+fname+'.jpg', img)




image_folder = "./"+vdo_name+"_speed/"
video_name = vdo_name+'_speed_fix_dz.avi'

images=[]
for i in range(0,3599):  # number of frames in your video
    images.append('frame'+str(i)+'.jpg')
try:
    frame = cv2.imread(os.path.join(image_folder, images[1]))
except:
    frame = cv2.imread(os.path.join(fpath, images[1]))

height, width, layers = frame.shape

fourcc = cv2.VideoWriter_fourcc(*'XVID')
video = cv2.VideoWriter(video_name,fourcc, 11.0, (width,height))

for image in images:

    if os.path.isfile(os.path.join(image_folder, image)):
        video.write(cv2.imread(os.path.join(image_folder, image)))
        cv2.imread(os.path.join(image_folder, image))
    else:
        video.write(cv2.imread(os.path.join(fpath, image)))


cv2.destroyAllWindows()
video.release()