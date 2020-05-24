import datetime
import dlib, cv2, os
import matplotlib.pyplot as plt

print(datetime.datetime.now())
detector = dlib.cnn_face_detection_model_v1('dogHeadDetector.dat')
cropped=[]

for i in range(4):
    img_path = str(i)+'.jpg'
    filename, ext = os.path.splitext(os.path.basename(img_path))
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  
    img_result = img.copy()
    img_size = img.shape
    img_sizex = 350/img_size[0]
    img_sizey = 350/img_size[1]
    img = cv2.resize(img, dsize=None, fx=img_sizex, fy=img_sizey) 
    dets = detector(img, upsample_num_times=0) 
    for i, d in enumerate(dets):
        print("Detection {}: Left: {} Top: {} Right: {} Bottom: {} Confidence: {}".format(i, d.rect.left(), d.rect.top(), d.rect.right(), d.rect.bottom(), d.confidence))
        x1, y1 = d.rect.left(), d.rect.top()
        x2, y2 = d.rect.right(), d.rect.bottom()
        #cv2.rectangle(img_result, pt1=( int(max(x1-10,0)/img_sizex),  int(max(y1-10,0)/img_sizey)), pt2=( int(min(x2+10,299)/img_sizex), int(min(y2+10,299)/img_sizey)), thickness=2, color=(255,0,0), lineType=cv2.LINE_AA)
        x1 =int(max(x1-10,0)/img_sizex)
        x2 =int(min(x2+10,249)/img_sizex)
        y1 =int(max(y1-10,0)/img_sizey)
        y2 =int(min(y2+10,249)/img_sizey)
        spanmax = max(x2-x1,y2-y1)
        cropped = img_result[y1:y1+spanmax,x1:x1+spanmax]
    if(len(cropped)):                           #输出方形裁剪结果 没有狗脸返回原图 
        plt.imshow(cropped)
        cropped=[]
    else:
        plt.imshow(img_result)
    plt.show()
        