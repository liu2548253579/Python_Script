import cv2
import glob
import re

photo = 0 #当前显示的照片标号
marktimes = 0 #鼠标点击次数
key_value = 0 #当前键入的键值
text_position = [0,1,2,3] #储存四个坐标的数组
text_content = 0 #输入文本的内容
x_position = [0,1,2,3]#四个X轴坐标数据
y_position = [0,1,2,3]#四个Y轴坐标数据
#fileaddr = 'C:\\Users\\25482\\Desktop\\'
#photoaddr = 'C:\\Users\\25482\\Desktop\\mark_pictures\\pictures\\'

def text_create(name):#创建指定文件的文本(如若存在则会覆盖原文件)
    text_path = fileaddr+'\\mark_pictures\\texts\\'    
    full_path = text_path + name + '.txt' 
    file = open(full_path,'w')             
    file.close() 


def text_write(num,msg):#打开路径下的标号文件并写入内容
    file = open(fileaddr+'\\mark_pictures\\texts\\'+txt_name+'.txt','w')             
    file.write(msg) 
    file.close() 
    

def mouse(event, x, y, flags, param):#鼠标点击事件并返回鼠标在图中的坐标
    if event == cv2.EVENT_LBUTTONDOWN:
        global marktimes
        global text_position
        xy = "%d,%d" % (x, y)
        x_position[marktimes-1]=x
        y_position[marktimes-1]=y
        cv2.circle(img, (x, y), 1, (0, 0, 255), thickness = -1)
        cv2.putText(img, xy, (x, y), cv2.FONT_HERSHEY_PLAIN,1.0, (0, 0, 255), thickness = 1)
        marktimes = marktimes + 1
        if marktimes >= 5:
            marktimes = 1
        text_position[marktimes-1]= xy


fileaddr = input("输入程序文件夹的绝对路径：")           
photoaddr = input("输入图片的绝对路径：")
image_photo=glob.glob(photoaddr+'\\*.png')

while True :
    global txt_name
    txt_name=str(re.findall(r'L_([^"]+).png',image_photo[photo]))
    
    img = cv2.imread(image_photo[photo])
    cv2.namedWindow("image")
    
    image_info="Page:%d click:%d"%(photo,marktimes)
    cv2.putText(img, image_info, (5, 30), cv2.FONT_HERSHEY_PLAIN,2, (0, 0, 255), thickness = 1) 
    
    cv2.line(img, (x_position[-1],y_position[-1]), (x_position[0],y_position[0]), (0, 255, 255))   
    cv2.line(img, (x_position[1],y_position[1]), (x_position[2],y_position[2]), (0, 255, 255)) 
    
    x1y1 = "%d,%d"%(x_position[-1],y_position[-1]); x2y2 = "%d,%d"%(x_position[0],y_position[0])
    x3y3 = "%d,%d"%(x_position[1],y_position[1]); x4y4 = "%d,%d"%(x_position[2],y_position[2])
    cv2.putText(img, x1y1, (x_position[-1],y_position[-1]), cv2.FONT_HERSHEY_PLAIN,1.0, (0, 0, 255), thickness = 1)
    cv2.putText(img, x2y2, (x_position[0],y_position[0]), cv2.FONT_HERSHEY_PLAIN,1.0, (0, 0, 255), thickness = 1)
    cv2.putText(img, x3y3, (x_position[1],y_position[1]), cv2.FONT_HERSHEY_PLAIN,1.0, (0, 0, 255), thickness = 1)
    cv2.putText(img, x4y4, (x_position[2],y_position[2]), cv2.FONT_HERSHEY_PLAIN,1.0, (0, 0, 255), thickness = 1)  
        
    cv2.imshow("image", img)
    cv2.setMouseCallback("image", mouse)
    key_value=cv2.waitKey(10) & 0xFF #读取键入值
    
    
    if key_value == ord('l'):#l键上一张
        photo = photo-1 
        if photo <= 0:
            photo = 0;
        x_position[-1]=0 ;x_position[0]=0 ;x_position[1]=0 ;x_position[2]=0 
        y_position[-1]=0 ;y_position[0]=0 ;y_position[1]=0 ;y_position[2]=0 
        text_create(txt_name) # 创建文本(此处是为了清除错误输入) 
        
    if key_value == ord('c'):#c键重新标定当前照片
        x_position[-1]=0 ;x_position[0]=0 ;x_position[1]=0 ;x_position[2]=0 
        y_position[-1]=0 ;y_position[0]=0 ;y_position[1]=0 ;y_position[2]=0 
        marktimes=0
            
    if key_value == ord('n'):#n键下一张
        text_create(txt_name) # 创建文本
        text_content="line1:start(%s)  end(%s)\n\nline2:start(%s)  end(%s)\n"%(text_position[0],text_position[1],text_position[2],text_position[3])
        if marktimes == 4:
            text_write(txt_name,text_content)#写入内容
        x_position[-1]=0 ;x_position[0]=0 ;x_position[1]=0 ;x_position[2]=0 
        y_position[-1]=0 ;y_position[0]=0 ;y_position[1]=0 ;y_position[2]=0 
        photo = photo+1
        marktimes=0
        
    if key_value == 27:#esc键退出
        text_create(txt_name) # 创建文本
        text_content="line1:start(%s)  end(%s)\n\nline2:start(%s)  end(%s)\n"%(text_position[0],text_position[1],text_position[2],text_position[3])
        if marktimes == 4:
            text_write(txt_name,text_content)#写入内容
        cv2.destroyAllWindows()
        break