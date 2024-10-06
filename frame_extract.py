# Importing all necessary libraries 
import cv2 
import os 

def frame_extract(input_file):

    # Read the video from specified path 
    cam = cv2.VideoCapture(input_file) 
    
    try: 
        
        # creating a folder named data 
        if not os.path.exists('data'): 
            os.makedirs('data/frame') 
    
    # if not created then raise error 
    except OSError: 
        print ('Error: Creating directory of data') 
    
    # frame 
    currentframe = 0
    
    while(True): 
        
        # reading from frame 
        ret,frame = cam.read() 
    
        if ret and (currentframe %30== 0): 
            # if video is still left continue creating images 
            name = './data/frame' + str(currentframe) + '.jpg'
            print ('Creating...' + name) 
    
            # writing the extracted images 
            cv2.imwrite(name, frame) 
    
            # increasing counter so that it will 
            # show how many frames are created 
            currentframe += 1

        elif ret: 
            currentframe += 1
        else:
            break
        

    
    # Release all space and windows once done 
    cam.release() 
    cv2.destroyAllWindows() 
