'''
Created by Daniel-Iosif Trubacs on 4/12/2021 for the Robosoc Society Project. This program
contains basic functions for image manipulation and object detection. The more complex functions
are explained by comments

'''

#importing the necessary libraries
import cv2 as cv
import numpy as np
import time

#DEFINIG GLOBAL VARIABLES AND FUNCTIONS USED IN ALL THE CLASSES
#a kernel for dilation and erosion
kernel_noise = np.ones((3,3), np.uint8)

#kernel for closing the picture
kernel_closing = np.ones((7,7), np.uint8)


kernel_noise = np.ones((5,5), np.uint8)

#kernel for closing the picture
kernel_closing = np.ones((9,9), np.uint8)

# for the canny functions (used to find the egdes)
ratio = 5
kernel_size = 5
low_threshold = 15

# the font used for showung different coordinates and positions on the image
font = cv.FONT_HERSHEY_SIMPLEX

# Defining basic functions that are used in all classes and methods
# a function to work out the area between 2 points (in xOy coordinates)
def area (P_1,P_2):
    return abs((P_2[0]-P_1[0])*(P_2[0]-P_1[0]))



# a class containing multiple image manipulation functions
class img_manipulation:
    def __init__(self):
        pass
    
    # img is the image that has to be rescaled and f is the fraction by which
    # the iamge will be rescaled
    def rescale(img,f):
         # getting the new dimensions
         height = int(f*img.shape[0])
         width = int(f*img.shape[1])
         dim = (width,height)
         
         # resizing the image
         img = cv.resize(img,dim,interpolation=cv.INTER_AREA)
         
         #returning the rescaled image
         return img
    
    # a normal resize function /  made only to not use all the cv functions every time 
    def resize(img,height,width):
        # getting the dimensions
        dim = (int(width),int(height))
        
        # resizing the image
        img = cv.resize(img,dim,interpolation=cv.INTER_AREA)
        
        #returning the resized image
        return img
    
    # basic edge detection for img. Fast and multiple objects in an image can be found,
    # but it is not very accurate.
    def basic_edge_detection(img):
        # changing the image to gray
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
        
        # bluring the image
        img = cv.blur(img, (3,3))
       
        # using the canny method
        img = cv.Canny(img, low_threshold, low_threshold*ratio, kernel_size)
        
        #closing the image
        #img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_closing)
        
        return img
    
    # Slower but more accurate. Helpful for finding one object in a background
    def advanced_edge_detection(img):
        # changing the image to gray
        img = cv.cvtColor(img, cv.COLOR_BGR2GRAY) 
        
        # bluring the image
        img = cv.blur(img, (3,3))
        
        #eroding the image
        img = cv.erode(img, kernel_noise)
        
        # using the canny method
        img = cv.Canny(img, low_threshold, low_threshold*ratio, kernel_size)
        
        #closing the image
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_closing)
        
        #dilating the image to pronounce the edges
        img = cv.dilate(img, kernel_noise, iterations=1)
        
        #eroding the image
        img = cv.erode(img, kernel_noise)
        
        #closing the image
        img = cv.morphologyEx(img, cv.MORPH_CLOSE, kernel_closing)
        
        #dilating the image to pronounce the edges
        img = cv.dilate(img, kernel_noise, iterations=1)
        
        
        return img
        


# a class containing multiple object detection functions
class obj_det:
    def __init__(self):
        pass
    
    '''
    A function to find the objects from a contours list. The contour list given should come 
    from the cv.findContours method. width and height of the original image are still 
    needed for some image manipulation.
    Inputs : contours - a list of the found contours
             n_ob - the number of objects
             width - width of the original image
             height - height of the original iamge
    Outputs: sorted_objects - an array containting the important coordinates of all the objects
                              sorted by their corresponding area.
    '''
    def find_objects(contours,width,height):
       # the array of objects found
       objects = []
       
       # going through all the contours found
       for i in range(len(contours)):
         # reshaping the contours in a list of 2d points
         shape = contours[i].shape
         aux = np.reshape(contours[i],(shape[0],shape[2]))
        
         #finding the 2 points that define the rectangle 
         P_min = (max(aux[:,0]),max(aux[:,1]))
         P_max = (min(aux[:,0]),min(aux[:,1]))
         
         #the area inside the contour
         cont_area  = area(P_min,P_max)
         
         # the objects found
         if 0.001*width*height < cont_area < 0.95*width*height:
          objects.append((P_min,P_max,cont_area))
       
       #sorting the array of objects
       aux = sorted(objects,key=lambda x:x[2]) 
       m = len(aux)
       sorted_objects = [aux[m-i-1] for i in range(m)]
       
       #returning the points that define the rectangles (objects)
       return sorted_objects
    
    
    '''
    A function to show all the objects found on the original image. The shape of the object
    should be [P_0,P_1,P_C] where P_0 are the coordinates of the lowest point, P_1 are the 
    are the coordinates of the highest point and P_C are the coordinates of the centre
    Inputs : img - the orignal img (has to be rgb)
             objects - the list of objects found
    Outputs: img - a copy of the original image but with all the objects emphasized
    '''
    def obj_show(img,objects,show_center,color):
        
        # the number of objects found
        m = len(objects)
        # a gradient to differentiate between the objects
        if m > 0:
         gradient = int(255/m)
         #showing the number of objects found
         cv.putText(img,str(m)+' OBJECTS FOUND',
               (20,50), font, 0.75,(0,0,0),2,cv.LINE_AA)
         for i in range(m):
            # unpacking the points
            p_0 = (int(objects[i][0][0]),(int(objects[i][0][1])))
            p_1 = (int(objects[i][1][0]),(int(objects[i][1][1])))
            if show_center:
             x_ob = int(objects[i][2][0])
             y_ob = int(objects[i][2][1])
            
            
            #showing the rectangle that contains the object
            cv.rectangle(img,p_0,p_1,color,2) 

            #showing the x and y in the iamge
            if show_center:
             cv.putText(img,'x='+str(x_ob),
                   (x_ob,y_ob), font, 0.5,(0,gradient*i,255-gradient*i),2,cv.LINE_AA)
             cv.putText(img,'y='+str(y_ob),
                   (x_ob,y_ob+20), font, 0.5,(0,gradient*i,255-gradient*i),2,cv.LINE_AA)
        else:
            cv.putText(img,'No objects found',
                  (int(img.shape[0]/2),int(img.shape[1]/2)), font, 0.5,(0,0,255),2,cv.LINE_AA)
   
        #returning the image with the object found
        return img
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
         


