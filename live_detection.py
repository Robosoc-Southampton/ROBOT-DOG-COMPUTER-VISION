'''
Created by Daniel-Iosif Trubacs on 4/12/2021 for the Robosoc Society Project. This program
contains basic functions for static object detection from a live feed. It cannot be used without 
the object detection module. 

All objects (objects found in an image) are given by:
[P_min,P_max,Area] or [P_min,P_Max,P_c] where P_min represents the min point (relative to a coordinate axis)
P_max is max point, Area- the area that defines the rectangle and P_c is the center point of 
the object


The more complex functions are explained by doc strings

'''

#importing the necessary libraries
import cv2 as cv
import numpy as np
from object_detection import img_manipulation,obj_det

#GLOBAL FUNCTIONS USED IN ALL CLASSES AND METHODS

# a function to wo work out the distance between 2 points
def distance(P_0,P_1):
    distance = np.sqrt((P_0[0]-P_1[0])**2+(P_0[1]-P_1[1])**2)
    return distance

# a function that sorts a list of objects by their areas
def sorting(Objects):
    aux = sorted(Objects,key=lambda x:x[2]) 
    m = len(aux)
    sorted_objects = [aux[m-i-1] for i in range(m)]
    return sorted_objects


# a function to work out the center of an object
def center(Object):
    x_c = int((Object[0][0]+Object[1][0])/2)
    y_c = int((Object[0][1]+Object[1][1])/2)
    P_c = (x_c,y_c)
    return P_c

# a function to check if 2 found represent relatively the same 'real' object
def closure(object_1,object_2):
    # area of the 2 objects
    Area_1 = object_1[2]
    Area_2 = object_1[2]
    
    # checking if the areas are not too different
    if Area_2 > 4*Area_1 or Area_1 > 4*Area_1:
        return False
    
    # center of the 2 objects
    P_c_1 = center(object_1)
    P_c_2 = center(object_2)
    
    # checking if the center of the 2 objects are  not too different
    if distance(P_c_1,P_c_2) > 2*np.sqrt(Area_1):
        return False
    if distance(P_c_1,P_c_2) > 2*np.sqrt(Area_2):
        return False
    
    # returning True if all the above conditions are met
    # expecteing that the 2 found objectr epresent relatively the same 'real' object
    else:
        return True

# a function to superpose a list of Objects (return an 'average' like object) 
# ALL THE OBJECTS GIVEN AS INPUTS HAVE TO CHECK THE CLOSURE RULE
def superpose(Objects):
    P_0 = np.array([0,0])
    P_1 = np.array([0,0])
    P_c = np.array([0,0])
    
    m = len(Objects)
    for i in range(m):
          P_0 = P_0+np.array(Objects[i][0])   
          P_1 = P_1+np.array(Objects[i][1])   
          P_c = P_c+np.array(center(Objects[i]))       
    P_0[0] = int(P_0[0]/m)
    P_0[1] = int(P_0[1]/m)
    P_1[0] = int(P_1[0]/m)
    P_1[1] = int(P_1[1]/m)
    P_c[0] = int(P_c[0]/m)
    P_c[1] = int(P_c[1]/m)
    
    return (P_0,P_1,P_c)



# specifically created to unpack the important coordinates of an object
def unpack(objects):
    #showing the rectangle that contains the object
    p_0 = np.array(objects[0])
    p_1 = np.array(objects[1])
    
    #finding the centre of the boject
    x_ob = int((p_0[0]+p_1[0])/2)
    y_ob = int((p_0[1]+p_1[1])/2)
    p_c = np.array([x_ob,y_ob])
    
    # returning p_0,p_1 and p_c
    return p_0,p_1,p_c
    
   

class live_det:
    def __init__(self):
        pass
    
    
    # finding the objects in a frame
    def find_frame_obj(frame):
        img_edge = img_manipulation.basic_edge_detection(frame)
        
        #finding the contours for all objects 
        contours, hierarchy = cv.findContours(img_edge, cv.RETR_LIST, cv.CHAIN_APPROX_NONE)
        
        #finding the objects given by the contours
        objects = obj_det.find_objects(contours,frame.shape[0],frame.shape[1])
        
        return objects
        
    '''
    A function to find the static objects from a live feed. It takes as input 
    a list of objects from consecutives frames and return a 'superpose' of the
    expected 'real' objects
    Inputs: Objects -  a list of objects (has to be of type [P_min,P_max,Area]
            n_ob -  the number of objects 
    Outputs: P- an array of position of all the objects found
    '''
    def find_static_obj(Objects,n_ob):
        # just a dummy variable 
        Objects_0 = Objects[0]
        # putting all found objects in a sinle list
        for i in range(len(Objects)-1):
            Objects_0 = Objects_0 + Objects[i+1]
        
        # sorting the list/ checking that the objects are sorted by area
        sorted_Objects = sorting(Objects_0)
        
        #the list that will contain all supeposed objects
        objects = []
        
        #the list used to contain all objects that are expected to be the same 'real' object
        aux_objects = []
        m = len(sorted_Objects)
        for i in range(m):
            # checking if the condition is already satisifed
            if len(objects) > n_ob-1:
                break
            # checking if there are no found objects atm
            n = len(aux_objects)
            if n == 0:
                aux_objects.append(sorted_Objects[i])
            
            # if the closure condition is satisfied the object found is appended to aux_objects
            # otherwise, all objects in aux_objects are superposed and appended to objects
            else:
                if closure(aux_objects[n-1],sorted_Objects[i]):
                    aux_objects.append(sorted_Objects[i])
                else:
                    sup_object = superpose(aux_objects)
                    objects.append(sup_object)
                    aux_objects=[]
                    #aux_objects.append(sorted_Objects[i])
                    
        # returning the expected found 'real' objects            
        return objects
         
            
            
        
        
      
    
   
    
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
        
    
         




