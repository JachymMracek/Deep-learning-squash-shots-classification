import cv2
from dataclasses import dataclass

@dataclass(frozen=True)  
class Click:
    x: int
    y: int

def middle(point_value1,point_value2 = None):
    
    if point_value2 is None:
        return point_value1 / 2
    
    return (point_value1 + point_value2) / 2

def click_event(event, x, y, flags, params:list):
    if event == cv2.EVENT_LBUTTONDOWN:
        print(x,y)
        click = Click(x,y)
        params.append(click)

def click_on_image(img,click_count,clicks):
    # https://www.geeksforgeeks.org/python/displaying-the-coordinates-of-the-points-clicked-on-the-image-using-python-opencv/
    
    cv2.imshow("image",img) # type: ignore
    cv2.setMouseCallback('image',click_event,clicks) # type: ignore

    for _ in range(click_count):
        cv2.waitKey(0)
    
      
    cv2.destroyAllWindows()
