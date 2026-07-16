import random
from dataclasses import dataclass
import numpy
import cv2
from enum import Enum
import os
import math
import argparse

argument_parser = argparse.ArgumentParser()
argument_parser.add_argument("--number_of_synthetic_shots",default = "5000", help="How many synthetic shots do you want for each class")
argument_parser.add_argument("--phase_folder",default = r"", help="Dataset where the synthetic data will be created, It creates for the given phase so folder needs be for train/val or test")

SEED = 42

random.seed(SEED)

@dataclass(frozen = True)
class Point:
    x : int
    y : int

class SyntheticShot:
    HEIGHT = 360
    WIDTH = 640

class SynthehticBoast(SyntheticShot):
    
    def __init__(self) -> None:
        super().__init__()
    
    class Side(Enum):
        LEFT = 0
        RIGHT = 1
    
    @classmethod
    def get_start_position(cls,middle_y):
        x_start = random.randint(0,cls.WIDTH) 
        y_start = random.randint(middle_y,cls.HEIGHT)
        
        return Point(x_start,y_start)
    
    @classmethod
    def get_end_position(cls,middle_y,middle_x,side):
        x_end = float("inf") 
        y_end = random.randint(0,middle_y)
        
        if side.value == cls.Side.RIGHT.value:
            x_end = random.randint(0,middle_x)
        
        else:
            x_end = random.randint(middle_x,cls.WIDTH)
        
        return Point(x_end,y_end)
    
    @classmethod
    def get_side(cls,start_position_x):
        
        if start_position_x > cls.WIDTH / 2:
            return cls.Side.RIGHT
        
        return cls.Side.LEFT
    
    @classmethod
    def _draw_ball_not_seen_before_front_wall_hit(cls,image_boast_trajectory_image,start_position_ball,end_position_ball,count_of_points_curve,side,
                                                  SMALL_BALL_MOVE = 30):
        
        cv2.arrowedLine(image_boast_trajectory_image,(start_position_ball.x,start_position_ball.y),(end_position_ball.x,end_position_ball.y),(255,0,0),1)
        current_ball_position = (end_position_ball.x,end_position_ball.y)
        acceppt_points = 2
        
        while count_of_points_curve > acceppt_points:
            move_x = random.randint(0,SMALL_BALL_MOVE)
            move_y = random.randint(-SMALL_BALL_MOVE,0)
            
            if side.value == cls.Side.RIGHT.value:
                move_x = -move_x
            
            new_ball_position = (current_ball_position[0] + move_x,current_ball_position[1] + move_y)
            
            if new_ball_position[0] > cls.WIDTH or new_ball_position[1] > cls.HEIGHT or new_ball_position[0] < 0 or new_ball_position[1] < 0:
                continue
            
            cv2.arrowedLine(image_boast_trajectory_image,current_ball_position,new_ball_position,(255,0,0),4)
            
            current_ball_position = new_ball_position
            
            acceppt_points += 1
    
    @classmethod
    def _draw_ball_seen_before_front_wall_hit(cls,start_position_ball,end_position_ball,count_of_points_curve,side,image_boast_trajectory_image,MAX_BALL_MOVE = 150):
        current_ball_position = (start_position_ball.x,start_position_ball.y)
        hit_side_wall = False
        acceppt_points = 2
        
        while count_of_points_curve > acceppt_points:
            
            move_x = random.randint(0,MAX_BALL_MOVE)
            move_y = random.randint(-MAX_BALL_MOVE,0)
            
            if not hit_side_wall and side.value == cls.Side.LEFT.value:
                move_x = -move_x
                hit_side_wall = True
            
            elif not hit_side_wall and side.value == cls.Side.RIGHT.value:
                hit_side_wall = True
            
            elif hit_side_wall and side.value == cls.Side.RIGHT.value:
                move_x = -move_x
            
            new_ball_position = (current_ball_position[0] + move_x,current_ball_position[1] + move_y)
            
            if new_ball_position[0] > cls.WIDTH or new_ball_position[1] > cls.HEIGHT or new_ball_position[0] < 0 or new_ball_position[1] < 0:
                continue
            
            cv2.arrowedLine(image_boast_trajectory_image,current_ball_position,new_ball_position,(255,0,0),4)
            
            current_ball_position = new_ball_position
            
            acceppt_points += 1
        
        cv2.arrowedLine(image_boast_trajectory_image,current_ball_position,(end_position_ball.x,end_position_ball.y),(255,0,0),4)
    
    @classmethod
    def _draw_three_points(cls,image_boast_trajectory_image,start_position_ball,end_position_ball,side):
        
        x_third_point = None
        
        if side.value == side.LEFT.value:
            x_third_point = random.randint(start_position_ball.x,end_position_ball.x)
        
        else:
            x_third_point = random.randint(end_position_ball.x,start_position_ball.x)
        
        y_third_point = random.randint(0,end_position_ball.y)
        
        cv2.arrowedLine(image_boast_trajectory_image,(start_position_ball.x,start_position_ball.y),(x_third_point,y_third_point),(255,0,0),4)
        cv2.arrowedLine(image_boast_trajectory_image,(x_third_point,y_third_point),(end_position_ball.x,end_position_ball.y),(255,0,0),4)
    
    @classmethod            
    def draw_curve(cls,count_of_points_curve,start_position_ball,end_position_ball,side,i,phase_dataset_path,WHITE_PIXEL = [255,255,255],
                   PROBABILITY_BALL_IS_HITTEN_BEFORE_FRONT_WALL_HIT = 0.5,SHOT = "boast"):
        
        image_boast_trajectory_image = WHITE_PIXEL[0] * numpy.ones((cls.HEIGHT,cls.WIDTH,3), numpy.uint8)
        is_ball_seen_before_front_wall_hit = PROBABILITY_BALL_IS_HITTEN_BEFORE_FRONT_WALL_HIT > random.uniform(0,1)
        
        if count_of_points_curve == 2:
            cv2.arrowedLine(image_boast_trajectory_image,(start_position_ball.x,start_position_ball.y),(end_position_ball.x,end_position_ball.y),(255,0,0),4)
            
        elif count_of_points_curve == 3:
            cls._draw_three_points(image_boast_trajectory_image,start_position_ball,end_position_ball,side)
        
        elif not is_ball_seen_before_front_wall_hit:
            cls._draw_ball_not_seen_before_front_wall_hit(image_boast_trajectory_image,start_position_ball,end_position_ball,count_of_points_curve,side)
        
        else:
            cls._draw_ball_seen_before_front_wall_hit(start_position_ball,end_position_ball,count_of_points_curve,side,image_boast_trajectory_image)
        
        cv2.circle(image_boast_trajectory_image,(start_position_ball.x,start_position_ball.y),8,(0,255,0),-1)
        cv2.circle(image_boast_trajectory_image,(end_position_ball.x,end_position_ball.y),8,(0,0,255),-1)
        
        boast_dataset_phase_path = os.path.join(phase_dataset_path,SHOT)
        
        os.makedirs(boast_dataset_phase_path,exist_ok=True)
        
        shot_path = os.path.join(boast_dataset_phase_path,f"{i}.png")
        
        cv2.imwrite(shot_path,image_boast_trajectory_image)
    
    @classmethod
    def create_synthetic_boast_path(cls,count_of_points_curve,i,root_dataset_path):
        
        middle_x = int(cls.WIDTH / 2)
        middle_y = int(cls.HEIGHT / 2)
        
        start_position_ball = cls.get_start_position(middle_y)
        side = cls.get_side(start_position_ball.x)
        end_position_ball = cls.get_end_position(middle_y,middle_x,side)
        
        cls.draw_curve(count_of_points_curve,start_position_ball,end_position_ball,side,i,root_dataset_path)

class SyntheticCross(SyntheticShot):
    
    def __init__(self) -> None:
        super().__init__()
    
    class Quarter(Enum):
        LeftUp = 0
        RightUp = 1
        RightDown = 2
        LeftDown = 3
    
    @classmethod
    def get_start_position(cls):
        third = random.choice([1,3])
        x_start = random.randint(int((third-1)*cls.WIDTH / 3),int(third*cls.WIDTH / 3))
        y_start = random.randint(0,cls.HEIGHT)
        
        return (x_start,y_start)
    
    @classmethod
    def end_start_position(cls,middle_x,middle_y,side):
        x_end = None
        y_end = None        
        
        if side.value == cls.Quarter.LeftUp.value or side.value == cls.Quarter.LeftDown.value:
            x_end = random.randint(int(2*cls.WIDTH/3),cls.WIDTH)
            y_end = random.randint(middle_y,cls.HEIGHT)
        
        elif side.value == cls.Quarter.RightUp.value or side.value == cls.Quarter.RightDown.value:
            x_end = random.randint(0,int(cls.WIDTH / 3))
            y_end = random.randint(middle_y,cls.HEIGHT)
        
        return (x_end,y_end)
    
    @classmethod
    def get_side(cls,start_ball_position,middle_x,middle_y):
        
        if start_ball_position[0] < middle_x and start_ball_position[1] < middle_y:
            return cls.Quarter.LeftUp
        
        elif start_ball_position[0] > middle_x and start_ball_position[1] < middle_y:
            return cls.Quarter.RightUp
        
        elif start_ball_position[0] > middle_x and start_ball_position[1] >= middle_y:
            return cls.Quarter.RightDown
        
        elif start_ball_position[0] < middle_x and start_ball_position[1] >= middle_y:
            return cls.Quarter.LeftDown
    
    @classmethod
    def _get_hit_fron_wall_point(cls,start_position_ball,end_position_ball):
        x_min = min(start_position_ball[0],end_position_ball[0])
        x_max = max(start_position_ball[0],end_position_ball[0])
        y_min = min(start_position_ball[1],end_position_ball[1]) if min(start_position_ball[1],end_position_ball[1]) < cls.HEIGHT / 2 else int(cls.HEIGHT / 2)
        
        x_middle = int((x_max + x_min) / 2)
        
        x_hit = random.randint(x_middle - 30, x_middle + 30)
        y_hit = random.randint(0,y_min)
             
        return (x_hit,y_hit)
    
    @classmethod
    def _draw_cross_half_up(cls,image_cross_trajectory_image,x_boundary,y_boundary,count_points_half,hit_front_wall_point):
        
        current_ball_position = (x_boundary,y_boundary)
        
        for _ in range(count_points_half):
            x = random.randint(current_ball_position[0],hit_front_wall_point[0])
            y = random.randint(hit_front_wall_point[1],current_ball_position[1])
            
            new_ball_position = (x,y)
            
            cv2.arrowedLine(image_cross_trajectory_image,current_ball_position,new_ball_position,(255,0,0),4)
            
            current_ball_position = new_ball_position
            
        cv2.arrowedLine(image_cross_trajectory_image,current_ball_position,hit_front_wall_point,(255,0,0),4)
    
    @classmethod
    def _draw_cross_half_down(cls,image_cross_trajectory_image,x_boundary,y_boundary,count_points_half,hit_front_wall_point):
        
        current_ball_position = (hit_front_wall_point[0],hit_front_wall_point[1])
        
        for _ in range(count_points_half):
            x = random.randint(current_ball_position[0],x_boundary)
            y = random.randint(current_ball_position[1],y_boundary)
            
            new_ball_position = (x,y)
            
            cv2.arrowedLine(image_cross_trajectory_image,current_ball_position,new_ball_position,(255,0,0),4)
            
            current_ball_position = new_ball_position
            
        cv2.arrowedLine(image_cross_trajectory_image,current_ball_position,(x_boundary,y_boundary),(255,0,0),4)
    
    @classmethod
    def _draw_more_than_two_points_cross_path(cls,image_cross_trajectory_image,side,count_of_points_curve,start_position_ball,end_position_ball,
                                              FRONT_PART_CROSS_Y_MIN = 270,FRONT_PART_CROSS_Y_MAX = 320):        
        
        hit_front_wall_point = cls._get_hit_fron_wall_point(start_position_ball,end_position_ball)
        count_points_half = int((count_of_points_curve - 2) / 2)
        
        x_min = min(start_position_ball[0],end_position_ball[0])
        first_boundary_point = start_position_ball if start_position_ball[0] == x_min else end_position_ball
        second_boundary_point = end_position_ball if start_position_ball[0] == x_min else start_position_ball
        
        cls._draw_cross_half_up(image_cross_trajectory_image,first_boundary_point[0],first_boundary_point[1],count_points_half,hit_front_wall_point)
        cls._draw_cross_half_down(image_cross_trajectory_image,second_boundary_point[0],second_boundary_point[1],count_points_half,hit_front_wall_point)
        
    
    @classmethod
    def _draw_cross_shot(cls,side,count_of_points_curve,start_ball_position,end_position_ball,i,phase_dataset_path,SHOT = "cross",WHITE_PIXEL = [255,255,255]):
        
        image_cross_trajectory_image = WHITE_PIXEL[0]*numpy.ones((cls.HEIGHT,cls.WIDTH,3),numpy.uint8)
        
        if count_of_points_curve == 2:
            cv2.arrowedLine(image_cross_trajectory_image,start_ball_position,end_position_ball,(255,0,0),4)
        
        elif count_of_points_curve > 2 :
            cls._draw_more_than_two_points_cross_path(image_cross_trajectory_image,side,count_of_points_curve,
                                                      start_ball_position,end_position_ball)
        
        cv2.circle(image_cross_trajectory_image,(start_ball_position[0],start_ball_position[1]),8,(0,255,0),-1)
        cv2.circle(image_cross_trajectory_image,(end_position_ball[0],end_position_ball[1]),8,(0,0,255),-1)
        
        cross_dataset_phase_path = os.path.join(phase_dataset_path,SHOT)
        
        os.makedirs(cross_dataset_phase_path,exist_ok=True)
        
        shot_path = os.path.join(cross_dataset_phase_path,f"{i}.png")
        
        cv2.imwrite(shot_path,image_cross_trajectory_image)
        
    @classmethod
    def create_synthetic_cross_path(cls,count_of_points_curve,i,phase_dataset_path):
        
        middle_x = int(cls.WIDTH / 2)
        middle_y = int(cls.HEIGHT / 2)
        
        start_ball_position = cls.get_start_position()
        
        side = cls.get_side(start_ball_position,middle_x,middle_y)
        
        end_ball_position = cls.end_start_position(middle_x,middle_y,side)
        
        cls._draw_cross_shot(side,count_of_points_curve,start_ball_position,end_ball_position,i,phase_dataset_path)

class SyntheticDrive(SyntheticShot):
    
    """
        generate randomly start position of the drive shot.
    """
    
    @classmethod
    def get_start_position(cls):
        x_start = random.randint(0,cls.WIDTH)
        y_start = random.randint(0,cls.HEIGHT)
        
        return (x_start,y_start)
    
    
    """
        get end position of the shot
        
        Arguments:
            start_ball_position (int,int): position where shot starts
        
        Returns:
            (int,int) pposition where shot ends
    """
        
    @classmethod
    def get_end_ball_position(cls,start_ball_position):
        
        x_end = random.randint(start_ball_position[0] - 90,start_ball_position[0] + 90)
        y_end = random.randint(int(cls.HEIGHT/2),cls.HEIGHT)
        
        return (x_end,y_end)
        
    """
        Draw trajectory path of the drive to the image
            
        Arguments:
           start_ball_position (int,int): begining of the drive
           end_ball_position (int,int): end of the drive
           count_of_points_curve (int): number of points in the shot
           i (int): index of the shot
    """
        
    @classmethod
    def draw_synthetic_drive_path(cls,start_ball_position,end_ball_position,count_of_points_curve,i,phase_dataset_path,SHOT = "drive"):
        x_min = min(start_ball_position[0],end_ball_position[0])
        x_max = max(start_ball_position[0],end_ball_position[0])
        
        current_ball_position = (start_ball_position[0],start_ball_position[1]) if x_min == start_ball_position[0] else (end_ball_position[0],end_ball_position[1])
        image_drive_trajectory_image = 255 * numpy.ones((cls.HEIGHT,cls.WIDTH,3), numpy.uint8)
        
        current_ball_position = (start_ball_position[0],start_ball_position[1])
        
        if count_of_points_curve == 2:
            cv2.arrowedLine(image_drive_trajectory_image,start_ball_position,end_ball_position,(255,0,0),4)
        
        else:
                                                        
            for point_number in range(count_of_points_curve):
                x = random.randint(x_min - 40,x_max + 40)
                y = random.randint(-270,0)
                
                if point_number > (count_of_points_curve - 2) / 2:
                    y = -y
                
                new_ball_position = (x,current_ball_position[1] + y)
                
                cv2.arrowedLine(image_drive_trajectory_image,current_ball_position,new_ball_position,(255,0,0),4)
                cv2.circle(image_drive_trajectory_image,(start_ball_position[0],start_ball_position[1]),8,(0,255,0),-1)
                cv2.circle(image_drive_trajectory_image,(end_ball_position[0],end_ball_position[1]),8,(0,0,255),-1)
                
                current_ball_position = new_ball_position
            
            cv2.arrowedLine(image_drive_trajectory_image,current_ball_position,end_ball_position,(255,0,0),4)
        
        drive_dataset_phase_path = os.path.join(phase_dataset_path,SHOT)
        
        os.makedirs(drive_dataset_phase_path,exist_ok=True)
        
        shot_path = os.path.join(drive_dataset_phase_path,f"{i}.png")
        
        cv2.imwrite(shot_path,image_drive_trajectory_image)
    
    
    """
    Generate synthetic drive
    
    Arguments:
        count_of_points_curve (int): number of points in the shot.
        i (int): index of the shot
    
    """
            
    @classmethod
    def create_synthetic_drive_path(cls,count_of_points_curve,i,phase_dataset_path):
        
        start_ball_position = cls.get_start_position()
        end_ball_position = cls.get_end_ball_position(start_ball_position)
        
        cls.draw_synthetic_drive_path(start_ball_position,end_ball_position,count_of_points_curve,i,phase_dataset_path)


# Create synthetic drop
class SyntheticDrop(SyntheticShot): 
        
    """
    generate randomly center of the drop shot.
    """
    
    @classmethod
    def get_center_position(cls):
        x = random.randint(0,cls.WIDTH)
        y = random.randint(int(cls.HEIGHT / 7),int(cls.HEIGHT / 3))
        
        return (x,y)
    
    """
    generate randomly angle of the drop shot.
    """

    @classmethod
    def get_start_end_angle(cls):
        full_angle = 2*math.pi
        
        start_angle = random.uniform(0,full_angle)  
        end_angle =  random.uniform(start_angle,full_angle)
        
        return start_angle,end_angle
    
    """
        Draw trajectory patt of the drop to the image
            
        Arguments:
           count_of_points_curve (int): number of points in the shot
           start_angle (int): start angle where drop starts to be drawn
           end_angle (int): angle where drops stoped to be generated
           x_center (int): horizontal position of the center of polar drop coordinates
           y_center (int): vertical position of the center of polar drop coordinates
           i (int): index of the shot
    """
        
    @classmethod
    def draw_drop_path(cls,count_of_points_curve,start_angle,end_angle,x_center,y_center,i,phase_dataset_path,SHOT = "drop"):
        
        image_drop_trajectory_image = 255 * numpy.ones((cls.HEIGHT,cls.WIDTH,3), numpy.uint8)
        max_radius = min(x_center % int(cls.WIDTH / 2), y_center % int(cls.HEIGHT / 2))
        
        current_angle = start_angle
        current_ball_position = None
        
        for idx_point in range(count_of_points_curve):
            radius = random.randint(0,max_radius)
            
            x = int(radius*math.cos(current_angle)) + x_center
            y = int(radius*math.sin(current_angle)) + y_center
            
            if current_ball_position is None:
                current_ball_position = (x,y)
                cv2.circle(image_drop_trajectory_image,(x,y),8,(0,255,0),-1)
            
            else:
                cv2.arrowedLine(image_drop_trajectory_image,current_ball_position,(x,y),(255,0,0),4)
                current_ball_position = (x,y)
            
            if idx_point == count_of_points_curve - 1:
                cv2.circle(image_drop_trajectory_image,(x,y),8,(0,0,255),-1)
            
            current_angle = random.uniform(current_angle,end_angle)
        
        drop_dataset_phase_path = os.path.join(phase_dataset_path,SHOT)
        
        os.makedirs(drop_dataset_phase_path,exist_ok=True)
        
        shot_path = os.path.join(drop_dataset_phase_path,f"{i}.png")
        
        cv2.imwrite(shot_path,image_drop_trajectory_image)
         
    """
        Creates synthetic drop
            
        Arguments:
           count_of_points_curve (int): number of points in the shot.
           i (int): index of the shot
    """
            
    @classmethod
    def create_synthetic_drop_path(cls,count_of_points_curve,i,phase_dataset_path):
        start_angle,end_angle = cls.get_start_end_angle()
        x_center,y_center = cls.get_center_position()
        
        cls.draw_drop_path(count_of_points_curve,start_angle,end_angle,x_center,y_center,i,phase_dataset_path)

"""
    Returns number of points in the synthetic shot.
        
    Arguments:
        count_of_points_curve (int): Previous number of points in previous shot. Variable represents how many times was ball detected.
    
    Returns:
        number of points in current shot (int)
"""

def update_count_of_points_curve(count_of_points_curve,MAX_COUNT_OF_POINTS_ON_CURVE = 8):
    return (count_of_points_curve + 1) % (MAX_COUNT_OF_POINTS_ON_CURVE + 1)

def main(BEGGINING_NUMBER_OF_SHOTS = 2):
    args = argument_parser.parse_args()
    number_of_synthetic_shot = int(args.number_of_synthetic_shots)
    phase_dataset_path = args.phase_folder
    
    count_of_points_curve = BEGGINING_NUMBER_OF_SHOTS
    for i in range(number_of_synthetic_shot):
        
        SynthehticBoast.create_synthetic_boast_path(count_of_points_curve,i,phase_dataset_path)
        SyntheticCross.create_synthetic_cross_path(count_of_points_curve,i,phase_dataset_path)
        SyntheticDrive.create_synthetic_drive_path(count_of_points_curve,i,phase_dataset_path)
        SyntheticDrop.create_synthetic_drop_path(count_of_points_curve,i,phase_dataset_path)
        
        count_of_points_curve = update_count_of_points_curve(count_of_points_curve)
        
        if count_of_points_curve == 0:
            count_of_points_curve = BEGGINING_NUMBER_OF_SHOTS

if __name__ == "__main__":
    main()
