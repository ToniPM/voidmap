import functools
import numpy as np
from math import ceil,floor

@functools.lru_cache(maxsize=None)
def getBrush(width,displacement):
    return Brush(width,displacement)

class Brush:
    """w.i. displacement é x->y = y-x ón estem centrats a y"""
    def __init__(self, radius, displacement):

        radius -= 0.01 #quickfix pq sino farà allo de deixar els blocs solets apuntant als eixos de coordenades

        self.points = []
        other_center = -np.array(displacement)

        left_limit, right_limit = floor(-radius),ceil(radius+1)

        for x in range(left_limit,right_limit):
            for y in range(left_limit,right_limit):
                for z in range(left_limit,right_limit):
                    vector = np.array([x,y,z])
                    other_vector = vector-other_center
                    if np.linalg.norm(vector)<=radius and np.linalg.norm(other_vector)>radius:
                        self.points.append(vector)
    def __iter__(self):
        return iter(self.points)


if __name__=="__main__":
    for point in getBrush(1,(1,1,0)):
        print(point)