from opensimplex import OpenSimplex
from PIL import Image
import numpy as np
import random
from scipy.spatial import KDTree as KDTree
import heapq
from time import time

simplex=OpenSimplex()
noise_f=simplex.noise2d
noise_f(20,30)



def clamp2byte(num):
    return min(255,int((num+1)*128))
def clamp2byte2(num):
    return min(255,int(num*256))

def drawChunksOn(im):
    xw,yw=im.size
    vertical_bar = Image.new("L",(1,yw),0)
    horizontal_bar = Image.new("L",(xw,1),0)
    for x in range(0,xw,16):
        im.paste(vertical_bar,(x,0))
    for y in range(0,yw,16):
        im.paste(horizontal_bar,(0,y))
    return im

def lattice2line(x,y):
    t=x+y
    return (t+1)*t//2+y
            

class Box:
    def __init__(self,x_min,x_max,y_min,y_max):
        self.x_min = x_min
        self.x_max = x_max
        self.y_min = y_min
        self.y_max = y_max
        self.x_w = x_max-x_min
        self.y_w = y_max-y_min
        self.w = (self.x_w,self.y_w)
    def __call__(self):
        for x in range(self.x_min,self.x_max):
            for y in range(self.y_min,self.y_max):
                yield (x,y)

class Simplex:
    def __init__(self,noise_function,x_offset=0,y_offset=0,frequency=1):
        self.noise_function = noise_function
        self.x_offset = x_offset
        self.y_offset = y_offset
        self.frequency = frequency
    def __call__(self,x,y):
        return self.noise_function(self.x_offset + self.frequency*x,
                                   self.y_offset + self.frequency*y)

class MultiSimplex:
    def __init__(self,simplex_list,amplitude_function = lambda x : 1/x):
        self.simplex_list = simplex_list
        self.amplitude_list = [amplitude_function(simplex.frequency) for simplex in simplex_list]
        self.total_amplitude = sum(self.amplitude_list)
    def __call__(self,x,y):
        return sum(simplex(x,y)*amplitude for simplex,amplitude in zip(self.simplex_list,self.amplitude_list))/self.total_amplitude


def drawSimplex(noise_function,box):
    array = np.zeros(box.w)
    #im = Image.new("L", box.w, 0)
    #im.show()
    for x,y in box():
        #array[x,y]=clamp2byte(noise_function(x,y))
        array[x-box.x_min,y-box.y_min]=clamp2byte(noise_function(x,y))
    return Image.fromarray(array)


"""
drawChunksOn(drawSimplex(MultiSimplex([Simplex(noise_f,frequency=2**(-i)) for i in range(3,10)]),Box(0,1000,0,1000))).show()
"""

def getNodesSmall(x,y, seed=""):
    random.seed(lattice2line(x,y)+hash(seed))
    qt=random.randint(2,3)
    nodes = [(random.randint(0,7),random.randint(0,7)) for _ in range(qt)]
    return nodes

def getNodes(chunk_x,chunk_y, seed=""):
    #upper, lower; left, right
    UL_nodes = [(x+0,y+0) for x,y in getNodesSmall(2*chunk_x+0, 2*chunk_y+0, seed=seed)]
    UR_nodes = [(x+8,y+0) for x,y in getNodesSmall(2*chunk_x+1, 2*chunk_y+0, seed=seed)]
    LL_nodes = [(x+0,y+8) for x,y in getNodesSmall(2*chunk_x+0, 2*chunk_y+1, seed=seed)]
    LR_nodes = [(x+8,y+8) for x,y in getNodesSmall(2*chunk_x+1, 2*chunk_y+1, seed=seed)]
    return UL_nodes+UR_nodes+LL_nodes+LR_nodes

def sliceChunk(chunk_x,chunk_y,seed=""):
    nodes=[]
    for dx,dy in Box(-1,2,-1,2)():
        nodes+=[(x+dx*16,y+dy*16) for x,y in getNodes(chunk_x+dx, chunk_y+dy, seed=seed)]
    tree=KDTree(nodes)

    array = np.zeros((16,16), dtype=int)
    
    #print(dongus(2))
    
    for x,y in Box(0,16,0,16)():
        array[x,y] = tree.query((x,y))[1]

    return nodes,array

class Map:
    def __init__(self,height_function):
        self.height_function = height_function
    def makeChunk(self,chunk_x,chunk_y):
        nodes,array = sliceChunk(chunk_x,chunk_y)
        heights = [self.height_function(16*chunk_x+x,16*chunk_y+y) for x,y in nodes]
        for x,y in Box(0,16,0,16)():
            array[x,y] = heights[array[x,y]]
        return array
    def getImage(self,box):
        map_image = Image.new("L",(16*box.x_w,16*box.y_w),0)

        for chunk_x in range(0,box.x_w):
            for chunk_y in range(0,box.y_w):
                pass

        for chunk_x,chunk_y in box():
            print(chunk_x,chunk_y)
            chunk_image = Image.fromarray(np.transpose(self.makeChunk(chunk_x,chunk_y)))
            map_image.paste(chunk_image,((chunk_x-box.x_min)*16,
                                         (chunk_y-box.y_min)*16))
        return map_image

class PriorityQueue:
    def __init__(self,*args):
        self.queue = []
        if args:
            for item,priority in args[0]:
                self.push(item,priority)
    def push(self,item,priority):
        heapq.heappush(self.queue,(priority,item))
    def pop(self,with_priority = False):
        priority,item = heapq.heappop(self.queue)
        if with_priority:
            return item,priority
        else:
            return item
    def __bool__(self):
        return bool(self.queue)

class WaitTimer:
    def __init__(self,message):
        print(message,end=' ... ',flush=True)
        self.start_time = time()
    def finish(self):
        print("{:.2f}s".format(time()-self.start_time))
        del self




if __name__ == "__main__":

    a=PriorityQueue([(2,3),(4,4),(8,2),(-6,0)])
    print(a.pop())
    assert False

    base_simplex = Simplex(noise_f,x_offset=300,y_offset=7000,frequency=0.002)
    cull_simplex = Simplex(noise_f,x_offset=999,y_offset=1000,frequency=0.008)
    height_function = lambda x,y: np.sin(np.arcsin(base_simplex(x,y))*30)
    drawSimplex(height_function,Box(-16*32,2*16*32,-16*32,2*16*32)).show()

    assert False

    base_simplex = Simplex(noise_f,x_offset=300,y_offset=7000,frequency=0.002)
    cull_simplex = Simplex(noise_f,x_offset=999,y_offset=1000,frequency=0.008)
    #height_function = lambda x,y: clamp2byte(np.sin(np.arcsin(simplex(x,y))*30))
    height_function = lambda x,y: clamp2byte2((cull_simplex(x,y)+3)/4*(1+np.sin(np.arcsin(base_simplex(x,y))*30))/2)
    mapo = Map(height_function)

    Image.fromarray(mapo.makeChunk(12,13)).show()
    cool_picture = mapo.getImage(Box(-8,8,-8,8))
    cool_picture.show()
    drawChunksOn(cool_picture).show()