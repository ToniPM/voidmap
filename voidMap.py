import anvil
from simplexTools import *
import os
import numpy as np
import random
from scipy.spatial import KDTree
from globalParams import *
from tree import Tree

USER = "Xavi"
saves_dir = r"C:\Users\{}\AppData\Roaming\.minecraft\saves".format(USER)

EDGE_CURVE_FUNCTION = lambda x : x * x
TOP_CURVE_FUNCTION = lambda x : x*x*0.75+0.25
BOT_CURVE_FUNCTION = lambda x : np.square(np.sin(np.pi*x/2))

class VoidMap:

    def __init__(self, name, region_radius, base_simplex, min_h, max_h, cull_factor, top_cull_simplex, bot_cull_simplex,
                 top_cell_seed, bot_cell_seed, portal_locations, portal_radius, max_root_length):
        #name : nom del mapa a mc
        #medirà 1+2*(region_radius) regions de costat, éssent l'anell exterior regions achapades

        self.name = name
        self.path = os.path.join(saves_dir,name)
        self.region_dir = os.path.join(self.path,"region")

        self.initRegions(region_radius)

        self.base_simplex = base_simplex
        self.min_h = min_h
        self.max_h = max_h
        self.initBaseTopograph()

        self.cull_factor = cull_factor
        self.top_cull_simplex = top_cull_simplex
        self.bot_cull_simplex = bot_cull_simplex
        self.initCullConstants()

        self.chunk_diameter   = self.region_diameter*CHUNKS_PER_REGION
        self.neg_chunk_radius = -self.region_radius*CHUNKS_PER_REGION
        self.pos_chunk_radius = (self.region_radius+1)*CHUNKS_PER_REGION

        self.portal_locations = portal_locations
        self.portal_radius = portal_radius
        self.initPortals()

        self.makeCells("top",top_cell_seed)
        self.makeCells("bot",bot_cell_seed)

        self.max_root_length = max_root_length
        self.genPortals()

        self.drawMap(with_branches = True)
        self.map.show()

        self.makeMap()

    def initPortals(self):
        for portal_x,portal_z in self.portal_locations:
            if not self.isInBounds_block(portal_x,portal_z):
                raise Exception("Portal OOB")

        self.portals_in_range=[[[] for _ in range(self.region_diameter)]
                                   for _ in range(self.region_diameter)]
        for region_x,region_z,_ in self.enumerateRegions():
            # TODO
            #this could be more exhaustive
            #particularly
            #    check corners
            #    check parallel distance by distance along one axes * is portal in next region over along perpendicular axis

            region_center_x = int(BLOCKS_PER_REGION * (region_x + 0.5))
            region_center_z = int(BLOCKS_PER_REGION * (region_z + 0.5))

            for portal_x,portal_z in self.portal_locations:
                distance = np.linalg.norm((region_center_x-portal_x,region_center_z-portal_z))
                if distance <= self.portal_radius:
                    self.portals_in_range[region_x][region_z].append((portal_x,portal_z))
            if len(self.portals_in_range[region_x][region_z])>1:
                raise Exception("2 many portal")
    def genPortals(self):
        self.trees = []
        self.branches_in_region = [[[] for _ in range(self.region_diameter)]
                                       for _ in range(self.region_diameter)]
        for location in self.portal_locations:
            self.trees.append(Tree(self,location,self.max_root_length))

    def initBaseTopograph(self):
        FREQUENCY = 30
        self.base_topograph = lambda x,y : ((1 + np.sin(np.arcsin(self.base_simplex(x, y)) * FREQUENCY)) / 2)
    def initCullConstants(self):
        #a-1/a+1 = cull_factor
        #a-1 = ac + c
        #a(1-c) = c+1
        #a=1+c/1-c
        self.cull_top = (1+self.cull_factor)/(1-self.cull_factor)
        self.cull_bot = self.cull_top+1
    def initRegions(self, region_radius):
        self.region_radius = region_radius
        self.region_diameter = 2*region_radius+1
        self.regions = [[[] for _ in range(self.region_diameter)]
                            for _ in range(self.region_diameter)]
        for x,y in Box(-region_radius,region_radius+1,
                       -region_radius,region_radius+1)():
            self.regions[x][y] = anvil.EmptyRegion(x,y)

    def makeCells(self,side,seed):

        timer = WaitTimer("Generating {} cells".format(side))

        #canvi corresponent al hack de enumregions
        #cells = [[None
        cells = [[[]
                  for chunk_x in range(self.region_diameter * CHUNKS_PER_REGION)]
                  for chunk_z in range(self.region_diameter * CHUNKS_PER_REGION)]
        for region_x,region_z,_ in self.enumerateRegions():
            height_fun = self.getTopographIn(side,region_x,region_z)
            for chunk_x,chunk_z in Box(region_x * CHUNKS_PER_REGION, (region_x + 1) * CHUNKS_PER_REGION,
                                       region_z * CHUNKS_PER_REGION, (region_z + 1) * CHUNKS_PER_REGION)():
                current_cells = getNodes(chunk_x, chunk_z,seed)
                #cells[chunk_x][chunk_z] = (current_cells,[height_fun(chunk_x*BLOCKS_PER_CHUNK+cell_x,
                #                                                     chunk_z*BLOCKS_PER_CHUNK+cell_z)
                #                                          for cell_x,cell_z in current_cells])
                cells[chunk_x][chunk_z] = [((cell_x,cell_z),int(height_fun(chunk_x*BLOCKS_PER_CHUNK+cell_x,
                                                                           chunk_z*BLOCKS_PER_CHUNK+cell_z)))
                                           for cell_x, cell_z in current_cells]
        if side == "top":
            self.top_cells = cells
        elif side =="bot":
            self.bot_cells = cells
        else:
            raise Exception("el primer argument de makeCells ha d'ésser top o bot")
        timer.finish()
    def getCellSafeWrapper(self,side,chunk_x,chunk_z):
        if self.neg_chunk_radius<=chunk_x<self.pos_chunk_radius and self.neg_chunk_radius<=chunk_z<self.pos_chunk_radius:
            if side=="top":
                return self.top_cells[chunk_x][chunk_z]
            elif side=="bot":
                return self.bot_cells[chunk_x][chunk_z]
            else:
                raise Exception("el primer argument de getCellSafeWrapper ha d'ésser top o bot")
        return []
    def getLocalCellsSafe(self,side,chunk_x,chunk_z):
        #big unroll
        return  [((x-BLOCKS_PER_CHUNK,z-BLOCKS_PER_CHUNK),h) for (x,z),h in self.getCellSafeWrapper(side,chunk_x-1,chunk_z-1)]\
               +[((x-BLOCKS_PER_CHUNK,z                 ),h) for (x,z),h in self.getCellSafeWrapper(side,chunk_x-1,chunk_z  )]\
               +[((x-BLOCKS_PER_CHUNK,z+BLOCKS_PER_CHUNK),h) for (x,z),h in self.getCellSafeWrapper(side,chunk_x-1,chunk_z+1)]\
               +[((x                 ,z-BLOCKS_PER_CHUNK),h) for (x,z),h in self.getCellSafeWrapper(side,chunk_x  ,chunk_z-1)]\
               +[((x                 ,z                 ),h) for (x,z),h in self.getCellSafeWrapper(side,chunk_x  ,chunk_z  )]\
               +[((x                 ,z+BLOCKS_PER_CHUNK),h) for (x,z),h in self.getCellSafeWrapper(side,chunk_x  ,chunk_z+1)]\
               +[((x+BLOCKS_PER_CHUNK,z-BLOCKS_PER_CHUNK),h) for (x,z),h in self.getCellSafeWrapper(side,chunk_x+1,chunk_z-1)]\
               +[((x+BLOCKS_PER_CHUNK,z                 ),h) for (x,z),h in self.getCellSafeWrapper(side,chunk_x+1,chunk_z  )]\
               +[((x+BLOCKS_PER_CHUNK,z+BLOCKS_PER_CHUNK),h) for (x,z),h in self.getCellSafeWrapper(side,chunk_x+1,chunk_z+1)]
    def getLocalTopNodes(self,chunk_x,chunk_z):
        if self.neg_chunk_radius<chunk_x<self.pos_chunk_radius-1 and self.neg_chunk_radius<chunk_z<self.pos_chunk_radius-1:
            return  [((x-BLOCKS_PER_CHUNK,z-BLOCKS_PER_CHUNK),h) for (x,z),h in self.top_cells[chunk_x-1][chunk_z-1]]\
                   +[((x-BLOCKS_PER_CHUNK,z                 ),h) for (x,z),h in self.top_cells[chunk_x-1][chunk_z  ]]\
                   +[((x-BLOCKS_PER_CHUNK,z+BLOCKS_PER_CHUNK),h) for (x,z),h in self.top_cells[chunk_x-1][chunk_z+1]]\
                   +[((x                 ,z-BLOCKS_PER_CHUNK),h) for (x,z),h in self.top_cells[chunk_x  ][chunk_z-1]]\
                   +[((x                 ,z                 ),h) for (x,z),h in self.top_cells[chunk_x  ][chunk_z  ]]\
                   +[((x                 ,z+BLOCKS_PER_CHUNK),h) for (x,z),h in self.top_cells[chunk_x  ][chunk_z+1]]\
                   +[((x+BLOCKS_PER_CHUNK,z-BLOCKS_PER_CHUNK),h) for (x,z),h in self.top_cells[chunk_x+1][chunk_z-1]]\
                   +[((x+BLOCKS_PER_CHUNK,z                 ),h) for (x,z),h in self.top_cells[chunk_x+1][chunk_z  ]]\
                   +[((x+BLOCKS_PER_CHUNK,z+BLOCKS_PER_CHUNK),h) for (x,z),h in self.top_cells[chunk_x+1][chunk_z+1]]
        return self.getLocalCellsSafe("top",chunk_x,chunk_z)
    def getLocalBotNodes(self,chunk_x,chunk_z):
        if self.neg_chunk_radius<chunk_x<self.pos_chunk_radius-1 and self.neg_chunk_radius<chunk_z<self.pos_chunk_radius-1:
            return  [((x-BLOCKS_PER_CHUNK,z-BLOCKS_PER_CHUNK),h) for (x,z),h in self.bot_cells[chunk_x-1][chunk_z-1]]\
                   +[((x-BLOCKS_PER_CHUNK,z                 ),h) for (x,z),h in self.bot_cells[chunk_x-1][chunk_z  ]]\
                   +[((x-BLOCKS_PER_CHUNK,z+BLOCKS_PER_CHUNK),h) for (x,z),h in self.bot_cells[chunk_x-1][chunk_z+1]]\
                   +[((x                 ,z-BLOCKS_PER_CHUNK),h) for (x,z),h in self.bot_cells[chunk_x  ][chunk_z-1]]\
                   +[((x                 ,z                 ),h) for (x,z),h in self.bot_cells[chunk_x  ][chunk_z  ]]\
                   +[((x                 ,z+BLOCKS_PER_CHUNK),h) for (x,z),h in self.bot_cells[chunk_x  ][chunk_z+1]]\
                   +[((x+BLOCKS_PER_CHUNK,z-BLOCKS_PER_CHUNK),h) for (x,z),h in self.bot_cells[chunk_x+1][chunk_z-1]]\
                   +[((x+BLOCKS_PER_CHUNK,z                 ),h) for (x,z),h in self.bot_cells[chunk_x+1][chunk_z  ]]\
                   +[((x+BLOCKS_PER_CHUNK,z+BLOCKS_PER_CHUNK),h) for (x,z),h in self.bot_cells[chunk_x+1][chunk_z+1]]
        return self.getLocalCellsSafe("bot",chunk_x,chunk_z)

    def makeMap(self):
        self.clearMap()
        #self.clearEdge(2)
        for region_x, region_z,_ in self.enumerateRegions():
            #TODO : 2,2
            #(i 2,3?) <- no, es transposa. Aixó és (3,2) (i 3,3)
            if (region_x,region_z) not in [(2,2),(3,2)][1:]:
                continue
            self.makeTopograph(region_x,region_z)
            self.drawBranches(region_x,region_z)
            self.saveRegion(region_x,region_z)
            self.regions[region_x][region_z] = None
    def drawBranches(self,region_x,region_z):
            timer = WaitTimer("Drawing branches in region ({x},{z})".format(x=region_x,z=region_z))
            for branch in self.branches_in_region[region_x][region_z]:
                branch.make()
            timer.finish()
    def clearMap(self):
        timer = WaitTimer("Clearing map")
        for region_x,region_z,region in self.enumerateRegions():
            for chunk_x in range(region_x * CHUNKS_PER_REGION, (region_x + 1) * CHUNKS_PER_REGION):
                for chunk_z in range(region_z * CHUNKS_PER_REGION, (region_z + 1) * CHUNKS_PER_REGION):
                    new_chunk = anvil.EmptyChunk(chunk_x,chunk_z)
                    region.add_chunk(new_chunk)
        timer.finish()
    def makeTopograph(self,region_x,region_z):
        timer = WaitTimer("Carving region ({x},{z})".format(x=region_x,z=region_z))

        TOPOGRAPH_MATERIAL = anvil.Block("minecraft","gray_concrete")

        region = self.regions[region_x][region_z]

        for chunk_x,chunk_z in Box(region_x * CHUNKS_PER_REGION, (region_x + 1) * CHUNKS_PER_REGION,
                                   region_z * CHUNKS_PER_REGION, (region_z + 1) * CHUNKS_PER_REGION)():

            top_nodes, top_height = zip(*self.getLocalTopNodes(chunk_x,chunk_z))
            bot_nodes, bot_height = zip(*self.getLocalBotNodes(chunk_x,chunk_z))

            top_tree = KDTree(top_nodes)
            bot_tree = KDTree(bot_nodes)

            chunk = region.get_chunk(chunk_x,chunk_z)
            for x, z in Box(0, BLOCKS_PER_CHUNK, 0, BLOCKS_PER_CHUNK)():
                top_bot = top_height[top_tree.query((x,z))[1]]
                bot_top = bot_height[bot_tree.query((x,z))[1]]

                if bot_top+top_bot>=255:
                    try:
                        chunk._unsafe_fill_column(TOPOGRAPH_MATERIAL, x, z, 255, 0)
                    except:
                        chunk._fill_column(TOPOGRAPH_MATERIAL, x, z, 255, 0)
                else:
                    try:
                        chunk._unsafe_fill_column(TOPOGRAPH_MATERIAL, x, z, 255, 255-top_bot)
                        chunk._unsafe_fill_column(TOPOGRAPH_MATERIAL, x, z, bot_top, 0)
                    except:
                        chunk._fill_column(TOPOGRAPH_MATERIAL,x,z,255,255-top_bot)
                        chunk._fill_column(TOPOGRAPH_MATERIAL,x,z,bot_top,0)
        timer.finish()
    def clearEdge(self,radius):
        timer = WaitTimer("Clearing edge")

        #air_block = anvil.Block("minecraft","air")

        for region_x,region_z in Box(-self.region_radius-radius,self.region_radius+radius+1,
                                     -self.region_radius-radius,self.region_radius+radius+1)():
            if -self.region_radius<=region_x<=self.region_radius and -self.region_radius<=region_z<=self.region_radius:
                continue

            print("Deleting region ({},{})".format(region_x, region_z))
            empty_region = anvil.EmptyRegion(region_x,region_z)
            for chunk_x in range(region_x * CHUNKS_PER_REGION, (region_x + 1) * CHUNKS_PER_REGION):
                for chunk_z in range(region_z * CHUNKS_PER_REGION, (region_z + 1) * CHUNKS_PER_REGION):
                    empty_chunk = anvil.EmptyChunk(chunk_x,chunk_z)
                    empty_region.add_chunk(empty_chunk)
                    #empty_chunk.set_block(air_block, 8, 0, 8)

            empty_region.save(os.path.join(self.region_dir, "r.{x}.{z}.mca".format(x=region_x,z=region_z)))
        timer.finish()
    def saveRegion(self,region_x,region_z):
        timer = WaitTimer("Saving region ({x},{z})".format(x=region_x,z=region_z))
        region = self.regions[region_x][region_z]
        region.save(os.path.join(self.region_dir,"r.{x}.{z}.mca".format(x=region_x,
                                                                        z=region_z)),
                    verbose = False)
        timer.finish()
    def getTopographIn(self,side,region_x,region_z):
        if self.portals_in_range[region_x][region_z]:
            portal_x,portal_z = self.portals_in_range[region_x][region_z][0]
            if side=="top":
                def local_base_simplex(x,z):
                    portal_d = np.linalg.norm((x-portal_x,z-portal_z))
                    if portal_d<=self.portal_radius:
                        return TOP_CURVE_FUNCTION(portal_d/self.portal_radius) * (self.top_cull_simplex(x,z) + self.cull_top) / self.cull_bot
                    return (self.top_cull_simplex(x,z) + self.cull_top) / self.cull_bot
            elif side=="bot":
                def local_base_simplex(x,z):
                    portal_d = np.linalg.norm((x-portal_x,z-portal_z))
                    if portal_d<=self.portal_radius:
                        return BOT_CURVE_FUNCTION(portal_d/self.portal_radius) * (self.bot_cull_simplex(x,z) + self.cull_top) / self.cull_bot
                    return (self.bot_cull_simplex(x,z) + self.cull_top) / self.cull_bot
            else:
                raise Exception("el primer argument de getTopographIn ha d'ésser top o bot")
        else:
            if side=="top":
                local_base_simplex = lambda x,z : (self.top_cull_simplex(x,z) + self.cull_top) / self.cull_bot
            elif side=="bot":
                local_base_simplex = lambda x,z : (self.bot_cull_simplex(x,z) + self.cull_top) / self.cull_bot
            else:
                raise Exception("el primer argument de getTopographIn ha d'ésser top o bot")

        #helpful constants
        BIR  = BLOCKS_PER_REGION
        BIR2 = 2*BLOCKS_PER_REGION
        d_h = self.max_h - self.min_h

        #min(256,orig+k*192)

        #unrolled to avoid deeper stack
        if region_x==-self.region_radius:
            if region_z==-self.region_radius:
                return lambda x,z : min(256, self.min_h + d_h * self.base_topograph(x,z) * local_base_simplex(x,z) + 192 * EDGE_CURVE_FUNCTION(1 - x % BIR / BIR) + 192 * EDGE_CURVE_FUNCTION(1 - z % BIR / BIR))
            elif region_z<self.region_radius:
                return lambda x,z : min(256, self.min_h + d_h * self.base_topograph(x,z) * local_base_simplex(x,z) + 192 * EDGE_CURVE_FUNCTION(1 - x % BIR / BIR))
            elif region_z==self.region_radius:
                return lambda x,z : min(256, self.min_h + d_h * self.base_topograph(x,z) * local_base_simplex(x,z) + 192 * EDGE_CURVE_FUNCTION(1 - x % BIR / BIR) + 192 * EDGE_CURVE_FUNCTION(z % BIR / BIR))
            else:
                raise Exception("OOB a getTopographIn")
        elif region_x<self.region_radius:
            if region_z==-self.region_radius:
                return lambda x,z : min(256, self.min_h + d_h * self.base_topograph(x,z) * local_base_simplex(x,z) + 192 * EDGE_CURVE_FUNCTION(1 - z % BIR / BIR))
            elif region_z<self.region_radius:
                return lambda x,z : self.min_h + d_h * self.base_topograph(x,z) * local_base_simplex(x,z)
            elif region_z==self.region_radius:
                return lambda x,z : min(256, self.min_h + d_h * self.base_topograph(x,z) * local_base_simplex(x,z) + 192 * EDGE_CURVE_FUNCTION(z % BIR / BIR))
            else:
                raise Exception("OOB a getTopographIn")
        elif region_x==self.region_radius:
            if region_z==-self.region_radius:
                return lambda x,z : min(256, self.min_h + d_h * self.base_topograph(x,z) * local_base_simplex(x,z) + 192 * EDGE_CURVE_FUNCTION(x % BIR / BIR) + 192 * EDGE_CURVE_FUNCTION(1 - z % BIR / BIR))
            elif region_z<self.region_radius:
                return lambda x,z : min(256, self.min_h + d_h * self.base_topograph(x,z) * local_base_simplex(x,z) + 192 * EDGE_CURVE_FUNCTION(x % BIR / BIR))
            elif region_z==self.region_radius:
                return lambda x,z : min(256, self.min_h + d_h * self.base_topograph(x,z) * local_base_simplex(x,z) + 192 * EDGE_CURVE_FUNCTION(x % BIR / BIR) + 192 * EDGE_CURVE_FUNCTION(z % BIR / BIR))
            else:
                raise Exception("OOB a getTopographIn")
        else:
            raise Exception("OOB a getTopographIn")

        #aqui els trucos dels portals

    def drawMap(self, with_branches = False):

        timer = WaitTimer("Drawing map")

        TOPOGRAPH_WIDTH = 64#32
        RESOLUTION = 4

        PIXELS_PER_REGION = BLOCKS_PER_REGION // RESOLUTION
        total_width = self.region_diameter * PIXELS_PER_REGION
        coord_displacement = self.region_radius * PIXELS_PER_REGION

        map_array = np.zeros((lambda x: (x, x))(total_width), dtype=int)

        for region_x,region_z,_ in self.enumerateRegions():
            height_function_top = self.getTopographIn("top",region_x,region_z)
            height_function_bot = self.getTopographIn("bot",region_x,region_z)
            for x,z in Box(region_x*PIXELS_PER_REGION,(region_x+1)*PIXELS_PER_REGION,
                           region_z*PIXELS_PER_REGION,(region_z+1)*PIXELS_PER_REGION)():
                px,pz = x+coord_displacement,z+coord_displacement
                rx,rz = RESOLUTION*x,RESOLUTION*z

                top_h = height_function_top(rx,rz)
                bot_h = height_function_bot(rx,rz)

                space = 256-(top_h+bot_h)
                color = 0 if space<=0 else int(128+space/2.1)//TOPOGRAPH_WIDTH*TOPOGRAPH_WIDTH

                map_array[pz,px] = color

        self.map = Image.fromarray(map_array)

        for tree in self.trees:
            tree.drawOnMap(RESOLUTION,min_branch_width = 4)

        timer.finish()
    def enumerateRegions(self):
        for x,y in Box(-self.region_radius,self.region_radius+1,
                       -self.region_radius,self.region_radius+1)():
                yield x,y,self.regions[x][y]
    def isInBounds_block(self,x,z):
        return self.neg_chunk_radius*BLOCKS_PER_CHUNK<=x<=self.pos_chunk_radius*BLOCKS_PER_CHUNK\
               and self.neg_chunk_radius*BLOCKS_PER_CHUNK<=z<=self.pos_chunk_radius*BLOCKS_PER_CHUNK


