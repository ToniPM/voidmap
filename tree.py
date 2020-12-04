from scipy.spatial import KDTree
from globalParams import *
import random
import numpy as np
from simplexTools import *
import networkx as nx
from PIL import ImageDraw
import anvil
from brush import getBrush
import pickle as rick
import os

NODE_ATTEMPTS_PER_SECTION = 5
CORE_HEIGHT = 230
DJIKSTRA_KRUSKAL_RATIO = 2
MAX_BRANCHING_FACTOR = 10
MAX_HALFNODE_DISTANCE = 10

ROOT_NEIGHBOR_AMOUNT = 5
MAX_SINGLE_BRANCH_SPAN = 16



class Tree:
    def __init__(self,parent_map,position,max_root_length):
        self.parent_map = parent_map
        self.center_x, self.center_z = position

        self.max_root_length = max_root_length
        self.max_radius = max_root_length

        self.root = (self.center_x,CORE_HEIGHT,self.center_z)
        self.nodes = [self.root]
        self.genNodes()

        self.genGraph()
        self.genTree()

        self.genBranches()

    def genNodesInChunk(self, chunk_x, chunk_z):

        nodes_in_chunk = set()

        top_nodes, top_height = zip(*self.parent_map.getLocalTopNodes(chunk_x, chunk_z))
        bot_nodes, bot_height = zip(*self.parent_map.getLocalBotNodes(chunk_x, chunk_z))

        top_tree = KDTree(top_nodes)
        bot_tree = KDTree(bot_nodes)

        def queryNode(x,y,z):
            return bot_height[bot_tree.query((x,z))[1]]<y<255-top_height[top_tree.query((x,z))[1]]

        coord_displacement_x = chunk_x * BLOCKS_PER_CHUNK
        coord_displacement_z = chunk_z * BLOCKS_PER_CHUNK

        for coord_displacement_y in range(0,256,16):
            for _ in range(coord_displacement_y*NODE_ATTEMPTS_PER_SECTION//128):
                x,y,z = random.randint(0,15),random.randint(0,15),random.randint(0,15)
                y += coord_displacement_y
                if queryNode(x,y,z):
                    nodes_in_chunk.add((coord_displacement_x+x,y,coord_displacement_z+z))
        self.nodes.extend(nodes_in_chunk)
    def genNodes(self):
        filename = "{mapname}_tree_{x},{z}_nodes".format(mapname = self.parent_map.name,x=self.center_x,z=self.center_z)
        if filename in os.listdir(os.getcwd()):
            timer = WaitTimer("Reading nodes for tree at {},{}".format(self.center_x, self.center_z))
            with open(filename,"rb") as f:
                self.nodes = rick.load(f)
            self.node_amount = len(self.nodes)
            timer.finish()
            return

        timer = WaitTimer("Generating nodes for tree at {},{}".format(self.center_x,self.center_z))
        for chunk_x,chunk_z in Box((self.center_x-self.max_radius)//BLOCKS_PER_CHUNK,
                                   (self.center_x+self.max_radius)//BLOCKS_PER_CHUNK+1,
                                   (self.center_z-self.max_radius)//BLOCKS_PER_CHUNK,
                                   (self.center_z+self.max_radius)//BLOCKS_PER_CHUNK+1,
                                   )():
            chunk_center_x,chunk_center_z = int((chunk_x+0.5)*BLOCKS_PER_CHUNK),int((chunk_z+0.5)*BLOCKS_PER_CHUNK)
            if np.linalg.norm((self.center_x-chunk_center_x,self.center_z-chunk_center_z))>self.max_radius:
                continue
            self.genNodesInChunk(chunk_x, chunk_z)
        self.node_amount = len(self.nodes)

        with open(filename,"wb") as f:
            rick.dump(self.nodes,f)

        timer.finish()
    def genGraph(self):
        filename = "{mapname}_tree_{x},{z}_edges".format(mapname = self.parent_map.name,x=self.center_x,z=self.center_z)
        if filename in os.listdir(os.getcwd()):
            timer = WaitTimer("Reading edges for tree at {},{}".format(self.center_x, self.center_z))
            with open(filename,"rb") as f:
                self.graph = rick.load(f)
            timer.finish()
            return

        timer = WaitTimer("Connecting nodes for tree at {},{}".format(self.center_x,self.center_z))

        self.graph = [[] for _ in range(self.node_amount)]
        self.kdtree = KDTree(self.nodes)

        bot_height_function = self.parent_map.getTopographIn("bot",self.center_x//BLOCKS_PER_REGION,self.center_z//BLOCKS_PER_REGION)
        top_height_function = self.parent_map.getTopographIn("top",self.center_x//BLOCKS_PER_REGION,self.center_z//BLOCKS_PER_REGION)

        def query_node(node):
            x,y,z = node
            return bot_height_function(x,z)<y<256-top_height_function(x,z)

        for first_node,second_node in self.kdtree.query_pairs(MAX_SINGLE_BRANCH_SPAN):
            first_point,second_point = np.array(self.nodes[first_node]),np.array(self.nodes[second_node])
            if query_node(0.5*(first_point+second_point)):
                distance = np.linalg.norm(first_point-second_point)
                self.graph[first_node].append((second_node,distance))
                self.graph[second_node].append((first_node,distance))

        root_neighbors = self.kdtree.query(self.root,ROOT_NEIGHBOR_AMOUNT+1)
        self.graph[0].extend([(root_neighbors[1][index],root_neighbors[0][index]) for index in range(1,ROOT_NEIGHBOR_AMOUNT+1)])

        with open(filename,"wb") as f:
            rick.dump(self.graph,f)

        timer.finish()
    def genTree(self):

        #parent_node podria ser un bool "explored"

        timer = WaitTimer("Picking branches for tree at {},{}".format(self.center_x,self.center_z))
        parent_node = [None]*self.node_amount
        distance = [None]*len(self.graph)
        self.child_nodes = [[] for _ in range(self.node_amount)]

        parent_node[0] = 0 #self.root
        distance[0] = 0

        #nodes a pqueue: ((antecessor,distància,node),prioritat)

        branch_queue = PriorityQueue([((0,weight,neighbor),DJIKSTRA_KRUSKAL_RATIO * weight)
                                      for neighbor,weight in self.graph[0]])
        while branch_queue:
            parent,edge_weight,current_node = branch_queue.pop()
            if parent_node[current_node] != None:
                continue #is in tree
            parent_node[current_node] = parent
            self.child_nodes[parent].append(current_node)
            current_distance = distance[parent]+edge_weight
            distance[current_node] = current_distance
            for neighbor,new_edge_weight in self.graph[current_node]:
                branch_queue.push((current_node,new_edge_weight,neighbor), current_distance + DJIKSTRA_KRUSKAL_RATIO * new_edge_weight)
        timer.finish()
    def genBranches(self):
        timer = WaitTimer("Generating branches for tree at {},{}".format(self.center_x,self.center_z))
        self.strahler = [-1]*self.node_amount
        self.branches = []
        root_strahler,root_branch = self.genBranches_recursive(0)
        if len(root_branch)>1:
            self.branches.append(Branch(self,root_branch,root_strahler))
        timer.finish()
    def genBranches_recursive(self,node):
        #readable > efficient
        child_info = [self.genBranches_recursive(child) for child in self.child_nodes[node]]

        if child_info:
            child_strahler = [strahler_num_child for strahler_num_child,branch_child in child_info]
            max_child_strahler = max(child_strahler)
            if child_strahler.count(max_child_strahler)>1:
                strahler_num = max_child_strahler + 1
            else:
                strahler_num = max_child_strahler

            branch = [self.nodes[node]]
            for strahler_num_child,branch_child in child_info:
                if strahler_num_child<strahler_num:
                    self.branches.append(Branch(self,[self.nodes[node]]+branch_child,strahler_num_child))
                else:
                    branch.extend(branch_child)
        else:
            strahler_num,branch = 1,[self.nodes[node]]

        self.strahler[node] = strahler_num
        return strahler_num,branch
    def drawOnMap(self,resolution,min_branch_width=1):
        draw = ImageDraw.Draw(self.parent_map.map)

        pixels_per_region = BLOCKS_PER_REGION // resolution
        coord_displacement = self.parent_map.region_radius * pixels_per_region

        for branch in self.branches:
            if branch.width >= min_branch_width:
                for node1,node2 in zip(branch.nodes,branch.nodes[1:]):
                    p1 = (node1[0]//resolution+coord_displacement,node1[2]//resolution+coord_displacement)
                    p2 = (node2[0]//resolution+coord_displacement,node2[2]//resolution+coord_displacement)
                    draw.line(p1+p2,fill=255,width=1)#branch.width+1-min_branch_width)
        del draw


    def genGraph_wnx(self):
        timer = WaitTimer("Connecting nodes for tree at {},{}".format(self.center_x,self.center_z))
        #NO COMPROVEM SI ELS NODES QUE AFEGIM NO FAN QUE LES BRANQUES ATRAVESSIN PARETS,
        #perque hauria de passar poc igualment i no es poc raonable que una branca atravessi el terra

        self.graph = nx.Graph()
        self.graph.add_nodes_from(((index,{"position":position}) for index,position in enumerate(self.nodes)))
        self.kdtree = KDTree(self.nodes)

        self.graph.add_weighted_edges_from(((first_node_index,
                                             second_node_index,
                                             np.linalg.norm(np.array(self.nodes[first_node_index])-np.array(self.nodes[second_node_index])))
                                            for first_node_index,second_node_index
                                            in self.kdtree.query_pairs(MAX_SINGLE_BRANCH_SPAN)))

        root_neighbors = self.kdtree.query(self.root,ROOT_NEIGHBOR_AMOUNT+1)
        self.graph.add_weighted_edges_from(((0,root_neighbors[1][index],root_neighbors[0][index]) for index in range(1,ROOT_NEIGHBOR_AMOUNT+1)))

        timer.finish()
    def genTree_wnx(self):

        timer = WaitTimer("Picking branches for tree at {},{}".format(self.center_x,self.center_z))
        parent_node = [None]*self.node_amount
        distance = [None]*len(self.graph)
        self.child_nodes = [[] for _ in range(self.node_amount)]

        parent_node[0] = 0 #self.root
        distance[0] = 0

        #nodes a pqueue: ((antecessor,distància,node),prioritat)

        branch_queue = PriorityQueue([((0,self.graph[0][neighbor]["weight"],neighbor),
                                       DJIKSTRA_KRUSKAL_RATIO * self.graph[0][neighbor]["weight"])
                                      for neighbor in self.graph[0]])
        while branch_queue:
            parent,edge_weight,current_node = branch_queue.pop()
            if parent_node[current_node] != None:
                continue #is in tree
            parent_node[current_node] = parent
            self.child_nodes[parent].append(current_node)
            current_distance = distance[parent]+edge_weight
            distance[current_node] = current_distance
            for neighbor in self.graph[current_node]:
                new_edge_weight = self.graph[current_node][neighbor]["weight"]
                branch_queue.push((current_node,new_edge_weight,neighbor), current_distance + DJIKSTRA_KRUSKAL_RATIO * new_edge_weight)
        timer.finish()
    def computeStrahler(self):
        timer = WaitTimer("Computing strahler numbers for tree at {},{}".format(self.center_x,self.center_z))
        self.strahler = [-1]*self.node_amount
        self.computeStrahler_recursive(0)
        timer.finish()
        print(self.strahler[0])
    def computeStrahler_recursive(self,node):
        child_strahlers = [self.computeStrahler_recursive(child) for child in self.child_nodes[node]]
        strahler_num = 0
        for child_strahler_num in child_strahlers:
            if strahler_num == child_strahler_num:
                strahler_num += 1
            strahler_num = max(strahler_num,child_strahler_num)
        self.strahler[node] = strahler_num
        return strahler_num
    def genNodesInChunk_old(self, chunk_x, chunk_z):
        """equiprobable segons altura"""

        top_nodes, top_height = zip(*self.parent_map.getLocalTopNodes(chunk_x, chunk_z))
        bot_nodes, bot_height = zip(*self.parent_map.getLocalBotNodes(chunk_x, chunk_z))

        top_tree = KDTree(top_nodes)
        bot_tree = KDTree(bot_nodes)

        def queryNode(x,y,z):
            return bot_height[bot_tree.query((x,z))[1]]<y<255-top_height[top_tree.query((x,z))[1]]

        coord_displacement_x = chunk_x * BLOCKS_PER_CHUNK
        coord_displacement_z = chunk_z * BLOCKS_PER_CHUNK
        for coord_displacement_y in range(0,256,16):
            for _ in range(NODE_ATTEMPTS_PER_SECTION):
                x,y,z = random.randint(0,15),random.randint(0,15),random.randint(0,15)
                y += coord_displacement_y
                if queryNode(x,y,z):
                    self.nodes.append((coord_displacement_x+x,y,coord_displacement_z+z))

class Branch:
    def __init__(self,parent_tree,node_list,width):
        self.parent_tree = parent_tree
        self.nodes = [np.array(node) for node in node_list]
        self.node_amount = len(self.nodes)
        self.width = width
        self.has_been_drawn = False

        self.computeHalfNodes()
        self.computeRegions()

    def computeHalfNodes(self):
        self.half_nodes = []
        self.half_nodes.append(2*self.nodes[0]-self.nodes[1])
        for i in range(1,self.node_amount):
            direction = self.nodes[i-1]-self.half_nodes[i-1]
            direction = direction/np.linalg.norm(direction)

            heading = self.nodes[i]-self.nodes[i-1]
            product = np.dot(direction,heading)
            if abs(product)>=0.01:
                modulus = 0.5*np.linalg.norm(heading)/product
                modulus = min(modulus,MAX_HALFNODE_DISTANCE)
            else:
                modulus = MAX_HALFNODE_DISTANCE
            self.half_nodes.append(self.nodes[i-1]+direction*modulus)
    def computeRegions(self):
        reverse_width_module = (-self.width)%BLOCKS_PER_REGION



        def getRegions(x,z):
            region_x,dx = divmod(int(x),BLOCKS_PER_REGION)
            region_z,dz = divmod(int(z),BLOCKS_PER_REGION)

            if dx <= self.width:
                if dz <= self.width:
                    return {(region_x, region_z),(region_x, region_z-1),(region_x-1, region_z),(region_x-1, region_z-1)}
                elif dz < reverse_width_module:
                    return {(region_x, region_z),(region_x-1, region_z)}
                else:
                    return {(region_x, region_z),(region_x, region_z+1),(region_x-1, region_z),(region_x-1, region_z+1)}
            elif dx < reverse_width_module:
                if dz <= self.width:
                    return {(region_x, region_z),(region_x, region_z-1)}
                elif dz < reverse_width_module:
                    return {(region_x, region_z)}
                else:
                    return {(region_x, region_z),(region_x, region_z+1)}
            else:
                if dz <= self.width:
                    return {(region_x, region_z),(region_x, region_z-1),(region_x+1, region_z),(region_x+1, region_z-1)}
                elif dz < reverse_width_module:
                    return {(region_x, region_z),(region_x+1, region_z)}
                else:
                    return {(region_x, region_z),(region_x, region_z+1),(region_x+1, region_z),(region_x+1, region_z+1)}

        self.regions = set().union(*[getRegions(x,z) for x,y,z in self.nodes+self.half_nodes])

        for region_x,region_z in self.regions:
            self.parent_tree.parent_map.branches_in_region[region_x][region_z].append(self)

    def make(self):

        TREE_MATERIAL = anvil.Block("minecraft", "white_concrete")

        if self.has_been_drawn:
            return
        self.has_been_drawn = True

        if self.width<=2:
            return

        if len(self.regions) == 1:
            region_x, region_z = next(iter(self.regions))

            # print("drawening branch as",list(zip(self.half_nodes,self.nodes)))
            # for node in self.nodes:
            #    print((region_x,region_z),node, getRegions(node[0],node[2]),"chunkcoords",node[0]//32,node[2]//32)

            region = self.parent_tree.parent_map.regions[region_x][region_z]
            def set_block(x,y,z):
                try:
                    region._unsafe_set_block(TREE_MATERIAL, x, y, z)
                except:
                    region._set_block(TREE_MATERIAL, x, y, z)
        else:
            def set_block(x,y,z):
                region_x,region_z = x//BLOCKS_PER_REGION,z//BLOCKS_PER_REGION
                region = self.parent_tree.parent_map.regions[region_x][region_z]
                try:
                    region._unsafe_set_block(TREE_MATERIAL, x, y, z)
                except:
                    region._set_block(TREE_MATERIAL, x, y, z)

        for node_1,halfnode,node_2 in zip(self.nodes,self.half_nodes[1:],self.nodes[1:]):
            step = 1/max(np.linalg.norm(node_1-halfnode),
                         np.linalg.norm(node_2-halfnode))


            points_to_draw = []
            k=1
            while k>=0:
                c_node = k*k*node_1 + 2*k*(1-k)*halfnode + (1-k)*(1-k)*node_2
                points_to_draw.append((int(c_node[0]), int(c_node[1]), int(c_node[2])))
                k -= step

            #draw first node
            for point in getBrush(self.width-1,(3*self.width,0,0)):
                set_block(*(node_1 + point))
            for point_1,point_2 in zip(points_to_draw,points_to_draw[1:]):
                diff = (point_2[0]-point_1[0],point_2[1]-point_1[1],point_2[2]-point_1[2])
                for point in getBrush(self.width-1,diff):
                    set_block(*(point_2+point))
