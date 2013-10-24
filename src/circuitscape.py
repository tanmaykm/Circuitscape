#!/usr/bin/python
##
## Circuitscape (C) 2013, Brad McRae and Viral B. Shah. 
##

import time, gc
import numpy as np
from numpy import *
from scipy import sparse
from scipy.sparse.csgraph import connected_components

from cs_util import print_timing, deletecol, deleterowcol, relabel
from cs_io import CSIO
from cs_raster import CSRaster

wx_available = True
try:
    import wx
except ImportError:
    wx_available = False
    wx = None


class circuitscape(CSRaster):
    def __init__(self, configFile, logger_func):
        super(circuitscape, self).__init__(configFile, logger_func)

    @print_timing
    def compute(self):
        """Main function for Circuitscape."""  
        # Code below provides a back door to network mode, because not incorporated into GUI yet
        if self.options.polygon_file == 'NETWORK': 
            self.options.data_type='network' #can also be set in .ini file
        if self.options.data_type=='network': 
            self.options.graph_file = self.options.habitat_file
            self.options.focal_node_file = self.options.point_file

        self.state.startTime = time.time()
        self.state.last_gui_yield_time = time.time()        
        self.log('',1)
        self.log('',2)

        #Test write privileges by writing config file to output directory
        self.options.write(self.options.output_file, True)
        
        if self.options.data_type=='network':
            result, solver_failed = self.compute_network() # Call module for solving arbitrary graphs (not raster grids)
            self.logCompleteJob()
            return result, solver_failed #Fixme: add in solver failed check

        return self.compute_raster()

    @print_timing
    def compute_network(self): 
        """Solves arbitrary graphs instead of raster grids."""  
        solver_failed = False
        (g_graph, node_names) = self.read_graph(self.options.graph_file)
        
        (num_components, C) = connected_components(g_graph)
        C += 1 # Number components from 1
        
        self.log('Graph has %d nodes and %d components'%(g_graph.shape[0], num_components), 2)
        
        if num_components > 1:
            full_graph = g_graph
        else:
            nodes_in_component = node_names
        
        full_resistances = [] # initialize
        full_node_currents = []
        for component in range(1, num_components+1):
            if num_components > 1:
                indices = node_names[np.where(C == component)]
                del_indices = np.where(C != component)
                indices = np.where(C == component)
                nodes_in_component = node_names[indices]                
                g_graph = deleterowcol(full_graph, delrow = del_indices, delcol = del_indices)
                
            G = self.laplacian(g_graph)
            del g_graph
            if self.options.scenario == 'advanced':
                raise RuntimeError('Advanced mode is not yet implemented in graph/network mode.')
                #(sources,grounds)= self.readSourcesGroundsNetwork(G, node_names, self.options.source_file,self.options.ground_file)
                #FIXME: retool readSourceGroundNodes to read both files and return complete sources and grounds
                #result,solver_failed = self.advanced_module_network(G,sources,grounds,node_names)           
                
            else:
                if self.options.use_included_pairs==True:
                    self.state.included_pairs = CSIO.read_included_pairs(self.options.included_pairs_file)
                focalNodes = self.readFocalNodes(self.options.focal_node_file)
                if num_components > 1:    #Prune out any focal nodes that are not in component
                    focalNodesInComponent = focalNodes
                    includeList = list(nodes_in_component[:])
                    numFocalNodes = focalNodesInComponent.shape[0]
                    row = 0
                    while row <numFocalNodes: 
                        if focalNodesInComponent[row] in includeList: #match
                            row = row+1
                        else:
                            n = focalNodesInComponent.shape[0]
                            keep = delete (arange(0, n), row)
                            focalNodesInComponent = focalNodesInComponent[keep]
                            numFocalNodes = numFocalNodes-1
                else:
                    focalNodesInComponent = focalNodes
                    numFocalNodes = focalNodes.shape[0]
                
                if self.options.scenario=='pairwise':
                    if numFocalNodes > 1:
                        # module returns arrays with node names
                        cumBranchCurrents,cumNodeCurrents,resistances3columns,solver_failed = self.pairwise_module_network(G,focalNodesInComponent,nodes_in_component,num_components)
                    else: #nothing to solve
                        cumNodeCurrents = zeros((len(nodes_in_component),2),dtype = 'float64')
                        cumNodeCurrents[:,0] = nodes_in_component
                        resistances3columns = []
                        cumBranchCurrents = []
                    
                    # if first connected component solved, 
                    # then create output arrays
                    if (full_resistances == []) and (resistances3columns != []):
                        full_resistances = resistances3columns
                        if self.options.write_cur_maps == True:
                            fullBranchCurrents = cumBranchCurrents 
                            full_node_currents = cumNodeCurrents
                    
                    # if ongoing solve, append results to ongoing output arrays
                    elif resistances3columns != []: 
                        full_resistances = append(full_resistances , resistances3columns, axis=0)
                        if self.options.write_cur_maps == True:
                            fullBranchCurrents = append(fullBranchCurrents , cumBranchCurrents, axis=0)
                            full_node_currents = append(full_node_currents , cumNodeCurrents, axis=0)
                    else: # If only node in component, just modify node current array
                        if self.options.write_cur_maps == True:
                            if full_node_currents == []: #if first time populated
                                full_node_currents = cumNodeCurrents
                            full_node_currents = append(full_node_currents , cumNodeCurrents, axis=0)
                    
                else:
                    raise RuntimeError('One-to-all and all-to-one modes are not yet implemented in graph/network mode.')
                    #result,solver_failed = self.one_to_all_module_network(G,focalNodes,node_names)  

        if self.options.write_cur_maps == True:
            ind = lexsort((fullBranchCurrents[:, 1], fullBranchCurrents[:, 0]))
            fullBranchCurrents = fullBranchCurrents[ind]                        
                
            ind = lexsort((full_node_currents[:, 1], full_node_currents[:, 0]))
            full_node_currents = full_node_currents[ind]                        

            fileadd='cum'
            CSIO.write_currents(self.options.output_file, fullBranchCurrents, full_node_currents, fileadd)

        #Make lists of focal node pairs.  Use to add disconnected pairs
        #to resistance output.
        resistancePairList = list()
        for row in range(0,full_resistances.shape[0]): 
            listEntry = str(int(full_resistances[row,0])) + "_" + str(int(full_resistances[row,1]))
            resistancePairList.append(listEntry)
            # print listEntry
        # print resistancePairList
        for sourceNode in range(0,len(focalNodes)-1):
            for targetNode in range(sourceNode+1,len(focalNodes)):                     
                pair = str(int(focalNodes[sourceNode])) + "_" + str(int(focalNodes[targetNode]))   
                # print pair
                if pair not in resistancePairList:
                    # Add in disconnected pair
                    newPair = array([[focalNodes[sourceNode],focalNodes[targetNode],-1]])
                    # print 'new',newPair
                    full_resistances = append(full_resistances , newPair, axis=0)
            
        ind = lexsort((full_resistances[:, 1], full_resistances[:, 0]))
        full_resistances = full_resistances[ind] 
        CSIO.write_resistances_3columns(self.options.output_file, full_resistances)
        
        return full_resistances,solver_failed #Fixme: need to check solver failed.

          
    def pairwise_module_network(self, G, focal_nodes, nodeNames, numComponents):
        """Overhead module to solves arbitrary graphs in pairwise mode.
        
        Returns branch currents in 3-col format plus 3-column voltages,
        resistances, 2-col node currents.

        Writes currents voltages and for each pair.

        """
        self.log('Graph has %d nodes and %d components'%(focal_nodes.size, numComponents), 2)
        
        solver_failed = False
        if self.options.use_included_pairs==True: #Prune points
            focal_nodes = self.prune_included_pairs_network(focal_nodes)          
            included_pairs = self.state.included_pairs 
        else:
            included_pairs = np.ones((focal_nodes.size+1, focal_nodes.size+1), dtype='int32')

        numpoints = focal_nodes.size
        resistances = -1*np.ones((focal_nodes.size, focal_nodes.size), dtype='float64')
        if self.options.write_cur_maps == True:
            cum_node_currents = np.zeros((G.shape[0],1),dtype = 'float64')
            cum_branch_currents = sparse.csr_matrix((G.shape))
        if (self.options.write_cur_maps == True) or (self.options.write_volt_maps == True) or (self.options.use_included_pairs==True):        
            use_resistance_calc_shortcut = False
        else:
            use_resistance_calc_shortcut = True
            # This uses a quicker way to calculate pairwise resistances.  There may be something even quicker, I'm looking into this. BHM 3-15-2013
            print 'Using shortcut'
            shortcut_resistances = -1 * np.ones((numpoints, numpoints), dtype='float64') 
            voltmatrix = np.zeros((numpoints, numpoints), dtype='float64')
                 
        dst_point = 0
        anchor_point = 0            
        
        x = 0
        for i in range(0, numpoints):
            if range(i, numpoints) == []:
                break
            
            if (use_resistance_calc_shortcut==True) and (dst_point>0):
                break #No need to continue, we've got what we need to calculate resistances
            
            dst_point = dst_point+1            
            dst = self.name_to_node(nodeNames, focal_nodes[i])
            G_dst_dst = G[dst, dst] 
            G[dst,dst] = 0
            self.state.amg_hierarchy = None
            gc.collect()
            self.create_amg_hierarchy(G)         
            
            for j in range(i+1, numpoints):
                x = x+1
                if use_resistance_calc_shortcut==True:
                    y = numpoints
                    self.log('solving focal node %d of %d'%(x,y), 1)
                else:
                    y = numpoints*(numpoints-1)/2
                    self.log ('solving focal pair %d of %d'%(x,y), 1)   
                             
                if included_pairs[i+1,j+1] != 1:
                    continue
                
                src = self.name_to_node(nodeNames, focal_nodes[j])
                try:
                    voltages = self.single_ground_solver(G, src, dst)
                    resistances[i, j] = voltages[src] - voltages[dst]
                    resistances[j, i] = voltages[src] - voltages[dst]
                except:
                    solver_failed = True
                    resistances[i, j] = -777
                    resistances[j, i] = -777

                if (use_resistance_calc_shortcut==True) and (dst_point==1) and (solver_failed==False): #this occurs for first i that is in component
                    anchor_point = i #for use later in shortcut resistance calc
                    voltmatrix = self.getVoltmatrixNetwork(i, j, numpoints, voltages, resistances, voltmatrix, focal_nodes, nodeNames) 

                if (self.options.write_cur_maps == True) and (solver_failed==False):
                    finitegrounds = [-9999]
                    (node_currents, branch_currents) = self.getCurrentsNetwork(G, voltages, finitegrounds)
                    
                    # Append node names and convert to array format
                    branch_currents_array = self.convert_graph_to_3_col(branch_currents,nodeNames)
                    node_currents_array = self.append_names_to_node_currents(node_currents, nodeNames)
                    
                    if self.options.write_cum_cur_map_only==False:                
                        fileadd = str(int(focal_nodes[i])) + '_' + str(int(focal_nodes[j]))
                        CSIO.write_currents(self.options.output_file, branch_currents_array, node_currents_array, fileadd)

                    cum_node_currents = cum_node_currents + node_currents
                    cum_branch_currents = cum_branch_currents + branch_currents                       

                if (self.options.write_volt_maps == True) and (solver_failed==False):
                    fileadd = str(int(focal_nodes[i])) + '_' + str(int(focal_nodes[j]))
                    CSIO.write_voltages(self.options.output_file, voltages, nodeNames, fileadd)

                if solver_failed==True:
                    solver_failed = False
                    #solver_failed_somewhere = True                 
            if (use_resistance_calc_shortcut==True) and (i==anchor_point): #this happens once per component. Anchorpoint is the first i in component
                shortcut_resistances = self.get_shortcut_resistances(anchor_point,voltmatrix,numpoints,resistances,shortcut_resistances)
             
            G[dst,dst] = G_dst_dst
             
        if self.options.write_cur_maps == True:
            cum_branch_currents_array = self.convert_graph_to_3_col(cum_branch_currents,nodeNames)
            cum_node_currents_array =  self.append_names_to_node_currents(cum_node_currents, nodeNames)
        else:
            cum_branch_currents_array = -1
            cum_node_currents_array = -1

        if use_resistance_calc_shortcut==True:
            resistances = shortcut_resistances
        for i in range(0,numpoints):
            resistances[i, i] = 0

        output_resistances = self.append_names_to_resistances(focal_nodes, resistances)       
        resistances3columns = self.convertResistances3cols(output_resistances) 
               
        return cum_branch_currents_array, cum_node_currents_array, resistances3columns, solver_failed  


        
    def append_names_to_node_currents(self, node_currents, nodeNames):
        """Adds names of focal nodes to node current lists."""    
        outputNodeCurrents=zeros((len(node_currents),2),dtype='float64')
        outputNodeCurrents[:,0]=nodeNames[:]
        try:
            outputNodeCurrents[:,1]=node_currents[:,0]
        except:
            outputNodeCurrents[:,1]=node_currents[:]
        return outputNodeCurrents        


    def getCurrentsNetwork(self, G, voltages, finitegrounds):
        """Returns node and branch currents given voltages in arbitrary graphs."""
        G =  G.tocoo()
        node_currents = self.get_node_currents(voltages, G, finitegrounds)
        node_currents_col = np.zeros((node_currents.shape[0],1), dtype='float64')
        node_currents_col[:,0] = node_currents[:]
        branch_currents = self.get_branch_currents(G, voltages, True) 
        branch_currents = np.absolute(branch_currents) 
        G = G.tocsr()
        return node_currents_col, branch_currents
    
    
    def getVoltmatrixNetwork(self, i, j, numpoints, voltages, resistances, voltmatrix, focalNodes, nodeNames):                                            
        """Returns a matrix of pairwise voltage differences between focal nodes when operating on arbitrary graphs.
        
        Used for shortcut calculations of effective resistance when no
        voltages or currents are mapped.
        
        """  
        voltvector = np.zeros((numpoints, 1), dtype='float64')  
        for point in range(1, numpoints):
            node = self.name_to_node(nodeNames, focalNodes[point])
            voltageAtPoint = voltages[node] 
            voltageAtPoint = 1-(voltageAtPoint/resistances[i, j])
            voltvector[point] = voltageAtPoint
        voltmatrix[:,j] = voltvector[:,0] 
        return voltmatrix             
             
             
    def name_to_node(self, node_names, name):
        """Returns node index given node ID."""  
        node_names = node_names.tolist()
        node = node_names.index(name)
        return node


    def namesToNodes(self, nodeNames, names):
        """Returns node indices given node IDs."""  
        nodeNames = nodeNames.tolist()
        nodes = zeros(len(names),dtype = 'int32')

        for i in range (0,len(names)):
            nodes[i] = nodeNames.index(names[i])
        return nodes

    
    def read_graph(self, filename):
        """Reads arbitrary graph from disk. Returns sparse adjacency matrix and node names ."""
        #print("read_graph: %s" % (filename,))  
        graphList = CSIO.load_graph(filename)

        try:
            zeros_in_resistance_graph = False           
            nodes = deletecol(graphList,2) 
            nodeNames = unique(asarray(nodes))
            nodes[where(nodes>= 0)] = relabel(nodes[where(nodes>= 0)], 0)
            node1 = nodes[:,0]
            node2 = nodes[:,1]
            data = graphList[:,2]
            
            ######################## Reclassification code
            if self.options.use_reclass_table == True:
                try:
                    reclassTable = CSIO.read_point_strengths(self.options.reclass_file)    
                except:
                    raise RuntimeError('Error reading reclass table.  Please check file format.')
                for i in range (0,reclassTable.shape[0]):
                    data = where(data==reclassTable[i,0],reclassTable[i,1],data)
                print'\n***** Reclassified habitat graph using', self.options.reclass_file,'*****'
            ########################
            
            if self.options.habitat_map_is_resistances == True:
                zeros_in_resistance_graph = (where(data==0, 1, 0)).sum() > 0
                conductances = 1/data
            else:
                conductances = data
                
            numnodes = nodeNames.shape[0]
            G = sparse.csr_matrix((conductances, (node1, node2)), shape = (numnodes, numnodes))

            Gdense=G.todense()
            g_graph = maximum(Gdense, Gdense.T) # To handle single or double entries for elements BHM 06/28/11
            g_graph = sparse.csr_matrix(g_graph)
        except:
            raise RuntimeError('Error processing graph/network file.  Please check file format')
        
        if zeros_in_resistance_graph == True:
            raise RuntimeError('Error: zero resistance values are not currently allowed in habitat network/graph input file.')
        #print "g_graph="
        #print g_graph
        #print "nodeNames="
        #print nodeNames
        return g_graph, nodeNames


    def readFocalNodes(self, filename):
        """Loads list of focal nodes for arbitrary graph."""  
        focalNodes = CSIO.load_graph(filename)
        try:    
            if filename==self.options.graph_file:#If graph was used as focal node file, then just want first two columns for focalNodes.
                focalNodes = deletecol(focalNodes, 2)
            focalNodes = unique(asarray(focalNodes))
        except:
            raise RuntimeError('Error processing focal node file.  Please check file format')
        return focalNodes
     
    def convert_graph_to_3_col(self,graph,nodeNames): 
        """Converts a sparse adjacency matrix to 3-column format."""  
        Gcoo =  graph.tocoo()
        mask = Gcoo.data > 0
        
        graphNcol = zeros((Gcoo.row[mask].size,3),dtype = "float64") #Fixme: this may result in zero-current nodes being left out.  Needed to make change so dimensions would match Gcoo.data[mask]
        
        if nodeNames==None:

            graphNcol[:,0] = Gcoo.row[mask]
            graphNcol[:,1] = Gcoo.col[mask]
        else:
            graphNcol[:,0]=nodeNames[Gcoo.row[mask]]
            graphNcol[:,1]=nodeNames[Gcoo.col[mask]]
        graphNcol[:,2] = Gcoo.data[mask]
        return graphNcol


    def prune_included_pairs_network(self, focalNodes):
        """Remove excluded points from focal node list when using extra file that lists pairs to include/exclude in network mode."""   
        includedPairs = (self.state.included_pairs)
        includeList = list(includedPairs[0,:])
        point = 0
        _drop_flag = False
        while point < focalNodes.size: #Prune out any points not in includeList
            if focalNodes[point] in includeList: #match
                point += 1
            else:
                _drop_flag = True   
                focalNodes = delete(focalNodes, point)
         
        includeList = list(focalNodes[:])
        numConnectionRows = includedPairs.shape[0]
        row = 1
        while row <numConnectionRows: #Prune out any entries in includeList that are not in focalNodes
            if includedPairs [row,0] in includeList: #match
                row = row+1
            else:
                includedPairs = deleterowcol(includedPairs, delrow = row, delcol = row)   
                _drop_flag = True
                numConnectionRows = numConnectionRows-1

        self.state.included_pairs = includedPairs                     
#         if _drop_flag==True:
#             print'\nNOTE: Code to exclude pairwise calculations is activated and \nsome entries did not match with focal node file.  \nSome focal nodes may have been dropped.'      
        return focalNodes

