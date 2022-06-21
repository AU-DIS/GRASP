import numpy as np
import scipy as sci
import networkx as nx
import time
import scipy.sparse as sps
import lapjv
import os
import fast_pagerank
import contextlib
from sklearn.preprocessing import normalize
import argparse
from scipy.sparse import csr_matrix, coo_matrix
from sklearn.neighbors import KDTree
import math
import matplotlib.pyplot as plt
#import base_align as ba
#import munkres
import base_align_pymanopt as ba

#import base_align as ba

from sklearn.neighbors import NearestNeighbors
#np.set_printoptions(precision=3)
np.set_printoptions(formatter={'float': lambda x: "{0:0.3f}".format(x)})




# 	#noise_level= [1]
# 	scores=np.zeros([5,reps])
# 	for i in noise_level:
# 		for j in range(1,reps+1):
#
#
# 			print('Noise level %d, round %d' %(i,j))


def parse_args():
    parser=argparse.ArgumentParser(description= "RUN GASP")
    parser.add_argument('--graph',nargs='?', default='arenas')
    # 1:nn 2:sortgreedy 3: jv
    parser.add_argument('--laa', type=int, default= 1, help='Linear assignment algorithm. 1=nn,2=sortgreedy, 3=jv')
    parser.add_argument('--icp', type=bool, default=False)
    parser.add_argument('--icp_its', type=int, default=0, help= 'how many iterations of iterative closest point')
    parser.add_argument('--q', type=int, default=20)
    parser.add_argument('--k', type= int, default=5)
    parser.add_argument('--lower_t', type=float,default=0.01, help='smallest timestep for corresponding functions')
    parser.add_argument('--upper_t', type=float,default=1.0, help='biggest timestep for corresponding functions')
    parser.add_argument('--linsteps', type=float, default=True, help='scaling of time steps of corresponding functions, logarithmically or linearly')
    parser.add_argument('--reps', type=int,default=5, help='number of repetitions per noise level')
    parser.add_argument('--noise_levels', type=list,default=[10])
    return parser.parse_args()


def main(args):
    for i in range(1,38):
        print(i)
        graph_name=args.graph
        #edge_list_G1 = '/home/au640507/spectral-graph-alignment/graphs/karate/karate.txt'

        #
        # load edges from graphs
        #edge_list_G2 = '/home/au640507/spectral-graph-alignment/graphs/karate/karate_var3.txt'
        #edge_list_G1 = 'permutations/' + graph_name + '/' + graph_name + '_orig.txt'
        #edge_list_G2 = 'permutations/' + graph_name + '/noise_level_1/edges_1.txt'
        edge_list_G1 = '../PycharmProjects/Test/prepare_graphs/transp/transp_all_'+str(i)+'.txt'
        edge_list_G2 = '../PycharmProjects/Test/prepare_graphs/transp/transp_'+str(i)+'.txt'

        #
        #gt_file = '/home/au640507/spectral-graph-alignment/graphs/karate/karate_gt.txt'
        #gt_file = 'permutations/' + graph_name + '/noise_level_1/gt_1.txt'
        #gt_file= 'permutations/' + graph_name + '/' + graph_name + '_identical_gt.txt'
        gt_file = '../PycharmProjects/Test/prepare_graphs/transp/gt.txt'
        #
        data = np.loadtxt(gt_file, delimiter=" ")
        #
        data= data.astype(int)

        # values = np.linspace(0, 1945, 1946).astype(int)
        # gt = np.c_[values, values]
        gt = dict(data)
        A1 = edgelist_to_adjmatrix(edge_list_G1)

        A2 = edgelist_to_adjmatrix(edge_list_G2)

        #matching,timw=functional_maps_base_align(A1,A2,args.q, args.k,args.laa,args.icp, args.icp_its, args.lower_t, args.upper_t, args.linsteps,args.graph,1,1)



        matching,a,b,c=functional_maps(A1,A2,args.q, args.k,args.laa,args.icp, args.icp_its, args.lower_t, args.upper_t, args.linsteps,args.graph,1,1)
        #print(matching)
        matching = matching#dict(matching.astype(int))


        acc = eval_matching(matching, gt)
        print(acc)
        print('\n')


def align_voting_heuristic(A1, A2,q,k_list,laa,icp, icp_its, lower_t, upper_t, linsteps,graph_name,i,j, corr_func, voting, ba_):
    
    matchings = [0]*len(k_list)
    # voting - 1:greedy, 2:jv
    # corr_func - 1:heat kern  2:pagerank 3: mix
    # 1:nn 2:sortgreedy 3: jv
    match = laa

    t = np.linspace(lower_t, upper_t, q)
    if (not linsteps):
        t = np.logspace(math.log(lower_t,10), math.log(upper_t,10), q)


    n = np.shape(A1)[0]


    D1, V1 = decompose_laplacian(A1)

    # # # #
    D2, V2 = decompose_laplacian(A2)

    start = time.time()
    if corr_func==2:
        Cor1 = calc_pagerank_corr_func(q, A1)
        Cor2 = calc_pagerank_corr_func(q, A2)
    elif corr_func==3:
        Cor1 = calc_personalized_pagerank_corr_func(q, A1)
        Cor2 = calc_personalized_pagerank_corr_func(q, A2)
    elif corr_func==1:
        Cor1 = calc_corresponding_functions(n, q, t, D1, V1)
        #print(Cor1)
        Cor2 = calc_corresponding_functions(n, q, t, D2, V2)
    

    
    print('Base Align')
    if ba_:
        B = ba.optimize_AB(Cor1, Cor2, n, V1, V2, D1, D2, k_list[-1]) #base alignment, manifold optimization
    print('Voting')
    for i in (range(len(k_list))):
        print(i)
        #print('Base Align')
        if ba_:
            B_ = B[:k_list[i], :k_list[i]]
            V1_rot=V1[:,0:k_list[i]] # pick k eigenvectors
            V2_rot = V2[:, 0:k_list[i]] @ B_ # pick k eigenvectors and rotate via B
        else:
            V1_rot = V1
            V2_rot = V2

        
        #print('Calculate C')
        C = calc_C_as_in_quasiharmonicpaper(Cor1, Cor2,V1_rot,V2_rot,k_list[i],q)
        



        G1_emb =  V1_rot.T#[:, 0: k].T;

        G2_emb =C @ V2_rot.T#[:, 0: k].T;
        matching = []

        if (icp):
            matching = iterative_closest_point(V1_rot, V2_rot, C, icp_its, k_list[i], match,Cor1, Cor2,q)
        else:
            if match == 1:
                matching = greedyNN(G1_emb, G2_emb)
            if match == 2:
                matching = sort_greedy(G1_emb, G2_emb)
            if match == 3:
                matching = hungarian_matching(G1_emb, G2_emb)
            if match == 4:
                matching = top_x_matching(G1_emb, G2_emb,10)
            if match == 5:
                matching = nearest_neighbor_matching(G1_emb, G2_emb)
            if match == 6:
                matching = kd_align(G1_emb, G2_emb)

        end=time.time()
    #  np.savetxt('/home/au640507/spectral-graph-alignment/permutations_no_subset/arenas/noise_level_1/matching_' + str(
    #     i) + '.txt', matching, fmt="%d")
        if not icp: 
            matching = dict(matching.astype(int))
        #matching=matching.astype(int)
        matchings[i] = matching

    match_freq = np.zeros((n,n))
    for i in range(len(matchings)):
        for j in range(n):
            m = matchings[i][j]
            match_freq[j][m] += 1
    if voting==2: # Hungarian
        cols, rows, _ = lapjv.lapjv(-match_freq + np.amax(match_freq))

        matching = np.c_[np.linspace(0, n-1, n).astype(int),cols]
        matching=matching[matching[:,0].argsort()]
        real_matching = dict(matching.astype(int))
    if voting==3: #not injective
        
        maxes = np.argmax(match_freq, axis= 1)
        matching = np.c_[np.linspace(0, n-1, n).astype(int),maxes]
        real_matching = dict(matching.astype(int))
    if voting==1:
        real_matching = sort_greedy_voting(match_freq)
    end=time.time()
    return real_matching, (end-start), 0,0





def functional_maps_base_align(A1, A2,q,k,laa,icp, icp_its, lower_t, upper_t, linsteps,graph_name,i,j, corr_func= 1, lower_q= 0):
    # corr_func - 1:heat kern  2:pagerank 3: mix
    # 1:nn 2:sortgreedy 3: jv
    match = laa

    t = np.linspace(lower_t, upper_t, q)
    if (not linsteps):
        t = np.logspace(math.log(lower_t,10), math.log(upper_t,10), q)


    n = np.shape(A1)[0]


    start = time.time()
    D1, V1 = decompose_laplacian(A1)

    # # # #
    D2, V2 = decompose_laplacian(A2)

    if corr_func==2:
        Cor1 = calc_pagerank_corr_func(q, A1)
        Cor2 = calc_pagerank_corr_func(q, A2)
    elif corr_func==1:
        Cor1 = calc_corresponding_functions(n, q, t, D1, V1)
        #print(Cor1)
        Cor2 = calc_corresponding_functions(n, q, t, D2, V2)
    elif corr_func==3:
        Cor1_heat = calc_corresponding_functions(n, q, t, D1, V1)
        #print(Cor1)
        Cor2_heat = calc_corresponding_functions(n, q, t, D2, V2)
        Cor1_pr = calc_pagerank_corr_func(q, A1)
        Cor2_pr = calc_pagerank_corr_func(q, A2)
        Cor1_heat = Cor1_heat / Cor1_heat.sum()
        Cor2_heat = Cor2_heat / Cor2_heat.sum() 
        Cor1 = np.hstack((Cor1_heat, Cor1_pr))
        Cor2 = np.hstack((Cor2_heat, Cor2_pr))
        q *= 2
    print(np.abs(Cor2-Cor1))

    print('Base Align')
    B = ba.optimize_AB(Cor1, Cor2, n, V1, V2, D1, D2, k) #base alignment, manifold optimization
    

    V1_rot=V1[:,0:k] # pick k eigenvectors
    V2_rot = V2[:, 0:k] @ B # pick k eigenvectors and rotate via B
    print('Calculate C')
    C = calc_C_as_in_quasiharmonicpaper(Cor1,Cor2,V1_rot,V2_rot,k,q)
    print(np.diagonal(C))



    G1_emb =  V1_rot.T#[:, 0: k].T;

    G2_emb =C @ V2_rot.T#[:, 0: k].T;
    matching = []

    if (icp):
        matching = iterative_closest_point(V1_rot, V2_rot, C, icp_its, k, match,Cor1,Cor2,q)
    else:
        if match == 1:
            matching = greedyNN(G1_emb, G2_emb)
        if match == 2:
            matching = sort_greedy(G1_emb, G2_emb)
        if match == 3:
            matching = hungarian_matching(G1_emb, G2_emb)
        if match == 4:
            matching = top_x_matching(G1_emb, G2_emb,10)
        if match == 5:
            matching = nearest_neighbor_matching(G1_emb, G2_emb)

    end=time.time()
  #  np.savetxt('/home/au640507/spectral-graph-alignment/permutations_no_subset/arenas/noise_level_1/matching_' + str(
   #     i) + '.txt', matching, fmt="%d")
    if not icp: 
        matching = dict(matching.astype(int))
    #matching=matching.astype(int)
    return matching,(end-start),V1_rot,V2_rot




def functional_maps(A1, A2,q,k,laa,icp, icp_its, lower_t, upper_t, linsteps,graph_name,i,j,corr_func=1, lower_q=0): 

    #1:nn 2:sortgreedy 3: jv
    match=laa

    t = np.linspace(lower_t, upper_t, q) #create q evenly distributed timesteps, linearly
    if(not linsteps):
        t=np.logspace(lower_t, upper_t, q) # create q evenly distributed timesteps, log scale

    n = np.shape(A1)[0]
    print(n)


    D1, V1 = decompose_laplacian(A1)
   # print(V1[:,0])
  #   D1, V1 = decompose_laplacian(A1)
  # # #  print(V1[20,200])
    D2, V2 = decompose_laplacian(A2)
  #   print(V2[20, 200])


    if corr_func==2:
        Cor1 = calc_pagerank_corr_func(q, A1, lower_q)
        Cor2 = calc_pagerank_corr_func(q, A2, lower_q)
    else:
        Cor1 = calc_corresponding_functions(n, q, t, D1, V1)
        #print(Cor1)
        Cor2 = calc_corresponding_functions(n, q, t, D2, V2)


#    print(Cor1[20,25])
#    print(Cor2[20, 25])


    # rotV1, rotV2=calc_rotation_matrices(Cor1, Cor2, V1, V2,k)
    # #
    # plt.imshow(rotV1[0:25,0:25])
    # plt.show()
    # V1=V1[:,0:k]@rotV1
    # V2 = V2[:,0:k] @ rotV2

    # A = calc_coefficient_matrix(Cor1, V1, k, q)

    # B = calc_coefficient_matrix(Cor2, V2, k, q)
#    print(A[20,25])
#    print(B[20,25])
    #C=calc_correspondence_matrix(A,B,k)

    #C=calc_correspondence_matrix_ortho(A,B,k)
    C = calc_C_as_in_quasiharmonicpaper(Cor1, Cor2, V1[:,0:k], V2[:,0:k], k, q)
    # plt.imshow(C)
    # plt.show()
    # print(np.diagonal(C))

    G1_emb = C @ V1[:, 0: k].T;

    G2_emb = V2[:, 0: k].T;
    matching=[]

    if(icp):
        matching = iterative_closest_point(V1, V2, C, icp_its, k,match,Cor1,Cor2,q)
    else:
        if match==1:
            matching = greedyNN(G1_emb, G2_emb)
        if match==2:
            matching=sort_greedy(G1_emb, G2_emb)
        if match==3:
            matching = hungarian_matching(G1_emb, G2_emb)
        if match == 4:
            matching = top_x_matching(G1_emb, G2_emb,10)
        if match == 5:
            matching = nearest_neighbor_matching(G1_emb, G2_emb)

    if not icp: 
        matching = dict(matching.astype(int))

    return matching,0,V1,V2

def calc_pagerank_corr_func(no_of_alphas, A):
    pagerank_G = np.zeros((len(A), no_of_alphas))
    #if lower_q==0: lower_q = 0.001
    #alphas = np.flip(1-np.logspace(np.log10(0), np.log(0.8), no_of_alphas))  # experiment with different distribution
    alphas = np.linspace(0.6,1, no_of_alphas)
    if no_of_alphas == 1:
        alphas[0] = 0.8
    print("alphas:", alphas)
    for i in range(len(alphas)):
        pagerankG_vector = fast_pagerank.pagerank(A,p=alphas[i])
        pagerank_G[:, i] = pagerankG_vector
    return pagerank_G



def calc_personalized_pagerank_corr_func(dim, A):
    pagerank_G = np.zeros((len(A), dim+1))
    
    pr_orig = fast_pagerank.pagerank(A,p= 0.80)
    
    signature_vector = pr_orig

    nodes_sorted = signature_vector.argsort()[::-1]

    avg = len(nodes_sorted) / float(dim)
    out = []
    last = 0.0

    while last < len(nodes_sorted):
        out.append(nodes_sorted[int(last):int(last + avg)])
        last += avg
    
    pagerank_G[:,0] = pr_orig
    for i in range(len(out)):
        #dic = { j : 1 for j in out[i]}
        pers = np.zeros(len(A))
        pers[out[i]] = 1
        
        pagerankG_vector = fast_pagerank.pagerank(A,p=0.80,personalize= pers)
        
        pagerank_G[:, i] = pagerankG_vector
    
    return pagerank_G    



def edgelist_to_adjmatrix(edgeList_file):

    edge_list = np.loadtxt(edgeList_file, usecols=range(2))

    n = int(np.amax(edge_list)+1)
    #n = int(np.amax(edge_list))
   # print(n)

    e = np.shape(edge_list)[0]


    a = np.zeros((n, n))

    # make adjacency matrix A1

    for i in range(0, e):
        n1 = int(edge_list[i, 0])#- 1

        n2 = int(edge_list[i, 1])#- 1

        a[n1, n2] = 1.0
        a[n2, n1] = 1.0

    return a


def decompose_laplacian(A):

    #  adjacency matrix

    Deg = np.diag((np.sum(A, axis=1)))


    n = np.shape(Deg)[0]

    Deg=sci.linalg.fractional_matrix_power(Deg, -0.5) #D^-1/2


    L  = np.identity(n) - Deg @ A @ Deg
   # print((sci.fractional_matrix_power(Deg, -0.5) * A * sci.fractional_matrix_power(Deg, -0.5)))
    # '[V1, D1] = eig(L1);

    D, V = np.linalg.eigh(L) # return eigenvalue vector, eigenvector matrix of L

    return [D, V]

def decompose_unnormalized_laplacian(A):

    #  adjacency matrix

    Deg = np.diag((np.sum(A, axis=1)))

    n = np.shape(Deg)[0]

    L = Deg- A

   # print((sci.fractional_matrix_power(Deg, -0.5) * A * sci.fractional_matrix_power(Deg, -0.5)))
    # '[V1, D1] = eig(L1);

    D, V = np.linalg.eig(L)

    return [D, V]

def decompose_rw_normalized_laplacian(A):

    #  adjacency matrix

    Deg = np.diag((np.sum(A, axis=1)))

    n = np.shape(Deg)[0]

    L = np.identity(n) - np.linalg.inv(Deg) @ A

   # print((sci.fractional_matrix_power(Deg, -0.5) * A * sci.fractional_matrix_power(Deg, -0.5)))
    # '[V1, D1] = eig(L1);

    D, V = np.linalg.eig(L)

    return [D, V]

def decompose_rw_laplacian(A):

    #  adjacency matrix

    Deg = np.diag((np.sum(A, axis=1)))

    n = np.shape(Deg)[0]

    L = np.linalg.inv(Deg) @ A

   # print((sci.fractional_matrix_power(Deg, -0.5) * A * sci.fractional_matrix_power(Deg, -0.5)))
    # '[V1, D1] = eig(L1);

    D, V = np.linalg.eig(L)

    return [D, V]






def calc_corresponding_functions(n, q, t, d, V):

    # corresponding functions are the heat kernel diagonals in each time step
    # t= time steps, d= eigenvalues, V= eigenvectors, n= number of nodes, q= number of corresponding functions
    t = t[:, np.newaxis] #newxis increas dimension of array by 1
    d = d[:, np.newaxis]

    V_square = np.square(V)

    time_and_eigv = np.dot((d), np.transpose(t))

    time_and_eigv = np.exp(-1*time_and_eigv)

    Cores=np.dot(V_square, time_and_eigv)

    return Cores


def calc_coefficient_matrix(Corr, V, k, q):
    coefficient_matrix = np.linalg.lstsq(V[:,0:k],Corr,rcond=None)
    #print(type(coefficient_matrix))
    return coefficient_matrix[0]

def calc_correspondence_matrix(A, B, k):
    C = np.zeros([k,k])
    At = A.T
    Bt = B.T

    for i in range(0,k):
        C[i, i] = np.linalg.lstsq(Bt[:,i].reshape(-1,1), At[:,i].reshape(-1,1),rcond=None)[0]

    return C

def calc_correspondence_matrix_ortho_diag(A, B, k):
    C = np.zeros([k,k])
    At = A.T
    Bt = B.T

    for i in range(0,k):
        C[i, i] = np.sign(np.linalg.lstsq(Bt[:,i].reshape(-1,1), At[:,i].reshape(-1,1),rcond=None)[0])

    return C

def calc_correspondence_matrix_ortho(A, B, k):
    #C = np.zeros([k,k])
    At = A.T
    Bt = B.T

    C=sci.linalg.orthogonal_procrustes(Bt,At)[0]

    C_norms=np.linalg.norm(C)

    C_normalized=normalize(C,axis=1)
    #for i in range(0,k):
    #print(np.shape(C))
    #print(C)
    #print('\n')
    #print(C_normalized)

    #print(np.sum(C_normalized,axis=1))
    #print(np.sum(C, axis=1))
    #return C_normalized
    return C_normalized


def nearest_neighbor_matching(G1_emb, G2_emb):
    n= np.shape(G1_emb)[1]
    nbrs = NearestNeighbors(n_neighbors=1, algorithm='ball_tree').fit(G1_emb.T)
    distances, indices = nbrs.kneighbors(G2_emb.T)
    # if not len(indices) == len(np.unique(indices)):
    #     print('Allignment not bijective')
    indices=np.c_[np.linspace(0, n-1, n).astype(int), indices.astype(int)]

    return indices

def hungarian_matching(G1_emb, G2_emb):
    #print('hungarian_matching: calculating distance matrix')
    dist = sci.spatial.distance_matrix(G1_emb.T, G2_emb.T)
    n = np.shape(dist)[0]
    # print(np.shape(dist))
    #print('hungarian_matching: calculating matching')

    cols, rows, _ = lapjv.lapjv(dist)
    
    #print('fertig')
    matching = np.c_[cols, np.linspace(0, n-1, n).astype(int)]
    matching=matching[matching[:,0].argsort()]
    return matching.astype(int)

def iterative_closest_point(V1, V2, C, it,k,match,Cor1,Cor2,q):
    G1= V1[:, 0: k].T
    G2_emb = V2[:, 0: k].T
    n = np.shape(G2_emb)[1]

    for i in range(0,it):

        #print('icp iteration '+str(i))
        G1_emb=C@V1[:,0:k].T
       # print('calculating hungarian in icp')
        M=[]

        if (match == 1):
            M = nearest_neighbor_matching(G1_emb, G2_emb)
        if match == 2:
            M = sort_greedy(G1_emb, G2_emb)
        if match == 3:
            M = hungarian_matching(G1_emb, G2_emb)
        if match == 4:
            M = top_x_matching(G1_emb, G2_emb,10)
        if match == 5:
            M = greedyNN(G1_emb, G2_emb)
        if match == 6:
            M = kd_align(G1_emb.T, G2_emb.T)
        G2_cur=np.zeros([k,n])
       ## print('finding nearest neighbors in eigenvector matrix icp')
        for j in range(0,n):

            G2idx = M[j, 1]
            G2_cur[:, G2idx]=G2_emb[:, j]
       ## print('calculating correspondence matrix in icp')
        #C=calc_correspondence_matrix(G1,G2_cur,k)
        C = calc_correspondence_matrix_ortho(G1, G2_cur, k)
        #calc_C_as_in_quasiharmonicpaper(Cor1, Cor2, V1[:,0:k], V2[:,0:k], k, q)
        #C = calc_correspondence_matrix_ortho_diag(G1, G2_cur, k)
        C_show = C
        # C_show[C_show < 0.13] = 0.0
       # plt.imshow(np.abs(C_show))
       # plt.show()




       # print('calculated correspondence matrix in icp')
       ## print('\n')
    G1_emb = C@V1[:,0:k].T

    if (match == 1):
        M = nearest_neighbor_matching(G1_emb, G2_emb)
    if match == 2:
        M = sort_greedy(G1_emb, G2_emb)
    if match == 3:
        M = hungarian_matching(G1_emb, G2_emb)
    if match == 4:
        M = top_x_matching(G1_emb, G2_emb,10)
    if match == 5:
        M = greedyNN(G1_emb, G2_emb)
    if match == 6:
        M = kd_align(G1_emb.T, G2_emb.T)

    return dict(M.astype(int))

def greedyNN(G1_emb, G2_emb):
    #print('greedyNN: calculating distance matrix')

    dist = sci.spatial.distance_matrix(G1_emb.T, G2_emb.T)
    n = np.shape(dist)[0]
    # print(np.shape(dist))
    #print('greedyNN: calculating matching')
    idx = np.argsort(dist, axis=0)
    matching=np.ones([n,1])*(n+1)
    for i in range(0,n):
        matched=False
        cur_idx=0
        while(not matched):
           #print([cur_idx,i])
           if(not idx[cur_idx,i] in matching):
               matching[i,0]=idx[cur_idx,i]

               matched=True
           else:
               cur_idx += 1
               #print(cur_idx)

    matching = np.c_[np.linspace(0, n-1, n).astype(int),matching]
    return matching.astype(int)

def sort_greedy(G1_emb, G2_emb):
    #print('sortGreedy: calculating distance matrix')
    start = time.time()
    dist = sci.spatial.distance_matrix(G1_emb.T, G2_emb.T)
    med1=time.time()
    print(med1-start)
    n = np.shape(dist)[0]
    # print(np.shape(dist))
    #print('sortGreedy: calculating matching')
    dist_platt=np.ndarray.flatten(dist)
    med2=time.time()
    print(med2-med1)
    idx = np.argsort(dist_platt)#
    med3=time.time()
    print(med3-med2)
    k=idx//n
    r=idx%n
    idx_matr=np.c_[k,r]
   # print(idx_matr)
    G1_elements=set()
    G2_elements=set()
    i=0
    j=0
    matching=np.ones([n,2])*(n+1)
    med4=time.time()
    print(med4-med3)
    while(len(G1_elements)<n):
        if (not idx_matr[i,0] in G1_elements) and (not idx_matr[i,1] in G2_elements):
            #print(idx_matr[i,:])
            matching[j,:]=idx_matr[i,:]

            G1_elements.add(idx_matr[i,0])
            G2_elements.add(idx_matr[i,1])
            j+=1
            #print(len(G1_elements))


        i+=1
    med5=time.time()
    print(med5-med4)
   # print(idx)
    matching = np.c_[matching[:,1], matching[:,0]]
    matching = matching[matching[:, 0].argsort()]
    return matching.astype(int)

def top_x_matching(G1_emb, G2_emb,x):
    dist = sci.spatial.distance_matrix(G1_emb.T, G2_emb.T)
    n = np.shape(dist)[0]

    idx = np.argsort(dist, axis=0)

    matches=idx[0:x,:]


    matching = np.c_[np.linspace(0, n-1, n).astype(int), matches.T]
    #matching = matching[matching[:, 0].argsort()]
    return matching.astype(int)

def kd_align(emb1, emb2, normalize=False, distance_metric="euclidean", num_top=10):
    kd_tree = KDTree(emb2, metric=distance_metric)
    if num_top > emb1.shape[0]:
        num_top = emb1.shape[0]
    row = np.array([])
    col = np.array([])
    data = np.array([])
    
    dist, ind = kd_tree.query(emb1, k=num_top)
    print("queried alignments")
    row = np.array([])
    for i in range(emb1.shape[0]):
        row = np.concatenate((row, np.ones(num_top) * i))
    col = ind.flatten()
    data = np.exp(-dist).flatten()
    sparse_align_matrix = coo_matrix((data, (row, col)), shape=(emb1.shape[0], emb2.shape[0]))
    # mat = sparse_align_matrix.tocsr().toarray()
    alignment_matrix = sparse_align_matrix.tocsr()
    n_nodes = alignment_matrix.shape[0]
    nodes_aligned = []
    counterpart_dict = {}

    if not sps.issparse(alignment_matrix):
        sorted_indices = np.argsort(alignment_matrix)

    for node_index in range(n_nodes):
        if sps.issparse(alignment_matrix):
            row, possible_alignments, possible_values = sps.find(alignment_matrix[node_index])
            node_sorted_indices = possible_alignments[possible_values.argsort()]
        else:
            node_sorted_indices = sorted_indices[node_index]
        
        for i in range(num_top):
            possible_node = (node_sorted_indices[-(i+1)])
            if not possible_node in nodes_aligned:
                counterpart = possible_node
                counterpart_dict[node_index] = counterpart
                nodes_aligned.append(counterpart)
                found_node = True
                break  
        if not found_node:
            for possible_node in range(n_nodes):
                if not possible_node in nodes_aligned:
                    counterpart = possible_node
                    counterpart_dict[node_index] = counterpart
                    nodes_aligned.append(counterpart)
                    break        

    # matches = np.argmax(mat, axis= 1)
    n = emb1.shape[0]
    matching = np.c_[ list(counterpart_dict.values()), np.linspace(0, n-1, n).astype(int)]
    matching = matching[np.argsort(matching[:,0])]
    matching = matching[matching[:, 0].argsort()]

    return matching

def eval_top_x_matching(matching, gt):
    n = float(len(gt))
    nn=len(gt)
    acc = 0.0
    for i in range(0,nn):
        if i in gt and np.isin(gt[i],matching[i, :]):
            acc += 1.0
    # print(acc/n)
    return acc / n



def eval_matching(matching, gt):
    n=float(len(gt))
    acc=0.0
    for i in matching:
        if i in gt and matching[i] == gt[i]:
                acc+=1.0
   # print(acc/n)
    return acc/n

def read_regal_matrix(file):

    nx_graph = nx.read_edgelist(file, nodetype=int, comments="%")
    A = nx.adjacency_matrix(nx_graph)
    n=int(np.shape(A)[0]/2)
    A1 = A[0:n,0:n]
    A2 = A[n:2*n,n:2*n]
    return A1.todense(), A2.todense()


def functional_maps_coupled_bases(A1, A2,q,k,laa,icp, icp_its, lower_t, upper_t, linsteps):
    #corresponding functions
    #eigenvectors
    #eigenvalues
    return 0

def calc_rotation_matrices(Cor1, Cor2, V1,V2,k,q):

    rotV1,_,rotV2=np.linalg.svd(V1[:,0:k].T@Cor1@Cor2.T@V2[:,0:k])

    return rotV1, rotV2.T

def calc_C_as_in_quasiharmonicpaper(Cor1,Cor2,V1,V2,k,q):
    leftside=Cor1.T@V1[:,0:k]
    rightside=V2[:,0:k].T@Cor2

    left=np.diag(leftside[0,:])

    right=rightside[:,0]
    for i in range(1,q):
        left=np.concatenate((left,np.diag(leftside[i,:])))
        right=np.concatenate((right,rightside[:,i]))

   # print(np.shape(left))
   # print(np.shape(right))

    C_diag=np.linalg.lstsq(left, right,rcond=None)[0]
   # print(C_diag)
  #  print(np.shape(C_diag))
    return np.diag(C_diag)

def eval_matching_top_x(matching, gt):

    n=float(len(gt))
    acc=0.0
    for i in matching:
        if i in gt and matching[i] == gt[i]:
                acc+=1.0
   # print(acc/n)
    return acc/n

def sort_greedy_voting(match_freq):
    dist_platt=np.ndarray.flatten(match_freq)
    idx = np.argsort(dist_platt)#
    n = match_freq.shape[0]
    k=idx//n
    r=idx%n
    idx_matr=np.c_[k,r]
# print(idx_matr)
    G1_elements=set()
    G2_elements=set()
    i= n**2 - 1
    j= 0
    matching=np.ones([n,2])*(n+1)
    while(len(G1_elements)<n):
        if (not idx_matr[i,0] in G1_elements) and (not idx_matr[i,1] in G2_elements):
            #print(idx_matr[i,:])
            matching[j,:]=idx_matr[i,:]

            G1_elements.add(idx_matr[i,0])
            G2_elements.add(idx_matr[i,1])
            j+=1
            #print(len(G1_elements))


        i-=1

    # print(idx)
    matching = np.c_[matching[:,0], matching[:,1]]
    real_matching = dict(matching[matching[:, 0].argsort()])
    return real_matching

if __name__ == '__main__':
    args = parse_args()
    main(args)
