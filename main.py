import argparse
from functionalMaps import *


def parse_args():
    parser = argparse.ArgumentParser(description="RUN GASP")
    parser.add_argument('--graph', nargs='?', default='arenas')
    # 1:nn 2:sortgreedy 3: jv

    parser.add_argument('--cor_func', type=int, default=3,
                        help='Corresponding Function. 1=Heat Kernel,2=PageRank, 3=Personalized PageRank')

    parser.add_argument('--k_span', type=int, default=10)
    parser.add_argument('--voting', type=int, default=1, help='1= SortGreedy, 2=Hungarian')

    parser.add_argument('--laa', type=int, default=3, help='Linear assignment algorithm. 1=nn,2=sortgreedy, 3=jv')
    parser.add_argument('--icp', type=bool, default=False)
    parser.add_argument('--ba', type=bool, default=True, help='Base alignment')
    parser.add_argument('--icp_its', type=int, default=3, help='how many iterations of iterative closest point')
    parser.add_argument('--q', type=int, default=20)
    parser.add_argument('--k', type=int, default=20)
    parser.add_argument('--lower_t', type=float, default=0.1, help='smallest timestep for corresponding functions')
    parser.add_argument('--upper_t', type=float, default=50.0, help='biggest timestep for corresponding functions')
    parser.add_argument('--linsteps', type=float, default=True,
                        help='scaling of time steps of corresponding functions, logarithmically or linearly')
    parser.add_argument('--reps', type=int, default=2, help='number of repetitions per noise level')
    parser.add_argument('--noise_levels', type=list,
                        default=[1])  # [5,10,15,20,25])#,11,12,13,14,15,16,17,18,19,20,21,22,23,24])
    return parser.parse_args()


def main(args):


    edge_list_G1 = 'graphs/arenas_orig.txt'
    edge_list_G2 = 'graphs/arenas_n5.txt'

    gt_file = 'graphs/arenas_n5_gt.txt'

    data = np.loadtxt(gt_file, delimiter=" ")
        #
    data= data.astype(int)


    gt = dict(data)
    A1 = edgelist_to_adjmatrix(edge_list_G1)

    A2 = edgelist_to_adjmatrix(edge_list_G2)

    matching,a,b,c=functional_maps(A1,A2,args.q, args.k,args.laa,args.icp, args.icp_its, args.lower_t, args.upper_t, args.linsteps,args.graph,1,1)

    acc = eval_matching(matching, gt)
    print("Accuracy GRASP: %f" %(acc))
    print('\n')

if __name__ == '__main__':
    args = parse_args()
    main(args)

