import random
import math
import matplotlib.pyplot as plt
import numpy as np
from timeit import timeit
from functools import partial

random.seed(0)

# UTILITY CLASSES
class Point:
    def __init__(self, x, y):
        self.x = x
        self.y = y

    def __repr__(self):
        return f"Point({self.x}, {self.y})"

class Pair:
    def __init__(self, p1, p2):
        self.p1 = p1
        self.p2 = p2
        dx = (p1.x-p2.x)**2
        dy = (p1.y-p2.y)**2
        self.dist = math.sqrt(dx+dy)

    def __repr__(self):
        return f"Pair({self.p1}, {self.p2}), Distance({self.dist})"        

# UTILITY FUNCTIONS
def gen_rdm_pts(n):
    pts = []
    for i in range(n):
        pts.append(Point(random.random(), random.random()))
    return pts

def gen_rdm_pts_same_x(n):
    pts = []
    for i in range(n):
        pts.append(Point(.5, random.random()))
    return pts

def sort_by_x(pts):
    return sorted(pts, key=lambda point: point.x)

def sort_by_y(pts):
    return sorted(pts, key=lambda point: point.y) 


# DIVIDE AND CONQUER SOLUTION
def find_close_pair(pts):
    p_x = sort_by_x(pts)
    p_y = sort_by_y(pts)
    return find_close_pair_aux(p_x,p_y)

def find_close_pair_aux(p_x,p_y):
    
    if len(p_x) < 2:
        exit(1)
    if len(p_x) == 2:
        return Pair(p_x[0],p_x[1])
    elif len(p_x) == 3:
        return min([Pair(p_x[0],p_x[1]), Pair(p_x[0],p_x[2]), Pair(p_x[1],p_x[2])], key=lambda pair: pair.dist)

    q_x = p_x[:int(len(p_x)/2)]
    r_x = p_x[int(len(p_x)/2):]
    q_y, r_y = [], []
    for pt in p_y:
        if pt in q_x:
            q_y.append(pt)
        else:
            r_y.append(pt)    

    q_pair = find_close_pair_aux(q_x,q_y)
    r_pair = find_close_pair_aux(r_x,r_y)

    dlt = min(q_pair.dist,r_pair.dist)
    x_ = q_x[-1].x
    S = []
    for pt in p_y:
        if abs(pt.x-x_) <= dlt:
            S.append(pt)

    s_pair = q_pair
    for i, s in enumerate(S):
        for j in range(i+1,i+16):
            if j >= len(S):
                break
            elif Pair(s, S[j]).dist < s_pair.dist:
                s_pair = Pair(s, S[j])

    if s_pair.dist < dlt:
        return s_pair
    elif q_pair.dist < r_pair.dist:
        return q_pair
    else:
        return r_pair


# NAIVE SOLUTION
def naive_sol(pts):
    min_pair = Pair(pts[0],pts[1])
    for i, p in enumerate(pts):
        for j in range(i+1,len(pts)):
            if Pair(p,pts[j]).dist < min_pair.dist:
                min_pair = Pair(p,pts[j])
    return min_pair

# LINEAR BENCHMARK (NOT VALID)
def linear_benchmark(pts):
    min_pair = Pair(pts[0],pts[1])
    for j in range(2,len(pts)):
        if Pair(pts[0],pts[j]).dist < min_pair.dist:
            min_pair = Pair(pts[0],pts[j])
    return min_pair

# TESTING
def test_linear_benchmark(n):
    return linear_benchmark(gen_rdm_pts(n))
def test_naive_sol(n):
    return naive_sol(gen_rdm_pts(n))
def test_d_and_c(n):
    return find_close_pair(gen_rdm_pts(n))

if __name__ == "__main__":
    print("\nBASIC, 100 POINTS UNIFORMLY RANDOM IN PLANE:")
    pts = gen_rdm_pts(100)
    print(find_close_pair(pts))
    print(naive_sol(pts))

    print("\nEDGE CASE, 100 POINTS ALL SAME X:")
    pts = gen_rdm_pts_same_x(100)
    print(find_close_pair(pts))
    print(naive_sol(pts))

    print("\nEDGE CASE, ALL POINTS ALONG EDGE:")
    pts = [Point(0,0), Point(1,1), Point(0,1), Point(1,0), Point(0,.5), Point(.5,0), Point(1,.5), Point(.6,1)]
    print(find_close_pair(pts))
    print(naive_sol(pts))

    plt.rcParams['figure.figsize'] = [10, 6] # set size of plot

    ns = np.linspace(10, 15000, 10, dtype=int)

    naive_ts = [timeit(partial(test_naive_sol, n), number=1) for n in ns]

    d_and_c_time_ts = [timeit(partial(test_d_and_c, n), number=2) for n in ns]

    linear_ts = [timeit(partial(test_linear_benchmark, n), number=2) for n in ns]

    plt.plot(ns, naive_ts, 'or')
    plt.plot(ns, d_and_c_time_ts, 'ob')
    plt.plot(ns, linear_ts, 'og')
    plt.show()
    

    
