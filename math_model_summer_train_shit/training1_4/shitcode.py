import copy
import sys  # 导入sys模块
import numpy as np
# points = [(1,22), (4,17), (5,18), (9,13), (20,23)]
points = [[1,11,22], [4,17], [5,18], [9,13], [20,23]]
paths = []

# 每个点的下一步能走的位置 四周
moves = [None]    # set position 0 with None
for i in range(1,26):
    m = []
    if i % 5 != 0:    # move right
        m.append(i+1)
    if i % 5 != 1:    # move left
        m.append(i-1)
    if i > 5:         # move up
        m.append(i-5)
    if i < 21:        # move down
        m.append(i+5)
    moves.append(m)

# Recursive function to walk path 'p' from 'start' to 'end'
def walk(p, start, ends, flag):
    # 复制一个ends用于记录还剩下那些点没有到达
    ends_mark = ends.copy()
    for m in moves[start]:    # try all moves from this point
        paths[p].append(m)    # keep track of our path
        if m in ends:          # reached the end point for this path?
            ends_mark.remove(m)
            if len(ends_mark) == 0:         # 这一条线路的每个点都走到了
                if p+1 == len(points):   # no more paths?
                    # if None not in grid[1:]:    # full coverage?
                    for i,path in enumerate(paths):
                        print("%d." % (i+1), '-'.join(map(str, path)))
                else:
                    # ends_mark空了，这一组点到达完了，下一组
                    _start, _ends = points[p+1][0], points[p+1][1:]  # now try to walk the next path
                    flag = p+1
                    walk(p+1, _start, _ends, flag)
            else:           # 还有点没走完，把某一个end当做起点,此时ends删除了一个元素，继续递归，然后flag+10
                _start = m
                _ends = ends_mark
                flag += 100
                walk(p, _start, _ends, flag)
        elif (grid[m] is None):    # can we walk onto the next grid spot?
            grid[m] = flag          # mark this spot as taken
            walk(p, m, ends, flag)
            grid[m] = None       # 每个方向都试过，证明这条路走不通，马上就要pop，标记这个点为未走过，unmark this spot
        elif (grid[m]>=100) and ((flag-grid[m])>=100):
            if (grid[m]==flag%100) or (np.absolute(flag-grid[m])%100==0):
                grid[m] = flag  # mark this spot as taken
                walk(p, m, ends, flag)
                grid[m] = grid[m] % 100

        paths[p].pop()       # backtrack on this path

grid = [None for i in range(26)]   # initialize the grid as empty points
for p in range(len(points)):
    start = points[p][0]
    paths.append([start])          # initialize path with its starting point
    for s_m_e in points[p]:
        grid[s_m_e] = p    # optimization: pre-set the known points/每个起点 终点 中点 都设为p

print(points[0][0])
start= points[0][0]
ends =points[0][1:]

walk(0, start, ends, 0)