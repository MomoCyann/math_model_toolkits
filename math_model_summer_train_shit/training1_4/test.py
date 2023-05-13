points = [(1,22), (4,17), (5,18), (9,13), (20,23)]
# points = [[1,11,22], [4,17], [5,18], [9,13], [20,23]]
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
def walk(p, start, end):
    for m in moves[start]:    # try all moves from this point
        paths[p].append(m)    # keep track of our path
        if m == end:          # reached the end point for this path?
            if p+1 == len(points):   # no more paths?
                # if None not in grid[1:]:    # full coverage?
                for i,path in enumerate(paths):
                    print("%d." % (i+1), '-'.join(map(str, path)))
            else:
                _start, _end = points[p+1]  # now try to walk the next path
                walk(p+1, _start, _end)

        elif grid[m] is None:    # can we walk onto the next grid spot?
            grid[m] = p          # mark this spot as taken
            walk(p, m, end)
            grid[m] = None       # 每个方向都试过，证明这条路走不通，马上就要pop，标记这个点为未走过，unmark this spot
        paths[p].pop()       # backtrack on this path

grid = [None for i in range(26)]   # initialize the grid as empty points
for p in range(len(points)):
    start, end = points[p]
    paths.append([start])          # initialize path with its starting point
    grid[start] = grid[end] = p    # optimization: pre-set the known points

start, end = points[0]
walk(0, start, end)