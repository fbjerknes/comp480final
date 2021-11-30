def read_graph(filename):
    f = open(filename)
    print("READING GRAPH")
    graph = {}
    i = 0
    a = f.readline()
    b = f.readline()
    while "\"" in f.readline():
        continue
    while "s" in f.readline():
        continue
    i = 0
    for line in f.readlines():
        i += 1
        if not i % 10000:
            print(i)
        l = line.split()
        d = [int(x) for x in l[:2]]
        d.append(float(l[2]))
        if d[0] not in graph:
            graph[d[0]] = {d[1]: d[2]}
        if d[1] not in graph:
            graph[d[1]] = {d[0]: d[2]}
        graph[d[0]][d[1]] = d[2]
        graph[d[1]][d[0]] = d[2]
    print("READING COMPLETE")
    return graph