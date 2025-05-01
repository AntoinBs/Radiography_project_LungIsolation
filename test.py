def build_graph(l:list[str]) -> dict[str, list[str]]:
    graph = {}
    for line in l:
        parts = line.split("-")
        if parts[0] not in graph:
            graph[parts[0]] = []
        if parts[1] not in graph:
            graph[parts[1]] = []

        graph[parts[0]].append(parts[1])
        graph[parts[1]].append(parts[0])
    return graph
def bfs(l:list[str], start_room:str, end_room:str) -> list[str]:
    if start_room == end_room:
        return [start_room]
    
    graph = build_graph(l)

    queue = [start_room]
    visited = []
    predecessors = {start_room: None}

    while queue:
        current_room = queue.pop(0)
        visited.append(current_room)
        
        for neighbor in graph[current_room]:
            if neighbor not in visited and neighbor not in queue:
                queue.append(neighbor)
                predecessors[neighbor] = current_room
                if neighbor == end_room:
                    path = [end_room]
                    print(predecessors)
                    while predecessors[path[-1]] is not None:
                        path.append(predecessors[path[-1]])
                    return path[::-1]
    return []

if __name__ == "__main__":
    l = ["A-B", "C-D", "B-C", "B-D"]
    start_room = "A"
    end_room = "D"
    path = bfs(l, start_room, end_room)
    print(path)  # Output: ['A', 'B', 'C', 'D', 'E']