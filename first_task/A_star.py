import heapq
import matplotlib.pyplot as plt
import os
import numpy as np
import time

# Начальные и целевые позиции для каждого робота
robot1_start = (0, 0)
robot1_goal = (9, 9)

robot2_start = (4, 0)
robot2_goal = (5, 6)

robot3_start = (2, 3)
robot3_goal = (9, 0)


class ManhattanGraph:

    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.obstacles = set()
        self.graph = [[0 for _ in range(self.height)] for _ in
                      range(self.width)]
        self.all_coordinates = [(x, y) for x in range(self.width) for y in
                                range(self.height)]

    def add_obstacle(self, coordinates):
        if self.is_valid_coordinates(coordinates):
            self.obstacles.add(coordinates)
            x, y = coordinates
            self.graph[x][y] = 1

    def is_valid_coordinates(self, coordinates):
        x, y = coordinates
        return 0 <= x < self.width and 0 <= y < self.height

    def is_obstacle(self, coordinates):
        return coordinates in self.obstacles

    def is_valid_move(self, current_coordinates, target_coordinates):
        if (self.is_valid_coordinates(current_coordinates) and
                self.is_valid_coordinates(target_coordinates)):
            x1, y1 = current_coordinates
            x2, y2 = target_coordinates
            return abs(x1 - x2) + abs(y1 - y2) == 1 and not self.is_obstacle(
                target_coordinates)
        return False

    def get_neighbors(self, coordinates):
        x, y = coordinates
        neighbors = [(x + 1, y), (x - 1, y), (x, y + 1), (x, y - 1)]
        valid_neighbors = [neighbor for neighbor in neighbors if
                           self.is_valid_move(coordinates, neighbor)]
        return valid_neighbors


class AStar:

    def __init__(self, graph, start, goal):
        self.graph = graph
        self.start = start
        self.goal = goal
        self.open_set = []
        self.came_from = {}
        self.g_score = {coord: float('inf') for coord in graph.all_coordinates}
        self.g_score[start] = 0
        self.f_score = {coord: float('inf') for coord in graph.all_coordinates}
        self.f_score[start] = self.heuristic(start, goal)

    def heuristic(self, current, goal):
        x1, y1 = current
        x2, y2 = goal
        return abs(x1 - x2) + abs(y1 - y2)

    def reconstruct_path(self, current):
        path = []
        while current in self.came_from:
            path.append(current)
            current = self.came_from[current]
        path.append(self.start)
        path.reverse()
        return path

    def check_collision(self, path, other_paths):
        for step in path:
            if step in other_paths:
                return True
        return False

    def a_star(self):
        heapq.heappush(self.open_set, (self.f_score[self.start], self.start))

        while self.open_set:
            _, current = heapq.heappop(self.open_set)

            if current == self.goal:
                return self.reconstruct_path(current)

            for neighbor in self.graph.get_neighbors(current):
                tentative_g_score = self.g_score[current] + 1

                if tentative_g_score < self.g_score[neighbor]:
                    self.came_from[neighbor] = current
                    self.g_score[neighbor] = tentative_g_score
                    self.f_score[
                        neighbor] = tentative_g_score + self.heuristic(
                        neighbor, self.goal)
                    heapq.heappush(self.open_set,
                                   (self.f_score[neighbor], neighbor))

        return [self.start]


# Создание объекта графа
map_graph = ManhattanGraph(10, 10)

# Добавление препятствий
# obstacles = [(3, 3), (3, 4), (3, 5), (4, 4), (8, 9), (8, 7), (8, 6), (8, 8),
#              (2, 1), (3, 1), (1, 1)]
# obstacles = [(3, 3), (3, 4), (3, 5), (4, 4),(0,5),(2,5)]
# obstacles = [(3, 3), (3, 4), (3, 5), (4, 4), (0, 5), (2, 5), (0, 1), (1, 0),
#              (2, 8), (2, 7)]
obstacles = [(3, 3), (4, 4), (5, 5)]
for obs in obstacles:
    map_graph.add_obstacle(obs)


def pred_heur(strat, fin):
    x1, y1 = strat
    x2, y2 = fin
    return abs(x1 - x2) + abs(y1 - y2)


# Поиск пути для каждого робота
a_star1 = AStar(map_graph, robot1_start, robot1_goal)
a_star2 = AStar(map_graph, robot2_start, robot2_goal)
a_star3 = AStar(map_graph, robot3_start, robot3_goal)
timer = time.time_ns()
path1 = a_star1.a_star()
path2 = a_star2.a_star()
path3 = a_star3.a_star()
print("TIME (nanosec):", time.time_ns() - timer)


def duplicate_to_length(lst, number):
    while len(lst) < number:
        last_element = lst[-1]
        lst.append(last_element)


def dupl_all(max_len):
    duplicate_to_length(path1, max_len)
    duplicate_to_length(path2, max_len)
    duplicate_to_length(path3, max_len)


print("Path for Robot 1:", path1)
print("Path for Robot 2:", path2)
print("Path for Robot 3:", path3)

isOk = False

while not isOk:
    isOk = True
    for i in range(len(path2)):
        if path2[i] == path1[min(i, len(path1) - 1)]:
            path2.insert(i, path2[i - 1])
            isOk = False

isOk = False
while not isOk:
    isOk = True
    for i in range(len(path3)):
        if path3[i] == path2[min(i, len(path2) - 1)] or path3[i] == path1[
            min(i, len(path1) - 1)]:
            path3.insert(i, path3[i - 1])
            isOk = False

max_l = max(len(path1), len(path2), len(path3))
dupl_all(max_l)

directory_name = "a_star_res"
if not os.path.exists(directory_name):
    # Если директория не существует, создать её
    os.mkdir(directory_name)
    print(f"Директория {directory_name} создана.")
else:
    print(f"Директория {directory_name} уже существует.")


def save_map_image(map_graph, num_iter, poss):
    width = map_graph.width
    height = map_graph.height

    # Создаем изображение, заполняя все клетки синим цветом
    image = np.zeros((height, width, 3), dtype=np.uint8)
    image[:, :] = [0, 0, 255]  # Синий цвет для свободных клеток

    # Окрашиваем препятствия в красный цвет
    for x in range(width):
        for y in range(height):
            if map_graph.is_obstacle((x, y)):
                image[y, x] = [255, 0, 0]  # Красный цвет для препятствий

    for i in range(len(poss)):
        image[poss[i][1], poss[i][0]] = [0, 255 / 3 * (i + 1), 0]

    plt.imshow(image)
    plt.title('Map')
    plt.axis('off')
    plt.savefig(directory_name + "/" + str(num_iter) + ".jpg",
                bbox_inches='tight',
                pad_inches=0)
    plt.close()


# p_iter = 0
# save_map_image(map_graph, p_iter, [[0, 0], [1, 1], [2, 2]])

for i in range(max(len(path1), len(path2), len(path3))):
    if i > len(path1) - 1:
        c1 = path1[-1]
    else:
        c1 = path1[i]

    if i > len(path2) - 1:
        c2 = path2[-1]
    else:
        c2 = path2[i]

    if i > len(path3) - 1:
        c3 = path3[-1]
    else:
        c3 = path3[i]

    save_map_image(map_graph, i, [c1, c2, c3])
