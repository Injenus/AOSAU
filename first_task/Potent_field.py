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


class PotentialFields:

    def __init__(self, graph, robots, goals, obstacles, k_att, k_rep,
                 k_rep_robots):
        self.graph = graph
        self.robots = robots
        self.goals = goals
        self.obstacles = obstacles
        self.k_att = k_att
        self.k_rep = k_rep
        self.k_rep_robots = k_rep_robots
        self.dd = 0.1

    def attractive_force(self, robot, goal):
        x1, y1 = robot
        x2, y2 = goal
        return -k_att / (x1 - x2 + self.dd), -k_att / (
                y1 - y2 + self.dd)

    def repulsive_force(self, robot, obstacle):
        x1, y1 = robot
        x2, y2 = obstacle
        return k_rep / (x1 - x2 + self.dd) ** 2, k_rep / (
                y1 - y2 + self.dd) ** 2

    def repulsive_force_robots(self, robot, other_robots):
        force_x, force_y = 0.0, 0.0
        for other_robot in other_robots:
            if robot == other_robot:
                continue
            x1, y1 = robot
            x2, y2 = other_robot
            force_x += k_rep_robots / (x1 - x2 + self.dd) ** 2
            force_y += k_rep_robots / (y1 - y2 + self.dd) ** 2
        return force_x, force_y

    def calculate_forces(self):
        forces = []
        for i, robot in enumerate(self.robots):
            total_force_x, total_force_y = 0.0, 0.0
            for goal in self.goals:
                att_force_x, att_force_y = self.attractive_force(robot, goal)
                total_force_x += att_force_x
                total_force_y += att_force_y
            for obs in self.obstacles:
                rep_force_x, rep_force_y = self.repulsive_force(robot, obs)
                total_force_x += rep_force_x
                total_force_y += rep_force_y
            rep_force_x, rep_force_y = self.repulsive_force_robots(robot,
                                                                   self.robots)
            total_force_x += rep_force_x
            total_force_y += rep_force_y
            forces.append((total_force_x, total_force_y))
        return forces

    def move_robots(self):
        forces = self.calculate_forces()
        new_positions = []
        for i, robot in enumerate(self.robots):
            force_x, force_y = forces[i]
            x, y = robot

            # Находим направление движения (dx и dy)
            if abs(force_x) > abs(force_y):
                dx = np.sign(force_x)
                dy = 0
            else:
                dx = 0
                dy = np.sign(force_y)

            new_x = x + dx
            new_y = y + dy

            # Проверяем, что новые координаты находятся внутри графа
            new_x = max(0, min(self.graph.width - 1, new_x))
            new_y = max(0, min(self.graph.height - 1, new_y))

            # Проверяем, что новые координаты не находятся на препятствиях и не пересекаются с другими роботами
            new_coordinates = (new_x, new_y)
            if (not self.graph.is_obstacle(new_coordinates) and
                    all(new_coordinates != r for j, r in enumerate(self.robots)
                        if j != i)):
                new_positions.append(new_coordinates)
            else:
                new_positions.append((x,
                                      y))  # Робот остается на месте, если попытался двигаться в препятствие или другого робота

        return new_positions

    def run(self, max_iterations):
        robot_paths = [[] for _ in self.robots]
        for i in range(len(self.robots)):
            robot_paths[i].append(self.robots[i])
        for iteration in range(max_iterations):
            new_positions = self.move_robots()
            self.robots = new_positions
            for i, robot in enumerate(self.robots):
                robot_paths[i].append(robot)
        return robot_paths


# Создание объекта графа
map_graph = ManhattanGraph(10, 10)

# Добавление препятствий
obstacles = [(3, 3), (4, 4), (5, 5)]
for obs in obstacles:
    map_graph.add_obstacle(obs)

# Параметры для потенциальных полей
k_att = 100.0  # Коэффициент притяжения
k_rep = 100.0  # Коэффициент отталкивания от препятствий
k_rep_robots = 100.0  # Коэффициент отталкивания от других робото

# Создание объектов роботов
robots = [robot1_start, robot2_start, robot3_start]
goals = [robot1_goal, robot2_goal, robot3_goal]

# Создание объекта планировщика на основе потенциальных полей
pf_planner = PotentialFields(map_graph, robots, goals, obstacles, k_att, k_rep,
                             k_rep_robots)

# Запуск метода потенциальных полей
max_iterations = 20
timer = time.time_ns()
robot_paths = pf_planner.run(max_iterations)
print("TIME (nanosec):", time.time_ns() - timer)

# Вывод результатов
pot_path1 = robot_paths[0]
pot_path2 = robot_paths[1]
pot_path3 = robot_paths[2]

print("Path for Robot 1:", pot_path1)
print("Path for Robot 2:", pot_path2)
print("Path for Robot 3:", pot_path3)

directory_name = "pot_fields_res"
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
        image[int(poss[i][1]), int(poss[i][0])] = [0, 255 / 3 * (i + 1), 0]

    plt.imshow(image)
    plt.title('Map')
    plt.axis('off')
    plt.savefig(directory_name + "/" + str(num_iter) + ".jpg",
                bbox_inches='tight',
                pad_inches=0)
    plt.close()


# p_iter = 0
# save_map_image(map_graph, p_iter, [[0, 0], [1, 1], [2, 2]])

for i in range(max(len(pot_path1), len(pot_path2), len(pot_path3))):
    if i > len(pot_path1) - 1:
        c1 = pot_path1[-1]
    else:
        c1 = pot_path1[i]

    if i > len(pot_path2) - 1:
        c2 = pot_path2[-1]
    else:
        c2 = pot_path2[i]

    if i > len(pot_path3) - 1:
        c3 = pot_path3[-1]
    else:
        c3 = pot_path3[i]

    save_map_image(map_graph, i, [c1, c2, c3])
