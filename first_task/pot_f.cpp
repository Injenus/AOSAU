#include <iostream>
#include <vector>
#include <filesystem>
#include <opencv2/opencv.hpp>

using namespace std;
using namespace cv;

// Начальные и целевые позиции для каждого робота
pair<int, int> robot1_start = make_pair(0, 0);
pair<int, int> robot1_goal = make_pair(9, 9);

pair<int, int> robot2_start = make_pair(4, 0);
pair<int, int> robot2_goal = make_pair(5, 6);

pair<int, int> robot3_start = make_pair(2, 3);
pair<int, int> robot3_goal = make_pair(9, 0);

class ManhattanGraph {
public:
    ManhattanGraph(int width, int height) : width(width), height(height), obstacles(height, vector<int>(width, 0)) {
        all_coordinates.reserve(width * height);
        for (int x = 0; x < width; ++x) {
            for (int y = 0; y < height; ++y) {
                all_coordinates.push_back(make_pair(x, y));
            }
        }
    }

    void add_obstacle(pair<int, int> coordinates) {
        if (is_valid_coordinates(coordinates)) {
            obstacles[coordinates.second][coordinates.first] = 1;
        }
    }

    bool is_valid_coordinates(pair<int, int> coordinates) {
        int x = coordinates.first;
        int y = coordinates.second;
        return (x >= 0 && x < width && y >= 0 && y < height);
    }

    bool is_obstacle(pair<int, int> coordinates) {
        int x = coordinates.first;
        int y = coordinates.second;
        return obstacles[y][x] == 1;
    }

    bool is_valid_move(pair<int, int> current_coordinates, pair<int, int> target_coordinates) {
        if (is_valid_coordinates(current_coordinates) && is_valid_coordinates(target_coordinates)) {
            int x1 = current_coordinates.first;
            int y1 = current_coordinates.second;
            int x2 = target_coordinates.first;
            int y2 = target_coordinates.second;
            return (abs(x1 - x2) + abs(y1 - y2) == 1) && !is_obstacle(target_coordinates);
        }
        return false;
    }

    vector<pair<int, int>> get_neighbors(pair<int, int> coordinates) {
        int x = coordinates.first;
        int y = coordinates.second;
        vector<pair<int, int>> neighbors;
        neighbors.emplace_back(x + 1, y);
        neighbors.emplace_back(x - 1, y);
        neighbors.emplace_back(x, y + 1);
        neighbors.emplace_back(x, y - 1);

        vector<pair<int, int>> valid_neighbors;
        for (const auto& neighbor : neighbors) {
            if (is_valid_move(coordinates, neighbor)) {
                valid_neighbors.push_back(neighbor);
            }
        }
        return valid_neighbors;
    }

private:
    int width;
    int height;
    vector<vector<int>> obstacles;
    vector<pair<int, int>> all_coordinates;
};

class PotentialFields {
public:
    PotentialFields(ManhattanGraph& graph, vector<pair<int, int>>& robots, vector<pair<int, int>>& goals,
                    vector<pair<int, int>>& obstacles, double k_att, double k_rep, double k_rep_robots)
        : graph(graph), robots(robots), goals(goals), obstacles(obstacles), k_att(k_att), k_rep(k_rep),
          k_rep_robots(k_rep_robots), dd(0.1) {}

    pair<double, double> attractive_force(pair<int, int> robot, pair<int, int> goal) {
        int x1 = robot.first;
        int y1 = robot.second;
        int x2 = goal.first;
        int y2 = goal.second;
        double force_x = -k_att / (x1 - x2 + dd);
        double force_y = -k_att / (y1 - y2 + dd);
        return make_pair(force_x, force_y);
    }

    pair<double, double> repulsive_force(pair<int, int> robot, pair<int, int> obstacle) {
        int x1 = robot.first;
        int y1 = robot.second;
        int x2 = obstacle.first;
        int y2 = obstacle.second;
        double force_x = k_rep / pow(x1 - x2 + dd, 2);
        double force_y = k_rep / pow(y1 - y2 + dd, 2);
        return make_pair(force_x, force_y);
    }

    pair<double, double> repulsive_force_robots(pair<int, int> robot, vector<pair<int, int>>& other_robots) {
        double force_x = 0.0;
        double force_y = 0.0;
        for (const auto& other_robot : other_robots) {
            if (robot == other_robot) {
                continue;
            }
            int x1 = robot.first;
            int y1 = robot.second;
            int x2 = other_robot.first;
            int y2 = other_robot.second;
            force_x += k_rep_robots / pow(x1 - x2 + dd, 2);
            force_y += k_rep_robots / pow(y1 - y2 + dd, 2);
        }
        return make_pair(force_x, force_y);
    }

    vector<pair<double, double>> calculate_forces() {
        vector<pair<double, double>> forces;
        for (int i = 0; i < robots.size(); ++i) {
            double total_force_x = 0.0;
            double total_force_y = 0.0;
            for (const auto& goal : goals) {
                auto att_force = attractive_force(robots[i], goal);
                total_force_x += att_force.first;
                total_force_y += att_force.second;
            }
            for (const auto& obs : obstacles) {
                auto rep_force = repulsive_force(robots[i], obs);
                total_force_x += rep_force.first;
                total_force_y += rep_force.second;
            }
            auto rep_force = repulsive_force_robots(robots[i], robots);
            total_force_x += rep_force.first;
            total_force_y += rep_force.second;
            forces.push_back(make_pair(total_force_x, total_force_y);
        }
        return forces;
    }

    void move_robots() {
        auto forces = calculate_forces();
        vector<pair<int, int>> new_positions;
        for (int i = 0; i < robots.size(); ++i) {
            double force_x = forces[i].first;
            double force_y = forces[i].second;
            int x = robots[i].first;
            int y = robots[i].second;
            int dx, dy;

            // Находим направление движения (dx и dy)
            if (abs(force_x) > abs(force_y)) {
                dx = (force_x > 0) ? 1 : -1;
                dy = 0;
            } else {
                dx = 0;
                dy = (force_y > 0) ? 1 : -1;
            }

            int new_x = x + dx;
            int new_y = y + dy;

            // Проверяем, что новые координаты находятся внутри графа
            new_x = max(0, min(graph.width - 1, new_x));
            new_y = max(0, min(graph.height - 1, new_y);

            // Проверяем, что новые координаты не находятся на препятствиях и не пересекаются с другими роботами
            pair<int, int> new_coordinates(new_x, new_y);
            bool valid_move = !graph.is_obstacle(new_coordinates);
            for (int j = 0; j < robots.size(); ++j) {
                if (j != i && new_coordinates == robots[j]) {
                    valid_move = false;
                    break;
                }
            }

            if (valid_move) {
                new_positions.push_back(new_coordinates);
            } else {
                new_positions.push_back(robots[i]);  // Робот остается на месте, если попытался двигаться в препятствие или другого робота
            }
        }

        robots = new_positions;
    }

    vector<vector<pair<int, int>> run(int max_iterations) {
        vector<vector<pair<int, int>>> robot_paths(robots.size(), vector<pair<int, int>>());
        for (int i = 0; i < robots.size(); ++i) {
            robot_paths[i].push_back(robots[i]);
        }
        for (int iteration = 0; iteration < max_iterations; ++iteration) {
            move_robots();
            for (int i = 0; i < robots.size(); ++i) {
                robot_paths[i].push_back(robots[i]);
            }
        }
        return robot_paths;
    }

private:
    ManhattanGraph& graph;
    vector<pair<int, int>>& robots;
    vector<pair<int, int>>& goals;
    vector<pair<int, int>>& obstacles;
    double k_att;
    double k_rep;
    double k_rep_robots;
    double dd;
};

void save_map_image(ManhattanGraph& map_graph, int num_iter, vector<pair<int, int>>& poss) {
    int width = map_graph.width;
    int height = map_graph.height;

    // Создаем изображение, заполняя все клетки синим цветом
    Mat image(height, width, CV_8UC3, Scalar(255, 255, 255));  // Белый цвет для свободных клеток

    // Окрашиваем препятствия в красный цвет
    for (int x = 0; x < width; ++x) {
        for (int y = 0; y < height; ++y) {
            if (map_graph.is_obstacle(make_pair(x, y))) {
                image.at<Vec3b>(y, x) = Vec3b(0, 0, 255);  // Красный цвет для препятствий
            }
        }
    }

    for (int i = 0; i < poss.size(); ++i) {
        pair<int, int> pos = poss[i];
        image.at<Vec3b>(pos.second, pos.first) = Vec3b(0, 255 / 3 * (i + 1), 0);
    }

    imwrite("iters_pot/" + to_string(num_iter) + ".jpg", image);
}

int WinMain() { // с main ругается
    // Создание объекта графа
    ManhattanGraph map_graph(10, 10);

    // Добавление препятствий
    vector<pair<int, int>> obstacles = {make_pair(3, 3), make_pair(4, 4), make_pair(5, 5)};
    for (const auto& obs : obstacles) {
        map_graph.add_obstacle(obs);
    }

    // Параметры для потенциальных полей
    double k_att = 100.0;  // Коэффициент притяжения
    double k_rep = 100.0;  // Коэффициент отталкивания от препятствий
    double k_rep_robots = 100.0;  // Коэффициент отталкивания от других роботов

    // Создание объектов роботов
    vector<pair<int, int>> robots = {robot1_start, robot2_start, robot3_start};
    vector<pair<int, int>> goals = {robot1_goal, robot2_goal, robot3_goal};

    // Создание объекта планировщика на основе потенциальных полей
    PotentialFields pf_planner(map_graph, robots, goals, obstacles, k_att, k_rep, k_rep_robots);

    // Запуск метода потенциальных полей
    int max_iterations = 20;
    vector<vector<pair<int, int>> > robot_paths = pf_planner.run(max_iterations);

    // Вывод результатов
    vector<vector<pair<int, int>>> pot_paths = robot_paths;
    for (int i = 0; i < pot_paths.size(); ++i) {
        cout << "Path for Robot " << (i + 1) << ": ";
        for (const auto& pos : pot_paths[i]) {
            cout << "(" << pos.first << ", " << pos.second << ") ";
        }
        cout << endl;
    }

    string directory_name = "iters_pot";
    if (!filesystem::exists(directory_name)) {
        // Если директория не существует, создать её
        filesystem::create_directory(directory_name);
        cout << "Директория " << directory_name << " создана." << endl;
    } else {
        cout << "Директория " << directory_name << " уже существует." << endl;
    }

    for (int i = 0; i < max(pot_paths[0].size(), pot_paths[1].size(), pot_paths[2].size()); ++i) {
        pair<int, int> c1, c2, c3;
        if (i >= pot_paths[0].size()) {
            c1 = pot_paths[0].back();
        } else {
            c1 = pot_paths[0][i];
        }

        if (i >= pot_paths[1].size()) {
            c2 = pot_paths[1].back();
        } else {
            c2 = pot_paths[1][i];
        }

        if (i >= pot_paths[2].size()) {
            c3 = pot_paths[2].back();
        } else {
            c3 = pot_paths[2][i];
        }

        save_map_image(map_graph, i, {c1, c2, c3});
    }

    return 0;
}
