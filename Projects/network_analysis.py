import networkx as nx
import logging
from typing import Dict, List, Tuple, Set

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('network_analysis.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class NetworkAnalyzer:
    def __init__(self):
        """Ініціалізація графа та базових даних"""
        self.graph = nx.DiGraph()
        # Додаємо ребра з вагами (тривалістю в тижнях)
        edges = [
            (0, 1, 5), (1, 2, 7), (1, 3, 3), (2, 4, 20), (3, 4, 2),
            (4, 5, 8), (5, 6, 2), (5, 7, 15), (5, 8, 6), (6, 9, 2),
            (7, 9, 14), (8, 9, 1), (9, 10, 7), (10, 11, 11),
            (10, 12, 19), (11, 13, 9), (12, 13, 30), (13, 14, 3)
        ]
        for start, end, weight in edges:
            self.graph.add_edge(start, end, weight=weight)
        
        logging.info("Граф ініціалізовано з %d вершинами та %d ребрами", 
                    self.graph.number_of_nodes(), 
                    self.graph.number_of_edges())

    def find_all_paths(self) -> Dict[str, Tuple[List[int], int]]:
        """Знаходження всіх можливих шляхів від початкової до кінцевої вершини"""
        logging.info("Пошук всіх можливих шляхів від вершини 0 до 14")
        paths = {}
        path_number = 1
        
        def calculate_path_length(path: List[int]) -> int:
            return sum(self.graph[path[i]][path[i + 1]]['weight'] for i in range(len(path) - 1))

        for path in nx.all_simple_paths(self.graph, 0, 14):
            length = calculate_path_length(path)
            path_name = f"L{path_number}"
            paths[path_name] = (path, length)
            
            logging.info(f"Знайдено шлях {path_name}: {' -> '.join(map(str, path))}")
            logging.info(f"Тривалість шляху {path_name}: {length} тижнів")
            
            path_number += 1
        
        return paths

    def find_critical_path(self, paths: Dict[str, Tuple[List[int], int]]) -> Tuple[List[int], int]:
        """Визначення критичного шляху"""
        critical_path = max(paths.items(), key=lambda x: x[1][1])
        logging.info("Критичний шлях %s: %s", 
                    critical_path[0], 
                    ' -> '.join(map(str, critical_path[1][0])))
        logging.info("Тривалість критичного шляху: %d тижнів", critical_path[1][1])
        return critical_path[1]

    def calculate_earliest_times(self) -> Dict[int, int]:
        """Розрахунок найраніших термінів завершення подій"""
        logging.info("Розрахунок найраніших термінів завершення подій")
        earliest_times = {node: 0 for node in self.graph.nodes()}
        
        for node in nx.topological_sort(self.graph):
            for pred in self.graph.predecessors(node):
                earliest_times[node] = max(
                    earliest_times[node],
                    earliest_times[pred] + self.graph[pred][node]['weight']
                )
            logging.info(f"Тр{node} = {earliest_times[node]}")
        
        return earliest_times

    def calculate_latest_times(self, critical_length: int) -> Dict[int, int]:
        """Розрахунок найпізніших термінів завершення подій"""
        logging.info("Розрахунок найпізніших термінів завершення подій")
        latest_times = {node: critical_length for node in self.graph.nodes()}
        
        for node in reversed(list(nx.topological_sort(self.graph))):
            if list(self.graph.successors(node)):
                latest_times[node] = min(
                    latest_times[succ] - self.graph[node][succ]['weight']
                    for succ in self.graph.successors(node)
                )
            logging.info(f"Тп{node} = {latest_times[node]}")
        
        return latest_times

    def calculate_time_reserves(self, earliest_times: Dict[int, int], 
                              latest_times: Dict[int, int]) -> Dict[int, int]:
        """Розрахунок часових резервів подій"""
        logging.info("Розрахунок часових резервів подій")
        reserves = {}
        
        for node in self.graph.nodes():
            reserves[node] = latest_times[node] - earliest_times[node]
            logging.info(f"R{node} = Тп{node} - Тр{node} = {latest_times[node]} - {earliest_times[node]} = {reserves[node]}")
        
        return reserves

    def calculate_work_parameters(self, earliest_times: Dict[int, int], 
                                latest_times: Dict[int, int]) -> Dict[Tuple[int, int], Dict]:
        """Розрахунок параметрів робіт"""
        logging.info("Розрахунок параметрів робіт")
        work_params = {}
        
        for start, end in self.graph.edges():
            duration = self.graph[start][end]['weight']
            
            # Найраніший час початку
            earliest_start = earliest_times[start]
            # Найраніший час завершення
            earliest_finish = earliest_start + duration
            # Найпізніший час завершення
            latest_finish = latest_times[end]
            # Найпізніший час початку
            latest_start = latest_finish - duration
            
            # Повний резерв часу
            total_reserve = latest_finish - earliest_times[start] - duration
            
            # Частковий резерв першого типу
            first_type_reserve = latest_times[end] - earliest_times[start] - duration
            
            # Частковий резерв другого типу
            second_type_reserve = earliest_times[end] - earliest_times[start] - duration
            
            work_params[(start, end)] = {
                'duration': duration,
                'earliest_start': earliest_start,
                'earliest_finish': earliest_finish,
                'latest_start': latest_start,
                'latest_finish': latest_finish,
                'total_reserve': total_reserve,
                'first_type_reserve': first_type_reserve,
                'second_type_reserve': second_type_reserve
            }
            
            logging.info(f"\nРобота ({start},{end}):")
            logging.info(f"Тривалість: {duration}")
            logging.info(f"Найраніший початок: {earliest_start}")
            logging.info(f"Найраніше завершення: {earliest_finish}")
            logging.info(f"Найпізніший початок: {latest_start}")
            logging.info(f"Найпізніше завершення: {latest_finish}")
            logging.info(f"Повний резерв: {total_reserve}")
            logging.info(f"Частковий резерв 1-го типу: {first_type_reserve}")
            logging.info(f"Частковий резерв 2-го типу: {second_type_reserve}")
        
        return work_params

    def calculate_tension_coefficients(self, critical_path: List[int], 
                                    critical_length: int) -> Dict[Tuple[int, int], float]:
        """Розрахунок коефіцієнтів напруженості робіт"""
        logging.info("Розрахунок коефіцієнтів напруженості робіт")
        tension_coeffs = {}
        critical_edges = set(zip(critical_path[:-1], critical_path[1:]))
        
        for start, end in self.graph.edges():
            if (start, end) in critical_edges:
                tension_coeffs[(start, end)] = 1.0
                logging.info(f"Кн({start},{end}) = 1.0 (критичний шлях)")
                continue
            
            # Знаходження максимального шляху через дану роботу
            max_path_length = 0
            for path in nx.all_simple_paths(self.graph, 0, 14):
                if start in path and end in path:
                    length = sum(self.graph[path[i]][path[i + 1]]['weight'] 
                               for i in range(len(path) - 1))
                    max_path_length = max(max_path_length, length)
            
            # Довжина шляху без врахування даної роботи
            path_without_work = max_path_length - self.graph[start][end]['weight']
            
            # Розрахунок коефіцієнта напруженості
            if critical_length != path_without_work:
                tension = (max_path_length - path_without_work) / (critical_length - path_without_work)
                tension_coeffs[(start, end)] = round(tension, 2)
                logging.info(f"Кн({start},{end}) = {tension_coeffs[(start, end)]}")
            else:
                tension_coeffs[(start, end)] = 1.0
                logging.info(f"Кн({start},{end}) = 1.0")
        
        return tension_coeffs

    def analyze_network(self):
        """Повний аналіз сіткового графіка"""
        logging.info("Початок аналізу сіткового графіка")
        
        # Знаходження всіх шляхів
        paths = self.find_all_paths()
        
        # Визначення критичного шляху
        critical_path, critical_length = self.find_critical_path(paths)
        
        # Розрахунок найраніших термінів
        earliest_times = self.calculate_earliest_times()
        
        # Розрахунок найпізніших термінів
        latest_times = self.calculate_latest_times(critical_length)
        
        # Розрахунок резервів часу подій
        time_reserves = self.calculate_time_reserves(earliest_times, latest_times)
        
        # Розрахунок параметрів робіт
        work_parameters = self.calculate_work_parameters(earliest_times, latest_times)
        
        # Розрахунок коефіцієнтів напруженості
        tension_coefficients = self.calculate_tension_coefficients(critical_path, critical_length)
        
        logging.info("Аналіз сіткового графіка завершено")
        
        return {
            'paths': paths,
            'critical_path': (critical_path, critical_length),
            'earliest_times': earliest_times,
            'latest_times': latest_times,
            'time_reserves': time_reserves,
            'work_parameters': work_parameters,
            'tension_coefficients': tension_coefficients
        }

if __name__ == "__main__":
    analyzer = NetworkAnalyzer()
    results = analyzer.analyze_network()
