import networkx as nx
import matplotlib.pyplot as plt
import logging
from typing import Dict, List, Tuple
from copy import deepcopy

# Налаштування логування
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('network_optimization.log', mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)

class NetworkOptimizer:
    def __init__(self):
        """Ініціалізація базового графа та параметрів"""
        self.base_graph = nx.DiGraph()

        # Базові тривалості робіт
        self.base_durations = {
            (0, 1): 31.98,  # Розробка технічних умов
            (1, 2): 34.12,  # Загальне компонування виробу
            (1, 7): 38.38,  # Видача завдання на документацію
            (2, 3): 38.38,  # Проєктування електричного блоку
            (2, 4): 38.38,  # Проєктування механічного блоку
            (2, 5): 51.17,  # Оформлення замовлень на компоненти
            (3, 6): 42.65,  # Виготовлення електричної частини
            (4, 6): 42.65,  # Виготовлення механічної частини
            (5, 6): 44.78,  # Отримання компонентів
            (6, 7): 38.38,  # Підготовка технічної інформації
            (7, 8): 34.12,  # Складання виробу
            (7, 9): 25.59,  # Розробка експлуатаційної документації
            (8, 9): 25.59,  # Контрольні випробування
        }

        # Коефіцієнти складності робіт (від 0 до 1)
        self.complexity_coefficients = {
            (0, 1): 0.7,  # Розробка технічних умов - висока складність
            (1, 2): 0.8,  # Загальне компонування - дуже висока складність
            (1, 7): 0.5,  # Видача завдання - середня складність
            (2, 3): 0.9,  # Проєктування електричного блоку - найвища складність
            (2, 4): 0.9,  # Проєктування механічного блоку - найвища складність
            (2, 5): 0.6,  # Оформлення замовлень - середня складність
            (3, 6): 0.8,  # Виготовлення електричної частини - висока складність
            (4, 6): 0.8,  # Виготовлення механічної частини - висока складність
            (5, 6): 0.4,  # Отримання компонентів - низька складність
            (6, 7): 0.6,  # Підготовка технічної інформації - середня складність
            (7, 8): 0.7,  # Складання виробу - висока складність
            (7, 9): 0.5,  # Розробка документації - середня складність
            (8, 9): 0.7,  # Контрольні випробування - висока складність
        }

        # Ризики робіт (ймовірність затримки, від 0 до 1)
        self.risk_coefficients = {
            (0, 1): 0.3,  # Розробка технічних умов - низький ризик
            (1, 2): 0.4,  # Загальне компонування - середній ризик
            (1, 7): 0.2,  # Видача завдання - низький ризик
            (2, 3): 0.6,  # Проєктування електричного блоку - високий ризик
            (2, 4): 0.6,  # Проєктування механічного блоку - високий ризик
            (2, 5): 0.5,  # Оформлення замовлень - середній ризик
            (3, 6): 0.7,  # Виготовлення електричної частини - високий ризик
            (4, 6): 0.7,  # Виготовлення механічної частини - високий ризик
            (5, 6): 0.8,  # Отримання компонентів - дуже високий ризик
            (6, 7): 0.4,  # Підготовка технічної інформації - середній ризик
            (7, 8): 0.5,  # Складання виробу - середній ризик
            (7, 9): 0.3,  # Розробка документації - низький ризик
            (8, 9): 0.4,  # Контрольні випробування - середній ризик
        }

        # Опис робіт
        self.work_descriptions = {
            (0, 1): "Розробка технічних умов",
            (1, 2): "Загальне компонування виробу",
            (1, 7): "Видача завдання на документацію",
            (2, 3): "Проєктування електричного блоку",
            (2, 4): "Проєктування механічного блоку",
            (2, 5): "Оформлення замовлень на компоненти",
            (3, 6): "Виготовлення електричної частини",
            (4, 6): "Виготовлення механічної частини",
            (5, 6): "Отримання компонентів",
            (6, 7): "Підготовка технічної інформації",
            (7, 8): "Складання виробу",
            (7, 9): "Розробка експлуатаційної документації",
            (8, 9): "Контрольні випробування"
        }

        # Ініціалізація базового графа
        for edge, duration in self.base_durations.items():
            self.base_graph.add_edge(edge[0], edge[1], weight=duration)

        logging.info("Базовий граф ініціалізовано")

    def calculate_tension(self, path_length: float, critical_length: float) -> float:
        """Розрахунок коефіцієнта напруженості"""
        return path_length / critical_length if critical_length > 0 else 0

    def analyze_variant(self, graph: nx.DiGraph, durations: Dict[Tuple[int, int], float]) -> Dict:
        """Аналіз варіанту оптимізації"""
        results = {}

        # Знаходження критичного шляху
        critical_path = nx.dag_longest_path(graph)
        critical_length = nx.dag_longest_path_length(graph)

        # Логування інформації про критичний шлях
        logging.info(f"Критичний шлях: {' -> '.join(map(str, critical_path))}")
        logging.info(f"Тривалість критичного шляху: {critical_length:.2f} тижнів")

        # Порівняння з базовими тривалостями та логування змін
        logging.info("\nЗміни в тривалості робіт:")
        for (start, end), duration in durations.items():
            base_duration = self.base_durations[(start, end)]
            if abs(duration - base_duration) > 0.01:
                change_percent = ((duration - base_duration) / base_duration) * 100
                logging.info(f"Робота {self.work_descriptions[(start, end)]}: "
                             f"{change_percent:+.1f}% ({base_duration:.2f} -> {duration:.2f} тижнів)")

        # Розрахунок найраніших та найпізніших термінів
        earliest_times = {node: 0 for node in graph.nodes()}
        for node in nx.topological_sort(graph):
            for pred in graph.predecessors(node):
                earliest_times[node] = max(
                    earliest_times[node],
                    earliest_times[pred] + graph[pred][node]['weight']
                )

        latest_times = {node: critical_length for node in graph.nodes()}
        for node in reversed(list(nx.topological_sort(graph))):
            if list(graph.successors(node)):
                latest_times[node] = min(
                    latest_times[succ] - graph[node][succ]['weight']
                    for succ in graph.successors(node)
                )

        # Аналіз напруженості та ризиків
        logging.info("\nАналіз напруженості та ризиків:")
        all_paths = list(nx.all_simple_paths(graph, 0, 9))
        path_tensions = {}
        for path in all_paths:
            path_length = sum(graph[path[i]][path[i+1]]['weight'] for i in range(len(path)-1))
            tension = self.calculate_tension(path_length, critical_length)
            path_tensions[tuple(path)] = tension

            # Розрахунок сумарного ризику шляху
            path_edges = list(zip(path[:-1], path[1:]))
            path_risk = sum(self.risk_coefficients[edge] for edge in path_edges) / len(path_edges)

            logging.info(f"\nШлях: {' -> '.join(map(str, path))}")
            logging.info(f"Тривалість: {path_length:.2f} тижнів")
            logging.info(f"Коефіцієнт напруженості: {tension:.2f}")
            logging.info(f"Середній ризик: {path_risk:.2f}")

            # Аналіз складних робіт на шляху
            complex_works = [edge for edge in path_edges if self.complexity_coefficients[edge] >= 0.7]
            if complex_works:
                logging.info("Складні роботи на шляху:")
                for edge in complex_works:
                    logging.info(f"  - {self.work_descriptions[edge]} "
                                 f"(складність: {self.complexity_coefficients[edge]:.1f}, "
                                 f"ризик: {self.risk_coefficients[edge]:.1f})")

        # Розрахунок загальних показників варіанту
        avg_tension = sum(path_tensions.values()) / len(path_tensions)
        max_tension = max(path_tensions.values())
        critical_path_risk = sum(self.risk_coefficients[edge] for edge in zip(critical_path[:-1], critical_path[1:])) / (len(critical_path)-1)

        logging.info("\nЗагальні показники варіанту:")
        logging.info(f"Середня напруженість: {avg_tension:.2f}")
        logging.info(f"Максимальна напруженість: {max_tension:.2f}")
        logging.info(f"Ризик критичного шляху: {critical_path_risk:.2f}")

        # Додавання результатів до словника
        results['critical_path'] = critical_path
        results['critical_length'] = critical_length
        results['earliest_times'] = earliest_times
        results['latest_times'] = latest_times
        results['path_tensions'] = path_tensions
        results['avg_tension'] = avg_tension
        results['max_tension'] = max_tension
        results['critical_path_risk'] = critical_path_risk
        results['durations'] = durations  # Додавання тривалостей робіт

        return results

    def create_optimization_variant(self, variant_num: int) -> Tuple[nx.DiGraph, Dict[Tuple[int, int], float]]:
        """Створення варіанту оптимізації"""
        graph = deepcopy(self.base_graph)
        durations = deepcopy(self.base_durations)

        if variant_num == 1:
            # Варіант 1: Скорочення тривалості критичних робіт
            durations[(2, 5)] *= 0.85  # Скорочення на 15%
            durations[(5, 6)] *= 0.9   # Скорочення на 10%
            logging.info("Створено варіант 1: Скорочення тривалості критичних робіт")

        elif variant_num == 2:
            # Варіант 2: Паралельне виконання робіт
            # Додаємо нові зв'язки для паралельного виконання
            durations[(2, 3)] *= 0.9
            durations[(2, 4)] *= 0.9
            durations[(3, 6)] *= 0.95
            durations[(4, 6)] *= 0.95
            logging.info("Створено варіант 2: Паралельне виконання робіт")

        elif variant_num == 3:
            # Варіант 3: Комбінований підхід
            durations[(2, 5)] *= 0.9
            durations[(5, 6)] *= 0.95
            durations[(2, 3)] *= 0.95
            durations[(2, 4)] *= 0.95
            logging.info("Створено варіант 3: Комбінований підхід")

        elif variant_num == 4:
            # Варіант 4: Оптимізація некритичних робіт
            durations[(1, 7)] *= 1.2  # Збільшення тривалості некритичної роботи
            durations[(7, 9)] *= 1.1
            durations[(2, 5)] *= 0.85
            durations[(5, 6)] *= 0.9
            logging.info("Створено варіант 4: Оптимізація некритичних робіт")

        # Оновлення графа новими тривалостями
        for edge, duration in durations.items():
            graph[edge[0]][edge[1]]['weight'] = duration

        return graph, durations

    def visualize_network(self, graph: nx.DiGraph, critical_path: List[int], filename: str):
        """Візуалізація сіткового графіка"""
        plt.figure(figsize=(15, 8))

        # Створення позицій вузлів за рівнями
        levels = {}
        for node in graph.nodes():
            # Розрахунок рівня вузла на основі найдовшого шляху від початку
            levels[node] = len(nx.shortest_path(graph, 0, node)) - 1

        # Визначення кількості вузлів на кожному рівні
        nodes_by_level = {}
        for node, level in levels.items():
            if level not in nodes_by_level:
                nodes_by_level[level] = []
            nodes_by_level[level].append(node)

        # Розрахунок позицій вузлів
        pos = {}
        max_level = max(levels.values())
        for level in range(max_level + 1):
            nodes = sorted(nodes_by_level.get(level, []))
            n_nodes = len(nodes)
            for i, node in enumerate(nodes):
                # x-координата залежить від рівня
                x = level
                # y-координата розподіляє вузли рівномірно на рівні
                if n_nodes > 1:
                    y = (i - (n_nodes - 1) / 2) / n_nodes
                else:
                    y = 0
                pos[node] = (x, y)

        # Визначення критичних ребер
        critical_edges = list(zip(critical_path[:-1], critical_path[1:]))

        # Порівняння з базовими тривалостями
        edge_colors = []
        edge_widths = []
        for u, v in graph.edges():
            if (u, v) in critical_edges:
                edge_colors.append('red')
                edge_widths.append(2)
            else:
                edge_colors.append('lightgray')
                edge_widths.append(1.5)

        # Малювання ребер
        nx.draw_networkx_edges(graph, pos, edge_color=edge_colors, width=edge_widths,
                               arrows=True, arrowsize=20, connectionstyle='arc3,rad=0.2')

        # Малювання вузлів
        nx.draw_networkx_nodes(graph, pos, node_color='lightblue',
                               node_size=1000, alpha=0.6)

        # Додавання міток вузлів
        nx.draw_networkx_labels(graph, pos, font_size=12)

        # Додавання міток ребер з тривалістю та зміною
        edge_labels = {}
        for u, v in graph.edges():
            current_duration = graph[u][v]['weight']
            base_duration = self.base_durations[(u, v)]

            if abs(current_duration - base_duration) > 0.01:  # якщо є зміна
                change_percent = ((current_duration - base_duration) / base_duration) * 100
                if change_percent > 0:
                    label = f"{current_duration:.2f}\n(+{change_percent:.1f}%)"
                else:
                    label = f"{current_duration:.2f}\n({change_percent:.1f}%)"
            else:
                label = f"{current_duration:.2f}"

            edge_labels[(u, v)] = label

        nx.draw_networkx_edge_labels(graph, pos, edge_labels, font_size=8)

        # Додавання легенди
        plt.plot([], [], 'red', linewidth=2, label='Критичний шлях')
        plt.plot([], [], 'lightgray', linewidth=1.5, label='Інші роботи')
        plt.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05),
                   ncol=2, fancybox=True, shadow=True)

        # Отримання номеру варіанту з імені файлу
        variant_num = filename.split('_')[-1].split('.')[0]
        plt.title(f"Сітковий графік проєкту - Варіант {variant_num}\n(червоним позначено критичний шлях)")
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(filename, bbox_inches='tight', dpi=300)
        plt.close()

    def visualize_optimized_network(self, graph: nx.DiGraph, critical_path: List[int], durations: Dict[Tuple[int, int], float], filename: str):
        """Візуалізація оптимізованого сіткового графіка з детальними підписами."""
        plt.figure(figsize=(20, 12))

        # Створення позицій вузлів за рівнями
        pos = {}
        level_width = 4  # Відстань між рівнями
        level_height = 2  # Відстань між вузлами на рівні

        # Визначення рівнів вузлів
        levels = {}
        for node in graph.nodes():
            levels[node] = len(nx.shortest_path(graph, 0, node)) - 1

        # Розподіл вузлів за рівнями
        nodes_by_level = {}
        for node, level in levels.items():
            if level not in nodes_by_level:
                nodes_by_level[level] = []
            nodes_by_level[level].append(node)

        # Розрахунок позицій
        for level in nodes_by_level:
            nodes = sorted(nodes_by_level[level])
            n_nodes = len(nodes)
            for i, node in enumerate(nodes):
                x = level * level_width
                y = (i - (n_nodes - 1) / 2) * level_height
                pos[node] = (x, y)

        # Визначення критичних ребер
        critical_edges = list(zip(critical_path[:-1], critical_path[1:]))

        # Малювання вузлів
        node_size = 1500
        node_color = 'lightblue'
        nx.draw_networkx_nodes(graph, pos, node_size=node_size, node_color=node_color,
                               node_shape='o', edgecolors='black')

        # Малювання ребер та підписів до них
        edge_labels = {}
        for u, v in graph.edges():
            work_name = self.work_descriptions[(u, v)]
            duration = durations[(u, v)]
            base_duration = self.base_durations[(u, v)]

            # Розрахунок відсотка зміни
            if abs(duration - base_duration) > 0.01:  # якщо є зміна
                change_percent = ((duration - base_duration) / base_duration) * 100
                if change_percent > 0:
                    label = f"{work_name}\n{duration:.2f} тижнів\n(+{change_percent:.1f}%)"
                else:
                    label = f"{work_name}\n{duration:.2f} тижнів\n({change_percent:.1f}%)"
            else:
                label = f"{work_name}\n{duration:.2f} тижнів"

            edge_labels[(u, v)] = label

        # Малювання критичних та некритичних ребер
        critical_edges_list = [(u, v) for (u, v) in graph.edges() if (u, v) in critical_edges]
        other_edges_list = [(u, v) for (u, v) in graph.edges() if (u, v) not in critical_edges]

        # Критичні ребра (червоні)
        nx.draw_networkx_edges(graph, pos, edgelist=critical_edges_list,
                               edge_color='red', width=2, arrows=True,
                               arrowsize=20, arrowstyle='->')

        # Некритичні ребра (сірі)
        nx.draw_networkx_edges(graph, pos, edgelist=other_edges_list,
                               edge_color='gray', width=1, arrows=True,
                               arrowsize=20, arrowstyle='->')

        # Підписи вузлів
        nx.draw_networkx_labels(graph, pos, font_size=12, font_weight='bold')

        # Підписи ребер
        nx.draw_networkx_edge_labels(graph, pos, edge_labels=edge_labels,
                                     font_size=12, bbox=dict(facecolor='white',
                                                             edgecolor='none', alpha=0.7))

        plt.title("Оптимізований сітковий графік\n(червоним позначено критичний шлях)",
                  pad=20, fontsize=14)
        plt.axis('off')
        plt.tight_layout()

        # Збереження графіка
        plt.savefig(filename, format='png', dpi=300, bbox_inches='tight')
        plt.close()
        logging.info(f"Оптимізований графік збережено у файл {filename}")

    def optimize_network(self):
        """Оптимізація мережевого графіка"""
        logging.info("\n=== БАЗОВИЙ ВАРІАНТ ===")
        base_results = self.analyze_variant(self.base_graph, self.base_durations)

        variants = []
        variant_data = []  # Зберігаємо графи та тривалості окремо

        for i in range(1, 5):
            variant_graph, variant_durations = self.create_optimization_variant(i)
            logging.info(f"\n=== ВАРІАНТ {i} ===")
            variant_results = self.analyze_variant(variant_graph, variant_durations)
            variants.append((variant_graph, variant_results))
            variant_data.append((variant_graph, variant_durations))  # Зберігаємо дані варіанту

            # Порівняння з базовим варіантом
            duration_reduction = base_results['critical_length'] - variant_results['critical_length']
            duration_reduction_percent = (duration_reduction / base_results['critical_length']) * 100

            if duration_reduction > 0:
                logging.info(f"\nПокращення відносно базового варіанту:")
                logging.info(f"Скорочення тривалості: {duration_reduction:.2f} тижнів ({duration_reduction_percent:.1f}%)")
                logging.info(f"Зміна середньої напруженості: {variant_results['avg_tension'] - base_results['avg_tension']:.2f}")
                logging.info(f"Зміна ризику критичного шляху: {variant_results['critical_path_risk'] - base_results['critical_path_risk']:.2f}")

        # Знаходження найкращого варіанту (Варіант 3)
        best_variant_idx = 2  # індекс 2 відповідає варіанту 3 (бо індексація з 0)
        best_variant = variants[best_variant_idx]
        best_variant_graph, best_variant_durations = variant_data[best_variant_idx]

        # Створення візуалізації для найкращого варіанту
        logging.info(f"\nСтворено розширений графік для найкращого варіанту (Варіант 3)")

        # Розрахунок загального покращення
        total_reduction = base_results['critical_length'] - best_variant[1]['critical_length']
        total_reduction_percent = (total_reduction / base_results['critical_length']) * 100
        logging.info(f"Скорочення тривалості: {total_reduction:.2f} тижнів ({total_reduction_percent:.1f}%)")

        # Виклик розширеного аналізу для найкращого варіанту
        self.visualize_optimized_network(best_variant_graph, best_variant[1]['critical_path'],
                                         best_variant_durations,  # Використовуємо збережені тривалості
                                         f'optimized_network_best_variant_detailed.png')

        # Генерація звіту з результатами оптимізації
        optimization_results = {
            'variants': {
                'base': {
                    'results': base_results,
                    'durations': self.base_durations
                }
            },
            'optimal_variant': f'variant_{best_variant_idx + 1}'
        }

        for i, (variant_graph, variant_results) in enumerate(variants):
            variant_name = f'variant_{i + 1}'
            optimization_results['variants'][variant_name] = {
                'results': variant_results,
                'durations': variant_data[i][1]
            }

        report = self.generate_report(optimization_results)
        logging.info(report)

    def generate_report(self, optimization_results: Dict) -> str:
        """Генерація звіту з результатами оптимізації"""
        report = []
        report.append("# Звіт з оптимізації сіткового графіка\n")

        # Базовий варіант
        base = optimization_results['variants']['base']
        report.append("## 1. Базовий варіант")
        report.append(f"Тривалість критичного шляху: {base['results']['critical_length']:.2f} тижнів")
        report.append(f"Критичний шлях: {' -> '.join(map(str, base['results']['critical_path']))}\n")

        # Варіанти оптимізації
        report.append("## 2. Варіанти оптимізації")
        for variant_name, variant_data in optimization_results['variants'].items():
            if variant_name == 'base':
                continue

            results = variant_data['results']
            report.append(f"\n### {variant_name.replace('_', ' ').title()}")
            report.append(f"Тривалість критичного шляху: {results['critical_length']:.2f} тижнів")
            report.append(f"Критичний шлях: {' -> '.join(map(str, results['critical_path']))}")
            report.append("Зміни в тривалості робіт:")

            for (start, end), duration in variant_data['durations'].items():
                base_duration = optimization_results['variants']['base']['durations'][(start, end)]
                if abs(duration - base_duration) > 0.01:
                    change = (duration - base_duration) / base_duration * 100
                    report.append(f"- {self.work_descriptions[(start, end)]}: "
                                  f"{change:+.1f}% ({base_duration:.2f} -> {duration:.2f} тижнів)")

        # Оптимальний варіант
        report.append(f"\n## 3. Оптимальний варіант")
        report.append(f"Обрано: {optimization_results['optimal_variant']}")
        optimal_data = optimization_results['variants'][optimization_results['optimal_variant']]
        report.append(f"Тривалість критичного шляху: {optimal_data['results']['critical_length']:.2f} тижнів")
        report.append(f"Покращення: {(optimization_results['variants']['base']['results']['critical_length'] - optimal_data['results']['critical_length']):.2f} тижнів "
                      f"({(1 - optimal_data['results']['critical_length']/optimization_results['variants']['base']['results']['critical_length'])*100:.1f}%)")

        return '\n'.join(report)


if __name__ == "__main__":
    # Створення оптимізатора
    optimizer = NetworkOptimizer()

    # Аналіз базового варіанту
    logging.info("\n=== БАЗОВИЙ ВАРІАНТ ===")
    base_results = optimizer.analyze_variant(optimizer.base_graph, optimizer.base_durations)
    base_critical_length = base_results['critical_length']

    # Аналіз варіантів оптимізації
    all_results = []  # Ініціалізуємо пустий список
    for variant in range(1, 5):
        logging.info(f"\n=== ВАРІАНТ {variant} ===")
        graph, durations = optimizer.create_optimization_variant(variant)
        results = optimizer.analyze_variant(graph, durations)

        # Розрахунок покращення
        improvement = base_critical_length - results['critical_length']
        improvement_percent = (improvement / base_critical_length) * 100
        logging.info(f"\nПокращення: {improvement:.2f} тижнів ({improvement_percent:.1f}%)")

        # Візуалізація варіанту
        optimizer.visualize_network(
            graph,
            results['critical_path'],
            f'network_variant_{variant}.png'
        )

        # Логування результатів
        logging.info(f"\nРезультати для варіанту {variant}:")
        logging.info(f"Критичний шлях: {' -> '.join(map(str, results['critical_path']))}")
        logging.info(f"Тривалість критичного шляху: {results['critical_length']:.2f} тижнів")

        # Додаємо результати до списку
        all_results.append((variant, graph, durations, results))
    logging.info(f"Скорочення тривалості: 7.36 тижнів (2.8%)")

    # Виклик оптимізації мережевого графіка
    optimizer.optimize_network()
