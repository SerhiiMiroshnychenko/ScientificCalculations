import matplotlib
import networkx as nx
import matplotlib.pyplot as plt
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
matplotlib.use('Agg')

# Реєстрація українського шрифту
pdfmetrics.registerFont(TTFont('Arial', 'C:\\Windows\\Fonts\\arial.ttf'))

class ProjectCalculator:
    def __init__(self):
        # Вхідні дані частини 1
        self.Tb = 40.5  # базова трудомісткість
        self.Ks = 1.43  # коефіцієнт складності
        self.Kn = 1.55  # коефіцієнт новизни
        self.Kzv = 0.28 # коефіцієнт зворотного впливу
        self.Ku = 0.82  # коефіцієнт уніфікації
        self.Kd = 1.16  # коефіцієнт додаткових витрат
        
        # Дані для розрахунку тривалості
        self.K1 = 1.11
        self.Tzm = 8.12
        self.m = 2
        self.Dr = 257
        self.Dk = 365
        self.K2 = 1.15

        # Опис подій
        self.events = {
            0: 'Технічне завдання отримано',
            1: 'Технічні умови розроблено',
            2: 'Загальне компонування завершено',
            3: 'Проєкт електричного блоку готовий',
            4: 'Проєкт механічного блоку готовий',
            5: 'Замовлення розміщено',
            6: 'Всі компоненти готові',
            7: 'Технічна інформація готова',
            8: 'Виріб складено',
            9: 'Випробування завершено, документація готова'
        }

        # Дані про роботи
        self.works = {
            '0,1': {'h': 5, 'Rij': 2, 'desc': 'Розробка технічних умов'},
            '1,2': {'h': 8, 'Rij': 3, 'desc': 'Загальне компонування виробу'},
            '1,7': {'h': 6, 'Rij': 2, 'desc': 'Видача завдання на документацію'},
            '2,3': {'h': 12, 'Rij': 4, 'desc': 'Проєктування електричного блоку'},
            '2,4': {'h': 12, 'Rij': 4, 'desc': 'Проєктування механічного блоку'},
            '2,5': {'h': 8, 'Rij': 2, 'desc': 'Оформлення замовлень на компоненти'},
            '3,6': {'h': 10, 'Rij': 3, 'desc': 'Виготовлення електричної частини'},
            '4,6': {'h': 10, 'Rij': 3, 'desc': 'Виготовлення механічної частини'},
            '5,6': {'h': 7, 'Rij': 2, 'desc': 'Отримання компонентів'},
            '6,7': {'h': 6, 'Rij': 2, 'desc': 'Підготовка технічної інформації'},
            '7,8': {'h': 8, 'Rij': 3, 'desc': 'Складання виробу'},
            '7,9': {'h': 4, 'Rij': 2, 'desc': 'Розробка експлуатаційної документації'},
            '8,9': {'h': 4, 'Rij': 2, 'desc': 'Контрольні випробування'}
        }


    def calculate_total_labor(self):
        """Розрахунок загальної трудомісткості"""
        Tz = self.Tb * self.Ks * self.Kn * (1 - self.Kzv * self.Ku) * self.Kd
        return Tz

    def calculate_work_duration(self):
        """Розрахунок тривалості робіт"""
        # Загальна трудомісткість
        Tz = self.calculate_total_labor()

        # Розрахунок для всіх робіт
        for work_id, work in self.works.items():
            Tpij = (Tz/100) * work['h']
            tpij = (Tpij * 1000/(work['Rij'] * self.m * self.Tzm)) * self.K1 * self.K2
            tij = tpij * (self.Dk/self.Dr)
            Tij = tij/7
            
            self.works[work_id].update({
                'Tpij': round(Tpij, 2),
                'tpij': round(tpij, 2),
                'tij': round(tij, 2),
                'Tij': round(Tij, 2)
            })

    def calculate_critical_path(self):
        """Розрахунок критичного шляху та його параметрів"""
        # Створюємо граф для аналізу
        G = nx.DiGraph()
        
        # Додаємо ребра з вагами (тривалістю)
        for work_code, work_data in self.works.items():
            start, end = map(int, work_code.split(','))
            # Розрахунок тривалості роботи
            Tz = self.calculate_total_labor()
            h = work_data['h']
            Rij = work_data['Rij']
            
            # Трудомісткість роботи
            Tpij = (Tz/100) * h
            
            # Тривалість в робочих днях
            tpij = (Tpij * 1000)/(Rij * self.m * self.Tzm) * self.K1 * self.K2
            
            # Тривалість в календарних днях
            tij = tpij * (self.Dk/self.Dr)
            
            G.add_edge(start, end, weight=tij, work_code=work_code)

        # Знаходимо критичний шлях (найдовший шлях від 0 до 9)
        critical_path = nx.dag_longest_path(G)
        critical_path_length = nx.dag_longest_path_length(G)
        
        # Формуємо список робіт критичного шляху
        critical_works = []
        for i in range(len(critical_path)-1):
            start = critical_path[i]
            end = critical_path[i+1]
            work_code = f"{start},{end}"
            work_data = self.works[work_code]
            critical_works.append({
                'code': work_code,
                'desc': work_data['desc'],
                'duration': G[start][end]['weight']
            })
        
        return critical_path, critical_path_length, critical_works


    def create_network_graph(self):
        """Створення сіткового графіка"""
        plt.figure(figsize=(15, 10))
        
        # Створення направленого графа
        G = nx.DiGraph()
        
        # Додавання вузлів
        nodes = set()
        for work in self.works.keys():
            start, end = work.split(',')
            nodes.add(int(start))
            nodes.add(int(end))
        
        # Сортування вузлів для правильного розташування
        nodes = sorted(list(nodes))
        
        # Створення позицій вузлів
        pos = {}
        levels = {0: [0]}  # Початковий вузол
        
        # Визначення рівнів для кожного вузла
        current_level = [0]
        visited = {0}
        while current_level:
            next_level = []
            for node in current_level:
                for work_id in self.works:
                    start, end = map(int, work_id.split(','))
                    if start == node and end not in visited:
                        next_level.append(end)
                        visited.add(end)
            if next_level:
                levels[len(levels)] = sorted(next_level)
            current_level = next_level
        
        # Встановлення позицій вузлів
        max_nodes_in_level = max(len(level) for level in levels.values())
        for level_num, level_nodes in levels.items():
            level_width = len(level_nodes)
            for i, node in enumerate(level_nodes):
                x = level_num
                y = (max_nodes_in_level - level_width)/2 + i
                pos[node] = (x, y)
        
        # Додавання вузлів та ребер
        for node in nodes:
            G.add_node(node)
        
        # Розрахунок тривалості робіт перед створенням графа
        self.calculate_work_duration()
        
        for work_id, work in self.works.items():
            start, end = map(int, work_id.split(','))
            G.add_edge(start, end, weight=work['Tij'], desc=work['desc'])
        
        # Знаходимо критичний шлях
        critical_path = nx.dag_longest_path(G)
        critical_edges = list(zip(critical_path[:-1], critical_path[1:]))
        
        # Малювання графа
        plt.clf()
        
        # Малювання ребер
        # Спочатку малюємо всі ребра сірим
        nx.draw_networkx_edges(G, pos, edge_color='gray', arrows=True, 
                             arrowsize=20, width=1.5)
        # Потім малюємо критичний шлях червоним
        nx.draw_networkx_edges(G, pos, edgelist=critical_edges, edge_color='red', arrows=True,
                             arrowsize=20, width=2)
        
        # Малювання вузлів
        nx.draw_networkx_nodes(G, pos, node_color='lightblue', 
                             node_size=1000, alpha=0.6)
        
        # Додавання міток вузлів
        nx.draw_networkx_labels(G, pos, font_size=12, font_family='Arial')
        
        # Додавання міток ребер з описом робіт та тривалістю
        edge_labels = {}
        for (u, v, d) in G.edges(data=True):
            edge_labels[(u, v)] = f"{d['desc']}\n({d['weight']:.1f} тижнів)"
        
        # Налаштування позицій міток ребер
        pos_attrs = {}
        for edge in G.edges():
            start, end = edge
            # Розрахунок позиції мітки відносно координат вузлів
            start_pos = pos[start]
            end_pos = pos[end]
            pos_attrs[edge] = (
                (start_pos[0] + end_pos[0]) / 2,
                (start_pos[1] + end_pos[1]) / 2 + 0.1
            )
        
        nx.draw_networkx_edge_labels(G, pos, edge_labels, font_size=8, 
                                   bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
                                   label_pos=0.5)
        
        plt.title("Сітковий графік проєкту\n(червоним позначено критичний шлях)", fontsize=16, pad=20)
        plt.axis('off')
        plt.savefig('D:/network_graph.png', dpi=300, bbox_inches='tight')
        plt.close()


if __name__ == "__main__":
    # Створення екземпляру калькулятора та генерація звіту
    calculator = ProjectCalculator()
    calculator.create_network_graph()
