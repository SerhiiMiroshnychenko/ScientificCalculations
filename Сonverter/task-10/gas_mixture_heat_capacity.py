# Розрахунок середньої об'ємної теплоємності газової суміші
# Температура: 100°С

# Вхідні дані
components = {
    'CO': {'content': 87.4, 'heat_capacity': 1.3223},
    'CO2': {'content': 10.1, 'heat_capacity': 1.9479},
    'O2': {'content': 1.6, 'heat_capacity': 1.3665},
    'N2': {'content': 0.9, 'heat_capacity': 1.3144}
}

def calculate_mixture_heat_capacity(components):
    """
    Розрахунок середньої об'ємної теплоємності газової суміші
    
    Args:
        components (dict): Словник з компонентами та їх характеристиками
        
    Returns:
        float: Середня об'ємна теплоємність суміші
    """
    # Перевірка суми об'ємних часток
    total_content = sum(comp['content'] for comp in components.values())
    print(f"\n1. Сума об'ємних часток: {total_content:.1f}%")
    
    # Розрахунок внесків компонентів
    print("\n2. Внески компонентів у теплоємність суміші:")
    mixture_heat_capacity = 0
    for name, data in components.items():
        fraction = data['content'] / 100  # переведення у десятковий вигляд
        contribution = fraction * data['heat_capacity']
        mixture_heat_capacity += contribution
        print(f"{name}: {fraction:.4f} * {data['heat_capacity']:.4f} = {contribution:.4f} кДж/(м³·К)")
    
    return mixture_heat_capacity

# Розрахунок
print("Розрахунок середньої об'ємної теплоємності газової суміші")
print("\nВхідні дані:")
print("Компонент\tВміст, %\tТеплоємність, кДж/(м³·К)")
for name, data in components.items():
    print(f"{name}\t\t{data['content']}\t\t{data['heat_capacity']}")

# Обчислення середньої теплоємності
result = calculate_mixture_heat_capacity(components)

print(f"\n3. Середня об'ємна теплоємність суміші: {result:.4f} кДж/(м³·К)")
