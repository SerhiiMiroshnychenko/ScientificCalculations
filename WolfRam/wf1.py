from wolframclient.evaluation import WolframLanguageSession
from wolframclient.language import wl, wlexpr

kernel_path = r"D:\PROGRAMs\WolfRam\Wolfram\WolframKernel.exe"

session = WolframLanguageSession(kernel_path)

result = session.evaluate(wlexpr('Range[5]'))
print('List:', result)

# Отримання геопозиції
geo_position = session.evaluate(wlexpr('GeoPosition[Here]'))

# Отримання координат із GeoPosition
coordinates = geo_position.args[0]  # Перший аргумент містить список координат
latitude, longitude = coordinates[0], coordinates[1]

# Форматований вивід
print(f"Here: {latitude:.6f}° N, {longitude:.6f}° E")


# Отримання кінематичної в'язкості повітря при 35°C
temp = wl.Quantity(35, "Celsius")
params = wl.List(wl.Rule("Temperature", temp))  # Створюємо список правил

# Створюємо вираз для кінематичної в'язкості як dynamicViscosity / density
kinematic_expr = wl.N(
    wl.QuantityMagnitude(
        wl.Divide(
            wl.ThermodynamicData("Air", "DynamicViscosity", params),
            wl.ThermodynamicData("Air", "Density", params)
        )
    )
)

# Обчислюємо весь вираз однією командою
kinematicViscosity = session.evaluate(kinematic_expr)
print("Kinematic viscosity of air at 35°C:", kinematicViscosity)

solveX = session.evaluate(wlexpr('Solve[2x^2 - 7x + 6 == 0, x]'))
root1 = solveX[0][0].args[1]
print(f"{root1 = }")
root2 = solveX[1][0].args[1]
print(f"{root2 = }")
print(f'Roots: {root1}; {root2}')




session.terminate()
