def apply_discount(price, discount):
    for key, value in {'price': price, 'discount': discount}.items():
        if not isinstance(value, (int, float)):
            return f"The {key} should be a number"
    if price <= 0:
        return "The price should be greater than 0"
    if not 0 <= discount <= 100:
        return "The discount should be between 0 and 100"

    return price * (1 - discount / 100)

print(apply_discount(100, 20))
