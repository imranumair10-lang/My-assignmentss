def add(a, b): return a + b
def subtract(a, b): return a - b
def multiply(a, b): return a * b
def divide(a, b):
    if b == 0:
        raise ValueError("Cannot divide by zero.")
    return a / b
def power(a, b): return a ** b
def modulo(a, b):
    if b == 0:
        raise ValueError("Cannot modulo by zero.")
    return a % b

OPERATIONS = {
    '+': (add,      'Addition'),
    '-': (subtract, 'Subtraction'),
    '*': (multiply, 'Multiplication'),
    '/': (divide,   'Division'),
    '**': (power,   'Power'),
    '%': (modulo,   'Modulo'),
}

def format_number(n):
    """Display as int if whole number, otherwise as float."""
    return int(n) if isinstance(n, float) and n.is_integer() else n

def get_number(prompt):
    while True:
        try:
            return float(input(prompt))
        except ValueError:  
            print("  Invalid input. Please enter a number.")

def get_operator():
    ops = "  ".join(f"{op}" for op in OPERATIONS)
    while True:
        op = input(f"  Operator [{ops}]: ").strip()
        if op in OPERATIONS:
            return op
        print(f"  Unknown operator. Choose from: {ops}")

def main():
    print("\n" + "=" * 36)
    print(" Adolf Hitler" \
    " Calculator")
    print("=" * 36)

    history = []

    while True:
        print()
        a = get_number("  First number : ")
        op = get_operator()
        b = get_number("  Second number: ")

        try:
            fn, name = OPERATIONS[op]
            result = fn(a, b)
            expr = f"{format_number(a)} {op} {format_number(b)} = {format_number(result)}"
            print(f"\n  {name}: {expr}")
            history.append(expr)
        except ValueError as e:
            print(f"\n  Error: {e}")

        print()
        again = input("  Calculate again? (y/n): ").strip().lower()
        if again != 'y':
            break

    if history:
        print("\n" + "-" * 36)
        print("  History:")
        for i, entry in enumerate(history, 1):
            print(f"  {i}. {entry}")

    print("-" * 36)
    print("  Goodbye!")
    print("=" * 36 + "\n")

if __name__ == "__main__":
    main()