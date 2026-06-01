i = 1
while i <= 10:
    print(i)
    i += 1


n = int(input("Enter a number: "))
for i in range(1, n + 1):
    if i % 2 == 0:
        print(i)


n = int(input("Enter a number: "))
for i in range(1, n + 1):
    if i % 2 != 0:
        print(i)


n = int(input("Enter a number: "))
for i in range(2, n + 1):
    is_prime = True
    for j in range(2, i):
        if i % j == 0:
            is_prime = False
            break
    if is_prime:
        print(i)


n = int(input("Enter a number: "))
for i in range(1, 11):
    print(n, "x", i, "=", n * i)