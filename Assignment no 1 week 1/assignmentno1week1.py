print("Question no1")
celsius = float(input("Enter temperature in Celsius: "))
 
fahrenheit = (celsius * 9/5) + 32
 
if fahrenheit > 100:
    print("Its very hot outside:", fahrenheit, "F")
elif fahrenheit > 60:
    print("Its warm outside:", fahrenheit, "F")
elif fahrenheit > 32:
    print("Its cold outside:", fahrenheit, "F")
else:
    print("Its freezing outside:", fahrenheit, "F")

print("Question no2")

length = float(input("Enter length of rectangle: "))
width = float(input("Enter width of rectangle: "))
 
area = length * width
perimeter = 2 * (length + width)
 
if area > 500:
    print("Its a very large rectangle")
elif area > 100:
    print("Its a medium rectangle")
else:
    print("Its a small rectangle")
 
print("Area:", area)
print("Perimeter:", perimeter)

print("Question no3")

P = float(input("Enter Principal amount: "))
R = float(input("Enter Rate of interest: "))
T = float(input("Enter Time in years: "))
CI = P * (1 + R/100)**T - P
total_amount = P + CI
if CI > 50000:
    print("Great investment! You earned a lot")
elif CI > 10000:
    print("Good investment!")
else:
    print("You earned a small amount")
print("Principal Amount:", P)
print("Compound Interest:", round(CI, 2))
print("Total Amount:", round(total_amount, 2))
 
 