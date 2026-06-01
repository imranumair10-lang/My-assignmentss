my_tuple = ("Karachi", "Lahore", "Islamabad", "Quetta", "Peshawar")
reversed_tuple = my_tuple[::-1]
print("Original Tuple:", my_tuple)
print("Reversed Tuple:", reversed_tuple)


my_tuple = (10, 20, 30, 40, 50)
print("Value 20:", my_tuple[1])


tuple1 = ("Karachi", "Lahore")
tuple2 = ("Quetta", "Peshawar")
tuple1, tuple2 = tuple2, tuple1
print("Tuple1 after swap:", tuple1)
print("Tuple2 after swap:", tuple2)