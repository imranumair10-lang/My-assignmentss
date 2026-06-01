my_dict = {"Karachi": 1, "Lahore": 2, "Islamabad": 3, "Quetta": 4, "Peshawar": 5}
value_to_check = 3
if value_to_check in my_dict.values():
    print("Value", value_to_check, "exists in the dictionary")
else:
    print("Value", value_to_check, "does not exist in the dictionary")


my_dict = {"Karachi": 50, "Lahore": 30, "Islamabad": 20, "Quetta": 10, "Peshawar": 40}
min_key = min(my_dict, key=my_dict.get)
print("Key with minimum value:", min_key)


my_dict = {"Karachi": 1, "Lahore": 2, "Islamabad": 3, "Quetta": 4, "Peshawar": 5, "Multan": 6}
keys_to_delete = ["Lahore", "Quetta"]
for key in keys_to_delete:
    del my_dict[key]
print("After Deleting Keys:", my_dict)