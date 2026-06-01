my_list = [1, 2, 3, 4, 5]
reversed_list = my_list[::-1]
print("Original List:", my_list)
print("Reversed List:", reversed_list)


my_list = [1, 2, 3, 4, 5]
squared_list = [x ** 2 for x in my_list]
print("Original List:", my_list)
print("Squared List:", squared_list)


my_list = ["Karachi", "", "Lahore", "", "Islamabad", ""]
cleaned_list = [x for x in my_list if x != ""]
print("Original List:", my_list)
print("After Removing Empty Strings:", cleaned_list)


my_list = ["Karachi", "Lahore", "Quetta", "Peshawar"]
specified_item = "Lahore"
new_item = "Multan"
index = my_list.index(specified_item)
my_list.insert(index + 1, new_item)
print("After Adding", new_item, "after", specified_item, ":", my_list)


my_list = ["Karachi", "Lahore", "Quetta", "Lahore", "Peshawar"]
old_value = "Lahore"
new_value = "Faisalabad"
my_list = [new_value if x == old_value else x for x in my_list]
print("After Replacing", old_value, "with", new_value, ":", my_list)