story = "Ahmed went to Karachi with 3 friends. He visited Lahore and Islamabad. He bought 2 books and 5 mangoes from the market in Multan."

nouns = ["Ahmed", "Karachi", "friends", "Lahore", "Islamabad", "books", "mangoes", "market", "Multan"]
numbers = [3, 2, 5]

noun_list = nouns + [numbers]
print("Assignment 2:", noun_list)


noun_list_b = nouns + [numbers]
print("Assignment 2b:", noun_list_b)


noun_tuple = tuple(nouns) + (tuple(numbers),)
print("Assignment 3:", noun_tuple)


noun_tuple_b = tuple(nouns) + (tuple(numbers),)
print("Assignment 3b:", noun_tuple_b)


noun_set = set(nouns)
numbers_set = frozenset(numbers)
print("Assignment 4:", noun_set)
print("Numbers in story:", numbers_set)