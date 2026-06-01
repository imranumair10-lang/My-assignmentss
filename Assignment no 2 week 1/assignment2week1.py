story = "Ahmed went to Karachi with 3 friends. He visited Lahore and Islamabad. He bought 2 books and 5 mangoes from the market in Multan."

noun_dict = {
    "person": "Ahmed",
    "city1": "Karachi",
    "group": "friends",
    "city2": "Lahore",
    "city3": "Islamabad",
    "item1": "books",
    "item2": "mangoes",
    "place": "market",
    "city4": "Multan",
    "numbers": {"friends": 3, "books": 2, "mangoes": 5}
}

print("Nouns in story:", noun_dict)
print("Last Element (nested dict):", noun_dict["numbers"])