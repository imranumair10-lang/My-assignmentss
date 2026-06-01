def first_middle_last(s):
    first = s[0]
    middle = s[len(s) // 2]
    last = s[-1]
    return first + middle + last

string = "Hello Umair"
result = first_middle_last(string)
string = "Hello Umair"
char_count = {}
for char in string:
    if char in char_count:
        char_count[char] += 1
    else:
        char_count[char] = 1

print("\nOriginal String:", string)
for char, count in char_count.items():
    print(f"  '{char}' : {count}")

string = "Hello Umair"
reversed_string = string[::-1]
print("\nOriginal String:", string)

string = "I am learning Python programming"
split_result = string.split(" ")

import re

string = "Hello! My name is Umair Imran? I'm fine."
cleaned = re.sub(r'[^a-zA-Z0-9\s]', '', string)
