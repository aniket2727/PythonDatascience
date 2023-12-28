

# Example: Print numbers from 1 to 5
for i in range(1, 6):
    print(i)



# Example: Print numbers from 1 to 5 using a while loop
i = 1
while i <= 5:
    print(i)
    i += 1


# Example: Nested loop to print a pattern
for i in range(5):
    for j in range(i + 1):
        print('*', end=' ')
    print()



# Example: Loop with break and continue statements
for i in range(1, 11):
    if i == 5:
        break  # Exit the loop when i is 5
    if i % 2 == 0:
        continue  # Skip even numbers
    print(i)



# Example: Loop through a string
text = "Python"
for char in text:
    print(char)



# Example: Loop through multiple lists simultaneously
names = ["Alice", "Bob", "Charlie"]
ages = [25, 30, 22]
for name, age in zip(names, ages):
    print(f"{name} is {age} years old")
