# 🐍 Python Tips & Tricks: Bootcamp Edition
*A curated list of modern Python patterns and productivity shortcuts.*

---

## 1. String Formatting & Basics
### f-Strings (The Modern Way)
Instead of `.format()`, use **f-strings** (Python 3.6+). They are faster and much easier to read.
```python
name = "Antigravity"
# Fast and readable
print(f"Hello, {name}!")

# Number formatting (2 decimal places)
price = 19.998
print(f"Price: ${price:.2f}") # Output: $20.00
```

---

## 2. Control Flow Shortcuts
### Ternary Operators (One-Line `if-else`)
Clean up simple conditional assignments.
```python
age = 20
# Modern way: <value_if_true> if <condition> else <value_if_false>
status = "Adult" if age >= 18 else "Minor"
```

### Loop with `enumerate()`
Instead of manually tracking a counter, use `enumerate()` to get the index and the item together.
```python
fruits = ["apple", "banana", "cherry"]
for index, fruit in enumerate(fruits):
    print(f"{index}: {fruit}")
```

---

## 3. Data Structure Power-Ups

### Tuples: Star Unpacking (`*`)
Use the asterisk to "catch" multiple items during unpacking.
```python
numbers = (1, 2, 3, 4, 5)
first, *middle, last = numbers
# first = 1, middle = [2, 3, 4], last = 5
```

### Sets: The Safety of `.discard()`
Always use `.discard()` instead of `.remove()` if you aren't 100% sure the item exists—it prevents your code from crashing.
```python
my_set = {1, 2, 3}
my_set.discard(10) # No error!
# my_set.remove(10) # <-- This would CRASH your program
```

### Dictionaries: Modern Merging
In Python 3.9+, you can use the pipe operator (`|`) to merge dictionaries instantly.
```python
dict1 = {"a": 1, "b": 2}
dict2 = {"b": 3, "c": 4}

# The Pipe Operator (Cleanest)
merged = dict1 | dict2 
# Output: {'a': 1, 'b': 3, 'c': 4} (Last one wins on duplicates!)
```

### List Comprehensions
The most "Pythonic" way to create new lists from old ones.
```python
nums = [1, 2, 3, 4]
# [expression for item in iterable if condition]
squares = [x**2 for x in nums if x % 2 == 0] 
# Output: [4, 16]
```

---

## 4. Modern Function Patterns
### `*args` (The Positional Bucket)
Catches any number of "unnamed" arguments and puts them into a **Tuple**.
```python
def add_all(*args):
    return sum(args)

print(add_all(1, 2, 3, 4)) # Output: 10
```

### `**kwargs` (The Keyword Bucket)
Catches all "named" arguments (key=value) and puts them into a **Dictionary**.
```python
def print_info(**kwargs):
    for key, value in kwargs.items():
        print(f"{key}: {value}")

print_info(name="Krish", age=32)
# Output:
# name: Krish
# age: 32
```

### The Gold Rule: The Order
If you use both in one function, they **must** be in this order:
1. `Normal arguments`
2. `*args`
3. `**kwargs`

---

### The "Two Sides" of `*` and `**`: Packing vs Unpacking
Both operators act like 2-way doors depending on where you use them.

| Operator | Context | Action | Name |
| :--- | :--- | :--- | :--- |
| `*` | **Parameters** (`def f(*args)`) | Loose $\rightarrow$ **Tuple** | **Packing** |
| `*` | **Call** (`f(*list)`) | Sequence $\rightarrow$ **Loose** | **Unpacking** |
| `**` | **Parameters** (`def f(**kw)`) | Named $\rightarrow$ **Dict** | **Packing** |
| `**` | **Call** (`f(**dict)`) | Dict $\rightarrow$ **Named** | **Unpacking** |

**Visual Example for `**`:**
```python
# 1. PACKING (Definition)
def welcome(**kwargs):
    # kwargs is now a Dict: {"user": "Krish", "mode": "admin"}
    print(kwargs)

# 2. UNPACKING (Call)
settings = {"user": "Krish", "mode": "admin"}
welcome(**settings) 
# Behaves like: welcome(user="Krish", mode="admin")
```

---

> [!NOTE]
> **For C/C++/Go Developers:** In Python, `*` is NOT a pointer. Python handles memory for you automatically. Here, `*` is the **"Unpack/Pack"** operator used for moving items in and out of collections.

---
> [!TIP]
> **Pro Tip:** When in doubt, use `dir(object)` or `help(object)` in your terminal to quickly see what methods are available for any Python data type!

---

## 5. Functional Tools: Map vs Filter
Understanding the difference is key to mastering Python data handling.

| Function | Action | Transformation | Output Length |
| :--- | :--- | :--- | :--- |
| `map(func, list)` | **Transformer** | **Yes** (Changes values) | Same as Input |
| `filter(func, list)` | **Selector** | **No** (Keeps logic only) | Same or Smaller |

### In Simple Terms:
*   **`map`** is a **Factory**: It takes raw material and turns it into a product. (e.g., `[1, 2]` --> `[10, 20]`)
*   **`filter`** is a **Security Guard**: It checks ID and decides who gets in. (e.g., `[1, 2, 3]` --> `[2, 3]`)

---

## 6. Lambda Functions (Anonymous Functions)
A `lambda` is just a **shortcut** to write a function in one line. It is not a transformation tool itself; it is the *rule* you give to tools like `map` and `filter`.

**Syntax:** `lambda inputs: output`

> [!NOTE]
> **Key Concept:** `lambda` is the **Fuel** (the logic). `map` and `filter` are the **Engines** (the processors). The engines need fuel to run!

### Comparison to `def`
```python
# The Long Way (def)
def square(num):
    return num ** 2

# The Short Way (lambda)
square = lambda num: num ** 2
```

### The Power Combo: Lambda + Map/Filter
You rarely use `lambda` alone. You usually plug it *inside* `map` or `filter` so you don't have to write a throwaway `def`.

```python
nums = [1, 2, 3, 4]

# Using map with lambda (Square everything)
squares = list(map(lambda x: x**2, nums)) 
# Output: [1, 4, 9, 16]

# Using filter with lambda (Keep evens)
evens = list(filter(lambda x: x % 2 == 0, nums))
# Output: [2, 4]
```

---

## 7. Standard Library Essentials
Python comes with "batteries included". Here are the gems you need to know.

### Data Handling (`json`, `csv`)
- **`json`**: The internet's language. Use `json.dumps(data)` to turn a dict into a string, and `json.loads(string)` to turn it back.
- **`csv`**: Read and write spreadsheet-like data easily.

### Files & Paths (`os`, `shutil`)
- **`os.getcwd()`**: "Where am I currently?"
- **`os.mkdir('folder')`**: Creates a new folder.
- **`shutil.copyfile(src, dst)`**: The easiest way to copy a file.

### Time Travel (`datetime`, `time`)
- **`datetime.now()`**: Get the current timestamp.
- **`timedelta(days=1)`**: Do math with dates (e.g., `now - timedelta(days=1)` is yesterday).
- **`time.sleep(2)`**: Pause your script for 2 seconds.

### Math & Randomness
- **`math.pi`**: $\pi$ (3.1415...)
- **`random.choice(['a', 'b'])`**: Pick a random winner.
- **`random.randint(1, 10)`**: Pick a random number in range.

### Regular Expressions (`re`)
Find patterns in text (emails, phone numbers).
```python
import re
match = re.search(r'\d+', "There are 123 apples")
print(match.group()) # Output: 123
```

---

## 8. Exception Handling Patterns
### The Full try-except-else-finally Structure
Use all four blocks for robust error handling. **finally** always executes (great for cleanup).
```python
try:
    num = int(input("Enter a number: "))
    result = 10 / num
except ValueError:
    print("That's not a valid number!")
except ZeroDivisionError:
    print("You can't divide by zero!")
except Exception as ex:  # Catch-all (use sparingly)
    print(f"Unexpected error: {ex}")
else:
    print(f"The result is {result}")  # Runs only if no exception
finally:
    print("Execution complete.")  # Always runs (cleanup code)
```

> [!WARNING]
> **Order Matters!** Always catch specific exceptions before generic `Exception`. Python matches top-to-bottom.

### Catching Exception Details
Capture the error message using `as` for debugging.
```python
try:
    a = undefined_variable
except NameError as ex:
    print(ex)  # Output: name 'undefined_variable' is not defined
```

### The `finally` Block for Resource Cleanup
Perfect for closing files, database connections, or releasing locks.
```python
try:
    file = open('data.txt', 'r')
    content = file.read()
except FileNotFoundError:
    print("File not found!")
finally:
    if 'file' in locals() and not file.closed:
        file.close()
        print("File closed.")
```

---

## 9. OOP (Object-Oriented Programming) Essentials

### The `self` Parameter
`self` refers to the **instance** of the class. It's automatically passed when you call a method on an object.
```python
class Dog:
    def __init__(self, name, age):
        self.name = name  # Instance variable
        self.age = age
    
    def bark(self):
        print(f"{self.name} says woof!")

dog1 = Dog("Buddy", 3)
dog1.bark()  # self is automatically dog1
```

### Inheritance: `super()` vs Direct Parent Call
Use `super()` for single inheritance, direct call for multiple inheritance.
```python
# Single Inheritance - use super()
class Tesla(Car):
    def __init__(self, windows, doors, enginetype, is_selfdriving):
        super().__init__(windows, doors, enginetype)  # Clean!
        self.is_selfdriving = is_selfdriving

# Multiple Inheritance - call parents directly
class Dog(Animal, Pet):
    def __init__(self, name, owner):
        Animal.__init__(self, name)  # Direct call
        Pet.__init__(self, owner)    # Direct call
```

### Access Modifiers: Public, Protected, Private
Python uses naming conventions (not true access control like Java).

| Modifier | Syntax | Access Level |
|----------|--------|--------------|
| **Public** | `name` | Accessible everywhere |
| **Protected** | `_name` | "Don't touch" (convention only) |
| **Private** | `__name` | Name mangled: `_ClassName__name` |

```python
class Person:
    def __init__(self, name, age):
        self.name = name        # Public
        self._age = age         # Protected (convention)
        self.__secret = "hidden" # Private (name mangled)

p = Person("Krish", 34)
print(p.name)        # ✓ Works
print(p._age)        # ✓ Works (but don't!)
# print(p.__secret)  # ✗ AttributeError!
print(p._Person__secret)  # ✓ Access via mangled name
```

### Getters and Setters (The Pythonic Way)
Use methods to control access and validate data.
```python
class Person:
    def __init__(self, name, age):
        self.__name = name
        self.__age = age
    
    # Getter
    def get_name(self):
        return self.__name
    
    # Setter with validation
    def set_age(self, age):
        if age > 0:
            self.__age = age
        else:
            print("Age must be positive!")

p = Person("Krish", 34)
print(p.get_name())   # Access via getter
p.set_age(-5)         # Validation prevents invalid data
```

### Magic Methods (Dunder Methods)
Special methods with double underscores that customize object behavior.

| Method | Purpose |
|--------|---------|
| `__init__` | Constructor (initialize object) |
| `__str__` | Human-readable string (print(obj)) |
| `__repr__` | Official representation (debugging) |
| `__len__` | Length of object (len(obj)) |
| `__eq__` | Equality comparison (==) |

```python
class Person:
    def __init__(self, name, age):
        self.name = name
        self.age = age
    
    def __str__(self):
        return f"{self.name}, {self.age} years old"  # For users
    
    def __repr__(self):
        return f"Person(name='{self.name}', age={self.age})"  # For devs

p = Person("Krish", 34)
print(str(p))   # Output: Krish, 34 years old
print(repr(p))  # Output: Person(name='Krish', age=34)
```

### Polymorphism: Same Interface, Different Behavior
Different classes implement the same method differently.
```python
class Animal:
    def speak(self):
        return "Some sound"

class Dog(Animal):
    def speak(self):  # Override
        return "Woof!"

class Cat(Animal):
    def speak(self):  # Override
        return "Meow!"

# Same function works with different types
def animal_sound(animal):
    print(animal.speak())

animal_sound(Dog())  # Output: Woof!
animal_sound(Cat())  # Output: Meow!
```

### Abstract Base Classes (ABC)
Force subclasses to implement specific methods using `@abstractmethod`.
```python
from abc import ABC, abstractmethod

class Vehicle(ABC):
    @abstractmethod
    def start_engine(self):
        pass  # Must be implemented by subclasses!

class Car(Vehicle):
    def start_engine(self):
        return "Car engine started"

# vehicle = Vehicle()  # ✗ Can't instantiate abstract class!
car = Car()          # ✓ Works if all abstract methods implemented
```

---

## 10. Advanced Python Concepts

### Iterators: The Manual Way to Loop
Iterators provide sequential access without loading everything into memory.
```python
my_list = [1, 2, 3, 4, 5]
iterator = iter(my_list)  # Create iterator

print(next(iterator))  # Output: 1
print(next(iterator))  # Output: 2
# ... until StopIteration exception

# Safe iteration with try-except
try:
    while True:
        print(next(iterator))
except StopIteration:
    print("No more items!")
```

### Generators: Memory-Efficient Iterators
Use `yield` to create values on-demand (lazy evaluation). Great for large datasets!
```python
def square_generator(n):
    for i in range(n):
        yield i ** 2  # Returns value, pauses here

# Usage
for val in square_generator(3):
    print(val)  # Output: 0, 1, 4 (one at a time)

# Generator expression (like list comprehension but lazy)
squares = (x**2 for x in range(1000000))  # No memory explosion!
```

> [!TIP]
> **When to use generators:** Processing large files, infinite sequences, or when you only need one item at a time.

### Generator for Large File Reading
Perfect for processing files line-by-line without loading entire file.
```python
def read_large_file(file_path):
    with open(file_path, 'r') as file:
        for line in file:
            yield line.strip()  # Process one line at a time

# Memory-efficient file reading
for line in read_large_file('huge_file.txt'):
    process(line)  # Only one line in memory at a time
```

### Decorators: Function Modifiers
Decorators wrap functions to add behavior without changing their code.

**Basic Decorator:**
```python
def my_decorator(func):
    def wrapper():
        print("Before function")
        func()
        print("After function")
    return wrapper

@my_decorator  # Syntactic sugar
def say_hello():
    print("Hello!")

say_hello()
# Output:
# Before function
# Hello!
# After function
```

**Decorator with Arguments:**
```python
def repeat(n):
    def decorator(func):
        def wrapper(*args, **kwargs):
            for _ in range(n):
                func(*args, **kwargs)
        return wrapper
    return decorator

@repeat(3)
def greet():
    print("Hi!")

greet()  # Prints "Hi!" three times
```

### Understanding Closures (Decorator Foundation)
A closure is a function that remembers variables from its enclosing scope.
```python
def outer_function(msg):
    def inner_function():
        print(msg)  # Remembers 'msg' even after outer exits
    return inner_function

my_func = outer_function("Hello!")
my_func()  # Output: Hello! (closure remembers msg)
```

---

> [!NOTE]
> **Key Takeaway:** Advanced Python features (generators, decorators) help you write more efficient, modular, and Pythonic code. Master these and you'll write production-quality Python!

---

*Happy Coding! 🐍*

