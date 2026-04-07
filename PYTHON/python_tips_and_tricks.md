# 🐍 Python Tips & Tricks: Bootcamp Edition
*A curated list of modern Python patterns and productivity shortcuts.*

---

## 1. String Formatting & Basics
### Core Concept: Format Specifier Anatomy
Most formatting power comes from this structure:

```text
{value:[fill][align][width][.precision][type]}
```

- `fill`: character used to fill empty width (example: `0`, `*`, `.`)
- `align`: `<` left, `>` right, `^` center
- `width`: minimum total characters to reserve
- `.precision`: depends on type (decimal places for floats, significant digits for `g`, max chars for strings)
- `type`: how value should be represented (`f`, `d`, `x`, etc.)

```python
name = "Ada"
score = 92.4567

print(f"|{name:>10}|")      # right align in width 10
print(f"|{score:08.2f}|")   # zero-fill, width 8, precision 2, float
```

### Width vs Precision (critical difference)
- `width` controls total field size (spacing/padding)
- `precision` controls numeric detail (or string truncation behavior)

```python
x = 3.14159
print(f"|{x:10.2f}|")  # width 10, precision 2 -> "|      3.14|"
print(f"|{x:.2f}|")    # no width, precision 2 -> "|3.14|"
```

### Type Specifiers
#### Floats: `f`, `e`, `g`
```python
x = 1234.56789
print(f"{x:f}")     # fixed point -> 1234.567890
print(f"{x:.2f}")   # fixed point 2 decimals -> 1234.57
print(f"{x:e}")     # scientific notation -> 1.234568e+03
print(f"{x:.3e}")   # scientific, 3 decimals -> 1.235e+03
print(f"{x:g}")     # general format (auto f/e)
print(f"{x:.5g}")   # general, 5 significant digits
```

#### Integers: `d`, `b`, `o`, `x`
```python
n = 42
print(f"{n:d}")  # decimal -> 42
print(f"{n:b}")  # binary  -> 101010
print(f"{n:o}")  # octal   -> 52
print(f"{n:x}")  # hex     -> 2a
print(f"{n:X}")  # HEX     -> 2A
```

### Precision in Depth
#### `.nf` behavior and rounding
```python
v = 19.998
print(f"{v:.2f}")  # 20.00
print(f"{v:.1f}")  # 20.0
```

Precision with float formatting rounds display output.

```python
print(f"{2.675:.2f}")  # may show 2.67 due to binary floating-point representation
```

#### Why precision does not work with integer `d`
```python
n = 7
# print(f"{n:.3d}")
# ValueError: Precision not allowed in integer format specifier
```

Use width + zero padding instead:

```python
print(f"{n:03d}")  # 007
```

### Width and Padding
#### Width usage
```python
n = 42
print(f"|{n:5d}|")   # "|   42|"
print(f"|{n:<5d}|")  # "|42   |"
```

#### Zero padding
```python
n = 7
print(f"{n:03d}")   # 007
print(f"{n:08d}")   # 00000007
```

#### Spacing vs zero padding
```python
n = 7
print(f"|{n:5d}|")   # spaces: |    7|
print(f"|{n:05d}|")  # zeros:  |00007|
```

### Alignment and Fill
#### Align operators `<`, `>`, `^`
```python
text = "Hi"
print(f"|{text:<8}|")  # left
print(f"|{text:>8}|")  # right
print(f"|{text:^8}|")  # center
```

#### Custom fill characters
```python
print(f"|{'A':*^7}|")  # |***A***|
print(f"|{'A':.>7}|")  # |......A|
print(f"|{'A':-_<7}|") # |A_-_-_-| (fill is '-' here, visible pattern depends on char)
```

### Sign Handling
```python
pos = 12
neg = -12

print(f"{pos:d}, {neg:d}")    # default: 12, -12
print(f"{pos:+d}, {neg:+d}")  # always show sign: +12, -12
print(f"{pos: d}, {neg: d}")  # leading space for positives: ' 12', '-12'
```

### Special Formatting
#### Thousand separators
```python
num = 1234567890
print(f"{num:,}")  # 1,234,567,890
print(f"{num:_}")  # 1_234_567_890
```

#### Percentage formatting
```python
ratio = 0.8732
print(f"{ratio:%}")    # 87.320000%
print(f"{ratio:.2%}")  # 87.32%
```

### Combinations (multiple specifiers together)
```python
price = 1234.5
print(f"{price:*>12,.2f}")  # ***1,234.50
print(f"{price:+012.2f}")   # +00001234.50
```

#### Real-world: aligned table output
```python
items = [
    ("Keyboard", 3, 2499.0),
    ("Mouse", 12, 799.5),
    ("Monitor", 2, 15499.99),
]

print(f"{'Item':<12} {'Qty':>5} {'Price':>12}")
print("-" * 31)
for name, qty, price in items:
    print(f"{name:<12} {qty:>5d} {price:>12,.2f}")
```

### Comparison of Methods
#### 1) f-strings (recommended)
```python
name = "Ava"
score = 91.234
print(f"{name} scored {score:.1f}")
```

#### 2) `format()`
```python
name = "Ava"
score = 91.234
print("{} scored {:.1f}".format(name, score))
```

#### 3) `%` formatting (legacy)
```python
name = "Ava"
score = 91.234
print("%s scored %.1f" % (name, score))
```

Use f-strings for modern code, `format()` for reusable templates, `%` only when maintaining old code.

### Common Mistakes
#### Mistake 1: `.3d`
```python
n = 7
# f"{n:.3d}"  # ValueError: precision not allowed for integer d
```

Correct:
```python
f"{n:03d}"  # 007
```

#### Mistake 2: Confusing width with precision
```python
x = 3.14159
print(f"{x:8f}")   # width 8, default precision 6
print(f"{x:.2f}")  # precision 2, no minimum width
```

#### Mistake 3: Misusing float formatting on integers
```python
n = 10
print(f"{n:.2f}")  # Works (10.00) but semantically this is float-style display
print(f"{n:d}")     # Integer-style display
```

### Advanced Notes
#### When to use `Decimal`
Use `Decimal` for money and exact base-10 arithmetic.

```python
from decimal import Decimal

price = Decimal("0.10") + Decimal("0.20")
print(price)           # 0.30 (exact)
print(f"{price:.2f}") # 0.30
```

#### When to use `round()`
Use `round()` when you need numeric rounding before further calculations, not only display formatting.

```python
x = 3.14159
y = round(x, 2)  # numeric value 3.14
print(y)
print(f"{x:.2f}")  # display formatting only
```

#### Edge cases and unexpected behavior
```python
print(round(2.5))      # 2 (banker's rounding: ties to even)
print(round(3.5))      # 4
print(f"{2.675:.2f}") # 2.67 (binary float representation effect)
```

### Summary Cheat Sheet
| Goal | Specifier | Example | Output |
| :--- | :--- | :--- | :--- |
| 2 decimal float | `.2f` | `f"{19.998:.2f}"` | `20.00` |
| Scientific notation | `.3e` | `f"{1234.5:.3e}"` | `1.234e+03` |
| General significant digits | `.5g` | `f"{1234.5:.5g}"` | `1234.5` |
| Integer width 5 | `5d` | `f"{42:5d}"` | `   42` |
| Zero padded integer | `03d` | `f"{7:03d}"` | `007` |
| Left / right / center | `<`, `>`, `^` | `f"{txt:^10}"` | centered text |
| Always show sign | `+` | `f"{12:+d}"` | `+12` |
| Space for positive sign | ` ` | `f"{12: d}"` | ` 12` |
| Thousand separators | `,` or `_` | `f"{1000000:,}"` | `1,000,000` |
| Percentage | `.2%` | `f"{0.8732:.2%}"` | `87.32%` |

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

### Control & Encapsulation: Getters, Setters, and `@property`
Getters and setters aren't just for "access"—they are for **Access Control**.

#### Why use them?
1.  **Validation (The Guard):** Ensure data is valid (e.g., age cannot be negative) before saving it.
2.  **Protection (Encapsulation):** Hide internal names like `__name` so users can't bypass your rules.
3.  **Derived Data:** Return data that isn't even stored (e.g., `full_name` from `first` and `last`).

> [!NOTE]
> **Name Mangling:** When you use `__name`, Python renames it internally to `_ClassName__name`. That’s why `p.__name` will throw an **AttributeError** if you try to access it directly from outside!

#### The "Modern" Pythonic Way: `@property`
In real Python projects, we don't usually write `get_age()`. Instead, we use the `@property` decorator to make methods *look* like normal attributes while keeping the logic behind them.

```python
class Person:
    def __init__(self, age):
        self.__age = age

    @property
    def age(self):  # This is the "Getter"
        return self.__age

    @age.setter
    def age(self, value):  # This is the "Setter"
        if value > 0:
            self.__age = value
        else:
            print("Age must be positive!")

p = Person(25)
print(p.age)   # No brackets needed! (Calls the getter)
p.age = -5     # Looks like simple assignment (Calls the setter)
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

### Polymorphism: The "Placeholder" Power
Polymorphism allows a single function or method to work with different types of objects, as long as they follow the same "interface" (contract).

> [!TIP]
> **The Placeholder Concept:** Think of the function parameter as a **placeholder**. It gets "filled" with a specific object at runtime. The behavior changes based on *which* object fills it, even though the function code stays the same.

```python
class Shape:
    def area(self):
        return "The area of the figure"

class Rectangle(Shape):
    def __init__(self, width, height):
        self.width = width
        self.height = height
    def area(self):
        return self.width * self.height

class Circle(Shape):
    def __init__(self, radius):
        self.radius = radius
    def area(self):
        return 3.14 * self.radius * self.radius

# 'shape' acts as a placeholder that gets filled when the function is called
def print_area(shape):
    print(f"The area is {shape.area()}")

rectangle = Rectangle(4, 5)
circle = Circle(3)

print_area(rectangle) # Placeholder filled with Rectangle
print_area(circle)    # Placeholder filled with Circle
```

### Abstract Base Classes (ABC): The "Official Rulebook"
ABCs and `@abstractmethod` make your polymorphism **Type Safe** by turning "hope" into a **guarantee**.

-   **The Rulebook (`ABC`):** You cannot create a generic `Vehicle` object. It is just a blueprint.
-   **The Mandatory Contract (`@abstractmethod`):** Any class inheriting from the blueprint **MUST** implement the method, or Python will stop you immediately with an error.

```python
from abc import ABC, abstractmethod

class Vehicle(ABC):
    @abstractmethod
    def start_engine(self):
        pass  # This is the rule: All vehicles MUST have a way to start!

class Car(Vehicle):
    def start_engine(self):
        return "Car engine started"

class Motorcycle(Vehicle):
    def start_engine(self):
        return "Motorcycle engine started"

# vehicle = Vehicle()  # ✗ Error: Can't instantiate abstract class!
car = Car()            # ✓ Works because it followed the rules
```

#### Why use ABCs?
| Feature | Standard Polymorphism | ABC + Abstract Methods |
| :--- | :--- | :--- |
| **Instantiate Base?** | Yes | **No** (Blueprint only) |
| **Method Required?** | No (Optional) | **Yes** (Mandatory) |
| **Safety** | Lower (Crashes at runtime) | **Higher** (Crashes before it starts) |

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
