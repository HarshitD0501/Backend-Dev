from abc import ABC, abstractmethod

# ==========================================
# 🎯 OOP MASTER DRILL (Topics 8.1 - 8.7)
# ==========================================
# Scenario: An E-Commerce System
# You will build a system with Products, Carts, and Payment Processors.
# Follow the TODOs to implement the missing pieces!

# ------------------------------------------
# 1. Abstraction (8.5)
# ------------------------------------------
class PaymentProcessor(ABC):
    """
    Abstract Base Class for payment processing.
    """
    @abstractmethod
    def process_payment(self, amount):
        pass
    

# ------------------------------------------
# 2. Encapsulation (8.4) & Basic OOP (8.1)
# ------------------------------------------
class Product:
    def __init__(self, name, price):
        self.name = name
        # TODO: Make price a protected attribute (use _price)
        self._price = price

    # TODO: Create a getter method for price
    @property
    def price(self):
        return self._price

    # TODO: Create a setter method for price that ensures it's positive
    @price.setter
    def price(self, value):
        if value < 0:
            raise ValueError("Price cannot be negative!")
        self._price = value

    # ------------------------------------------
    # 3. Magic Methods (8.6)
    # ------------------------------------------
    def __str__(self):
        # TODO: Return a string like "Laptop: $999"
        return f"{self.name}: ${self.price}"

    # ------------------------------------------
    # 4. Operator Overloading (8.7)
    # ------------------------------------------
    def __add__(self, other):
        # TODO: Allow adding two Product objects to get their combined price
        return self.price + other.price

# ------------------------------------------
# 5. Inheritance (8.2)
# ------------------------------------------
class Electronic(Product):
    def __init__(self, name, price, warranty_months):
        # TODO: Call the parent class (Product) __init__
        super().__init__(name, price)
        self.warranty_months = warranty_months

class Clothing(Product):
    def __init__(self, name, price, size):
        super().__init__(name, price)
        self.size = size

# ------------------------------------------
# 6. Polymorphism (8.3)
# ------------------------------------------
class CreditCardPayment(PaymentProcessor):
    def process_payment(self, amount):
        print(f"Processing credit card payment of ${amount}")

class PayPalPayment(PaymentProcessor):
    def process_payment(self, amount):
        # TODO: Implement this method to print specific PayPal message
        print(f"Processing PayPal payment of ${amount}")

# ==========================================
# 🚀 TEST YOUR CODE
# ==========================================
if __name__ == "__main__":
    # 1. Create Products
    laptop = Electronic("MacBook", 1200, 12)
    shirt = Clothing("T-Shirt", 25, "L")

    # 2. Test Encapsulation
    # laptop.price = -50  # Should raise error if uncommented
    print(f"Product: {laptop.name}, Price: ${laptop.price}")

    # 3. Test Magic Method (__str__)
    print(laptop) # Expected: eBook: $1200 or similar

    # 4. Test Operator Overloading (__add__)
    total = laptop + shirt
    print(f"Total Cart Value: ${total}") # Expected: 1225

    # 5. Test Polymorphism
    payments = [CreditCardPayment(), PayPalPayment()]
    for p in payments:
        p.process_payment(total)
