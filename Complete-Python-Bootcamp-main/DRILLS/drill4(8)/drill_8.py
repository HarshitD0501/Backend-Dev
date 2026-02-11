class BankAccount:
    def __init__(self, owner, starting_balance=0):
        self.owner = owner
        self.balance = starting_balance

    def deposit(self, amount):
        self.balance += amount
        print(f"Deposited {amount}. New Balance: {self.balance}")

    def withdraw(self, amount):
        if self.balance >= amount:
            self.balance -= amount
            print(f"Withdrew {amount}. New Balance: {self.balance}")
        else:
            raise ValueError("Insufficient Funds")

# Test Code
print("--- Creating Account ---")
my_account = BankAccount("Alice", 100)
print(f"Account for {my_account.owner} created with ${my_account.balance}")

print("\n--- Testing Deposit ---")
my_account.deposit(50)

print("\n--- Testing Withdraw ---")
my_account.withdraw(30)

print("\n--- Testing Overdraft (Should crash if not handled!) ---")
try:
    my_account.withdraw(500)
except ValueError as e:
    print(f"Success! Caught expected error: {e}")