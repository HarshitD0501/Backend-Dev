import csv
import datetime
import os

class Expense:
    """
    Represents a single expense.
    Attributes: date (str), category (str), amount (float)
    """
    def __init__(self, category, amount):
        self.category = category
        self.amount = amount
        # Auto-set date to today (YYYY-MM-DD)
        self.date = datetime.date.today().isoformat()

    def to_csv_row(self):
        """Returns a list suitable for csv.writer: [date, category, amount]"""
        return [self.date, self.category, self.amount]

class ExpenseTracker:
    """
    Manages the list of expenses and file operations.
    """
    def __init__(self, filename="expenses.csv"):
        self.filename = filename
        
        if not os.path.exists(self.filename):
            with open(self.filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(['date', 'category', 'amount'])
        
    def add_expense(self):
        """
        1. Ask user for Category and Amount.
        2. Create an Expense object.
        3. Append it to the CSV file.
        """
        print("\n--- Add New Expense ---")
        # TODO: Ask for inputs (Handle ValueError for amount!)
        category = input("Enter category: ")
        amount = float(input("Enter amount: "))
        # TODO: Create Object
        expense = Expense(category, amount)
        # TODO: Write to file (Mode 'a')
        with open(self.filename, 'a', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(expense.to_csv_row())
        print("Expense Saved!")

    def show_summary(self):
        """
        Reads CSV, sums up amount per category, and prints report.
        Example:
        Food: $50
        Travel: $100
        """
        print("\n--- Expense Summary ---")
        if not os.path.exists(self.filename):
            print("No expenses yet.")
            return

        summary = {} # Key: Category, Value: Total Amount

        # TODO: Read CSV
        # Loop through rows, add amounts to the dictionary
        with open(self.filename, 'r') as f:
            reader = csv.reader(f)
            next(reader) # Skip header
            for row in reader:
                category = row[1]
                amount = float(row[2])
                if category in summary:
                    summary[category] += amount
                else:
                    summary[category] = amount
        
        # TODO: Print results from dictionary
        for category, amount in summary.items():
            print(f"{category}: ${amount}")
        

# --- Main Game Loop (Don't touch this) ---
tracker = ExpenseTracker()

while True:
    print("\n1. Add Expense")
    print("2. Show Summary")
    print("3. Exit")
    choice = input("Select (1-3): ")

    if choice == '1':
        tracker.add_expense()
    elif choice == '2':
        tracker.show_summary()
    elif choice == '3':
        print("Goodbye!")
        break
    else:
        print("Invalid choice!")
