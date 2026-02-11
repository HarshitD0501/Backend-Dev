def get_valid_number():
    """
    Asks the user for a number.
    Keeps asking until they enter a valid integer.
    Handles 'abc' (ValueError) and Ctrl+C (KeyboardInterrupt).
    """
    
    while True:
        try:
            user_input = input("Enter a number: ")
            number = int(user_input)
            return number
        except ValueError:
            print("Invalid input. Please enter a valid number.")
        except (KeyboardInterrupt, EOFError):
            print("\nGoodbye!")
            return None
            
# Test Code
print("Please enter your age:")
age = get_valid_number()

if age is not None:
    print(f"Success! Your age is {age}")
