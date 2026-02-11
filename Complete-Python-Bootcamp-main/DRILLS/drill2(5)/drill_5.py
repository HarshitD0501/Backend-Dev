import csv

def generate_stats_report(input_file, output_file):
    """
    Reads a CSV file containing 'name,age'.
    Calculates the average age.
    Writes the result to output_file.
    """
    
    total_age = 0
    count = 0
    
    # Step 1: Read the CSV
    # Hint: Use csv.DictReader or csv.reader
    # Remember to skip the header if using csv.reader!
    
    # with open(input_file, 'r') as ...
        # reader = ...
        # for row in reader:
            # age = int(row['age'])
            # ...
    with open("drill_5_data.csv","r") as file:
        reader=csv.reader(file)
        next(reader)
        for row in reader:
            age=int(row[1])
            total_age+=age
            count+=1
    avg=total_age/count
    
    
    # Step 2: Calculate Average
    # avg = ...
    
    
    # Step 3: Write report to stats.txt
    # Content should look like: "Average Age: 30.0"
    
    with open("stats.txt","w") as file:
        file.write(f"Average Age: {avg}")            
# Test Code
generate_stats_report('drill_5_data.csv', 'stats.txt')
print("Check stats.txt for the result!")
