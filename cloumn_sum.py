import csv

# Path to your CSV file
csv_file_path = 'Images_cam4/Summary of Images_cam4.csv'

# Initialize a variable to store the sum
total_sum = 0

# Open and read the CSV file
with open(csv_file_path, mode='r') as file:
    csv_reader = csv.DictReader(file)  # Use DictReader to access columns by their names
    for row in csv_reader:
        # Convert the second column value to a number and add to the sum
        total_sum += float(row[list(row.keys())[1]])  # Use the second column by index
#sdfadsf
# Print the result
print("Sum of the second column:", total_sum)
