import csv
from collections import Counter

# Define file paths
csv_file_path = r'20250203_analysis\rec14\smoothed_trajectories'  # Input file path
output_file_path = r'20250203_analysis\rec14\trej_length.txt'  # Final output file

# Initialize a Counter for counting occurrences
counts = Counter()

# Step 1: Process the input file and count numbers from the first column (excluding -1)
with open(csv_file_path, mode='r') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter='\t')  # Assuming tab-separated values
    next(csv_reader)  # Skip the header row if present
    
    for row in csv_reader:
        try:
            number = int(row[0])  # Convert the first column value to an integer
            if number != -1:  # Exclude -1
                counts[number] += 1
        except ValueError:
            print(f"Skipping invalid row: {row}")

# Step 2: Count how many times each frequency (number of occurrences) appears
frequency_counts = Counter(counts.values())

# Step 3: Sort the frequency counts in descending order
sorted_frequency_counts = sorted(frequency_counts.items(), key=lambda x: x[0], reverse=True)


# Step 4: Calculate the sum of multiplication for the first 1/5 of the list
first_fifth = sorted_frequency_counts[:max(1, len(sorted_frequency_counts) // 5)]  # Get the first 1/5
sum_multiplication = sum(frequency * count for frequency, count in first_fifth)

# Step 5: Write the summarized results to the final output file
with open(output_file_path, mode='w') as output_file:
    for frequency, count in sorted_frequency_counts:
        output_file.write(f"{frequency}: {count}\n")
    
    # Write the calculated sum at the end of the file
    output_file.write(f"\nSum of multiplication for the first 1/5 of the list: {sum_multiplication}\n")

print(f"Summarized counts have been written to {output_file_path}")
