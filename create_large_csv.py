#!/usr/bin/env python3
import csv
import random

# Create a larger CSV file for testing parallel processing
with open('large_test.csv', 'w', newline='') as csvfile:
    writer = csv.writer(csvfile)
    
    # Header
    writer.writerow(['id', 'name', 'age', 'score', 'department', 'salary', 'years_experience'])
    
    departments = ['Engineering', 'Marketing', 'Sales', 'HR', 'Finance']
    
    # Generate 50,000 rows
    for i in range(50000):
        writer.writerow([
            i + 1,
            f'Employee_{i+1:05d}',
            random.randint(22, 65),
            round(random.uniform(60, 100), 1),
            random.choice(departments),
            random.randint(40000, 150000),
            random.randint(0, 40)
        ])

print("Created large_test.csv with 50,000 rows")