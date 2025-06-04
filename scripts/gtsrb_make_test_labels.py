import csv

# Paths to your CSV and output txt
csv_path = "datasets/gtsrb/GT-final_test.csv"
output_txt = "datasets/gtsrb/test_labels.txt"

with open(csv_path, newline='') as csvfile, open(output_txt, 'w') as out:
    reader = csv.DictReader(csvfile, delimiter=';')
    for row in reader:
        filename = row["Filename"]
        label = row["ClassId"]
        out.write(f"{filename} {label}\n")

print(f"Written: {output_txt}")