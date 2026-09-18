import csv
with open("trainingdata.csv") as file:
    data = list(csv.reader(file))
print("Prudhvi Raj.B(192324274)\n")
print("Training Data:")
for row in data:
    print(row)
hypothesis = data[0][:-1]
for row in data:
    if row[-1].lower() == "yes":
        for i in range(len(hypothesis)):
            if hypothesis[i] != row[i]:
                hypothesis[i] = "?"
print("\nMost Specific Hypothesis:")
print(hypothesis)
