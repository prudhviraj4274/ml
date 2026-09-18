import csv
with open("trainingdata.csv") as file:
    data = list(csv.reader(file))
data = data[1:]
concepts = [row[:-1] for row in data]
target = [row[-1] for row in data]
for i, val in enumerate(target):
    if val.lower() == "yes":
        S = concepts[i].copy()
        break
G = [['?' for _ in range(len(S))] for _ in range(len(S))]
for i, h in enumerate(concepts):
    if target[i].lower() == "yes":
        for j in range(len(S)):
            if h[j] != S[j]:
                S[j] = '?'
                G[j][j] = '?'
    else:   
        for j in range(len(S)):
            if h[j] != S[j]:
                G[j][j] = S[j]
            else:
                G[j][j] = '?'
new_G = []
for g in G:
    if g != ['?'] * len(S) and g not in new_G:
        new_G.append(g)
print("Prudhvi Raj.B(192324274)\n")
print("Specific Hypothesis (S)")
print(S)
print("\nGeneral Hypothesis (G)")
if len(new_G) == 0:
    print("No General Hypothesis")
else:
    for g in new_G:
        print(g)
