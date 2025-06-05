import pandas as pd

# Créer un DataFrame d'exemple
df = pd.DataFrame({
    "Nom": ["Alice", "Bob", "Charlie"],
    "Age": [25, 30, 35]
})

# Sauvegarde en différents formats
df.to_excel("fichier.xlsx", index=False)      # .xlsx
