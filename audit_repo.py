import os

print("="*60)
print("📂 REPOSITORY AUDIT: guava_ai")
print("="*60)

# List all files in the directory
files = os.listdir('.')
files.sort()

# Categorize files
py_files = [f for f in files if f.endswith('.py')]
csv_files = [f for f in files if f.endswith('.csv')]
png_files = [f for f in files if f.endswith('.png')]
txt_files = [f for f in files if f.endswith('.txt')]

print("\n🐍 Python Scripts (Code):")
for f in py_files: print(f"  - {f}")

print("\n📊 Data Files (CSV):")
for f in csv_files: print(f"  - {f}")

print("\n🖼️ Figures (PNG):")
for f in png_files: print(f"  - {f}")

print("\n📄 Text/Docs (TXT):")
for f in txt_files: print(f"  - {f}")

print("\n" + "="*60)
print("✅ Audit Complete. Compare this list to the requirements below.")
