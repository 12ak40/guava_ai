import os
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

# The folder to search
SEARCH_FOLDER = os.path.expanduser("~/Desktop/guava_ai")

# Keywords to look for (English and Chinese)
KEYWORDS = [
    'glucosidase', 'inhib', 'abts', 'antioxidant', 'activity', 
    'ic50', 'ec50', '抑制', '活性', '葡萄糖苷酶', '抗氧化'
]

print(f"🔍 Scanning folder: {SEARCH_FOLDER}")
print("=" * 70)

found_files = []

# Walk through all folders and files
for root, dirs, files in os.walk(SEARCH_FOLDER):
    for file in files:
        # Check Excel files
        if file.endswith('.xlsx') or file.endswith('.xls'):
            filepath = os.path.join(root, file)
            try:
                xls = pd.ExcelFile(filepath)
                for sheet_name in xls.sheet_names:
                    # Read just the first 3 rows to get column names
                    df = pd.read_excel(filepath, sheet_name=sheet_name, nrows=3)
                    cols = df.columns.astype(str).tolist()
                    
                    # Check if any keyword is in any column
                    for kw in KEYWORDS:
                        matching_cols = [c for c in cols if kw.lower() in c.lower()]
                        if matching_cols:
                            print(f"\n✅ MATCH FOUND!")
                            print(f"   📁 File: {file}")
                            print(f"   📑 Sheet: {sheet_name}")
                            print(f"   🔑 Matching columns: {matching_cols}")
                            found_files.append((filepath, sheet_name, matching_cols))
                            break
            except Exception as e:
                pass

# Check CSV files too
for root, dirs, files in os.walk(SEARCH_FOLDER):
    for file in files:
        if file.endswith('.csv'):
            filepath = os.path.join(root, file)
            try:
                df = pd.read_csv(filepath, nrows=3)
                cols = df.columns.astype(str).tolist()
                for kw in KEYWORDS:
                    matching_cols = [c for c in cols if kw.lower() in c.lower()]
                    if matching_cols:
                        print(f"\n✅ MATCH FOUND!")
                        print(f"   📁 File: {file}")
                        print(f"   🔑 Matching columns: {matching_cols}")
                        found_files.append((filepath, "CSV", matching_cols))
                        break
            except Exception as e:
                pass

print("\n" + "=" * 70)
if found_files:
    print(f"🎯 Total: {len(found_files)} match(es) found!")
else:
    print("⚠️ No matches found. The bioactivity data might be in the PDF report.")
    print("   Let's list all Excel files instead...")
    
    # Fallback: List all Excel files and their columns
    for root, dirs, files in os.walk(SEARCH_FOLDER):
        for file in files:
            if file.endswith('.xlsx') and not file.startswith('~'):
                filepath = os.path.join(root, file)
                try:
                    xls = pd.ExcelFile(filepath)
                    print(f"\n📁 {file}")
                    for sheet in xls.sheet_names:
                        df = pd.read_excel(filepath, sheet_name=sheet, nrows=1)
                        print(f"   └── {sheet}: {df.columns.tolist()[:5]}...")
                except:
                    pass
