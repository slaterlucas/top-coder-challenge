#!/usr/bin/env python3

import json
import math

with open('public_cases.json', 'r') as f:
    data = json.load(f)

# Case 996 details (the fraud case)
fraud_case = None
for i, case in enumerate(data):
    if i == 995:  # Case 996 (0-indexed)
        fraud_case = case
        break

fraud_miles = fraud_case['input']['miles_traveled']  # 1082
fraud_receipts = fraud_case['input']['total_receipts_amount']  # 1809.49
fraud_expected = fraud_case['expected_output']  # 446.94

print(f"FRAUD CASE 996: {fraud_miles} miles, ${fraud_receipts:.2f} → ${fraud_expected:.2f}")
print("=" * 60)

# Investigation 1: Look for cases with similar miles (±20)
print("1. CASES WITH SIMILAR MILES (1060-1100):")
similar_miles = []
for i, case in enumerate(data):
    inp = case['input']
    if inp['trip_duration_days'] == 1 and 1060 <= inp['miles_traveled'] <= 1100:
        similar_miles.append((i+1, case))

for case_num, case in similar_miles[:10]:
    inp, exp = case['input'], case['expected_output']
    print(f"  Case {case_num}: {inp['miles_traveled']} miles, ${inp['total_receipts_amount']:.2f} → ${exp:.2f}")

print()

# Investigation 2: Look for cases with similar spending (±100)
print("2. CASES WITH SIMILAR SPENDING ($1710-1910):")
similar_spending = []
for i, case in enumerate(data):
    inp = case['input']
    if inp['trip_duration_days'] == 1 and 1710 <= inp['total_receipts_amount'] <= 1910:
        similar_spending.append((i+1, case))

for case_num, case in similar_spending[:10]:
    inp, exp = case['input'], case['expected_output']
    print(f"  Case {case_num}: {inp['miles_traveled']} miles, ${inp['total_receipts_amount']:.2f} → ${exp:.2f}")

print()

# Investigation 3: Mathematical relationships
print("3. MATHEMATICAL PATTERNS FOR CASE 996:")
miles_plus_receipts = fraud_miles + fraud_receipts
miles_times_receipts = fraud_miles * fraud_receipts
receipts_div_miles = fraud_receipts / fraud_miles

print(f"  Miles + Receipts = {miles_plus_receipts:.2f}")
print(f"  Miles × Receipts = {miles_times_receipts:.2f}")
print(f"  Receipts ÷ Miles = {receipts_div_miles:.3f}")
print(f"  Miles ÷ 100 = {fraud_miles/100:.2f}")
print(f"  Receipts mod 100 = {fraud_receipts % 100:.2f}")

print()

# Investigation 4: Look for other low-payout cases (potential fraud)
print("4. OTHER POTENTIAL FRAUD CASES (Expected < $600):")
potential_fraud = []
for i, case in enumerate(data):
    inp = case['input']
    exp = case['expected_output']
    if inp['trip_duration_days'] == 1 and inp['miles_traveled'] > 800 and exp < 600:
        potential_fraud.append((i+1, case))

for case_num, case in potential_fraud:
    inp, exp = case['input'], case['expected_output']
    miles_receipts_sum = inp['miles_traveled'] + inp['total_receipts_amount']
    print(f"  Case {case_num}: {inp['miles_traveled']} miles, ${inp['total_receipts_amount']:.2f} → ${exp:.2f}")
    print(f"    Sum: {miles_receipts_sum:.2f}, Product: {inp['miles_traveled'] * inp['total_receipts_amount']:.0f}")

print()

# Investigation 5: Specific range hypothesis
print("5. TESTING SPECIFIC RANGE HYPOTHESIS:")
print("   Looking for miles in 1080-1090 range...")
range_cases = []
for i, case in enumerate(data):
    inp = case['input']
    if inp['trip_duration_days'] == 1 and 1080 <= inp['miles_traveled'] <= 1090:
        range_cases.append((i+1, case))

for case_num, case in range_cases:
    inp, exp = case['input'], case['expected_output']
    is_fraud = exp < 600
    print(f"  Case {case_num}: {inp['miles_traveled']} miles, ${inp['total_receipts_amount']:.2f} → ${exp:.2f} {'[FRAUD]' if is_fraud else '[NORMAL]'}")

print()

# Investigation 6: Hash-like behavior
print("6. HASH-LIKE PATTERNS:")
fraud_digits_sum = sum(int(d) for d in str(fraud_miles))
receipts_digits_sum = sum(int(d) for d in str(int(fraud_receipts)))

print(f"  Miles digit sum: {fraud_digits_sum}")
print(f"  Receipts digit sum: {receipts_digits_sum}")
print(f"  Combined digit sum: {fraud_digits_sum + receipts_digits_sum}")

print(f"\nTotal potential fraud cases found: {len(potential_fraud)}") 