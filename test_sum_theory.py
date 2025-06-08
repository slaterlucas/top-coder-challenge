#!/usr/bin/env python3

import json

data = json.load(open('public_cases.json'))

# Test if there are other cases near the fraud sum ranges
fraud_sums = [913, 1483, 2891]
print('Cases with Miles+Receipts near fraud sums (±10):')
print()

for fraud_sum in fraud_sums:
    print(f"CHECKING SUM ≈ {fraud_sum}:")
    found_cases = []
    
    for i, case in enumerate(data):
        inp = case['input']
        if inp['trip_duration_days'] == 1:
            total = inp['miles_traveled'] + inp['total_receipts_amount']
            if abs(total - fraud_sum) < 10:
                exp = case['expected_output']
                is_fraud = exp < 600
                found_cases.append((i+1, inp, exp, total, is_fraud))
    
    for case_num, inp, exp, total, is_fraud in found_cases:
        fraud_str = "[FRAUD]" if is_fraud else "[NORMAL]"
        print(f"  Case {case_num}: {inp['miles_traveled']} miles, ${inp['total_receipts_amount']:.2f}, Sum: {total:.2f} → ${exp:.2f} {fraud_str}")
    
    fraud_count = sum(1 for _, _, _, _, is_fraud in found_cases if is_fraud)
    print(f"  Fraud cases in this range: {fraud_count}/{len(found_cases)}")
    print()

# Additional check: Look for patterns in the exact fraud cases
print("DETAILED ANALYSIS OF ALL 4 FRAUD CASES:")
fraud_cases = [75, 759, 899, 996]
for case_num in fraud_cases:
    case = data[case_num - 1]  # Convert to 0-indexed
    inp = case['input']
    exp = case['expected_output']
    total = inp['miles_traveled'] + inp['total_receipts_amount']
    
    print(f"Case {case_num}: {inp['miles_traveled']} miles, ${inp['total_receipts_amount']:.2f}")
    print(f"  Sum: {total:.2f}, Expected: ${exp:.2f}")
    print(f"  Miles % 100: {inp['miles_traveled'] % 100}")
    print(f"  Receipts % 100: {inp['total_receipts_amount'] % 100:.2f}")
    print() 