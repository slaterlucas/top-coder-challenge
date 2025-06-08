#!/usr/bin/env python3

import json

with open('public_cases.json', 'r') as f:
    data = json.load(f)

extreme_cases = []
for i, case in enumerate(data):
    inp = case['input']
    days = inp['trip_duration_days']
    miles = inp['miles_traveled']
    receipts = inp['total_receipts_amount']
    
    # Find cases with: 1 day, >1000 miles, >$1000 spending
    if days == 1 and miles > 1000 and receipts > 1000:
        extreme_cases.append((i+1, case))

print(f'Found {len(extreme_cases)} extreme cases (1 day, >1000 miles, >$1000 spending):')
print()

for case_num, case in extreme_cases:
    inp = case['input']
    exp = case['expected_output']
    days = inp['trip_duration_days']
    miles = inp['miles_traveled']
    receipts = inp['total_receipts_amount']
    
    spend_per_mile = receipts / miles if miles > 0 else 0
    
    print(f'Case {case_num}: {days} day, {miles} miles, ${receipts:.2f} receipts')
    print(f'  Expected: ${exp:.2f}')
    print(f'  Spend per mile: ${spend_per_mile:.2f}')
    print()
    
print(f'Total found: {len(extreme_cases)} cases') 