#!/usr/bin/env python3

import json

def analyze_public_patterns():
    with open('public_cases.json') as f:
        data = json.load(f)

    print('🔍 EDGE CASE PATTERN ANALYSIS - Public Training Data\n')

    # Find cases with very low mileage but high spending
    print('💸 HIGH SPENDING + LOW MILES (suspicious patterns):')
    high_spend_low_miles = []
    for case in data:
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        days = case['input']['trip_duration_days']
        expected = case['expected_output']
        
        if miles <= 20 and receipts > 1500:
            spend_per_mile = receipts / miles if miles > 0 else float('inf')
            high_spend_low_miles.append((spend_per_mile, days, miles, receipts, expected))
    
    high_spend_low_miles.sort(reverse=True)
    for spend_per_mile, days, miles, receipts, expected in high_spend_low_miles[:5]:
        print(f'  {days} days, {miles} miles, ${receipts:.2f} → ${expected:.2f} (${spend_per_mile:.2f}/mile)')

    print()

    # Find cases where reimbursement >> spending
    print('🎯 MASSIVE REIMBURSEMENT MULTIPLIERS (spending vs reimbursement):')
    multipliers = []
    for case in data:
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        
        if receipts > 0:
            multiplier = expected / receipts
            multipliers.append((multiplier, days, miles, receipts, expected))

    # Sort by highest multiplier
    multipliers.sort(reverse=True)
    print('  Top multipliers (getting much more back than spent):')
    for mult, days, miles, receipts, expected in multipliers[:8]:
        print(f'    {days} days, {miles} miles, ${receipts:.2f} → ${expected:.2f} ({mult:.1f}x multiplier)')

    print()

    # Find cases where reimbursement < spending (penalties)
    print('⚠️ PENALTY CASES (reimbursement < spending):')
    penalties = []
    for case in data:
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        
        if expected < receipts:
            ratio = expected / receipts
            penalties.append((ratio, days, miles, receipts, expected))

    penalties.sort()  # Sort by lowest ratio (biggest penalty)
    print('  Biggest penalties (getting much less back than spent):')
    for ratio, days, miles, receipts, expected in penalties[:8]:
        savings = receipts - expected
        print(f'    {days} days, {miles} miles, ${receipts:.2f} → ${expected:.2f} ({ratio:.2f}x, penalty: ${savings:.2f})')

    print()

    # Analyze extreme mileage cases
    print('🚗 EXTREME MILEAGE PATTERNS:')
    extreme_miles = []
    for case in data:
        miles = case['input']['miles_traveled']
        receipts = case['input']['total_receipts_amount']
        days = case['input']['trip_duration_days']
        expected = case['expected_output']
        miles_per_day = miles / days
        
        if miles_per_day > 400:  # Very high daily mileage
            extreme_miles.append((miles_per_day, days, miles, receipts, expected))
    
    extreme_miles.sort(reverse=True)
    print('  Highest daily mileage cases:')
    for mpd, days, miles, receipts, expected in extreme_miles[:5]:
        print(f'    {days} days, {miles} miles, ${receipts:.2f} → ${expected:.2f} ({mpd:.1f} miles/day)')

    print()

    # Find weird spending patterns
    print('🤔 WEIRD SPENDING PATTERNS:')
    
    # Very low spending per day but high reimbursement
    weird_patterns = []
    for case in data:
        receipts = case['input']['total_receipts_amount']
        expected = case['expected_output']
        days = case['input']['trip_duration_days']
        miles = case['input']['miles_traveled']
        
        spend_per_day = receipts / days
        reimb_per_day = expected / days
        
        if spend_per_day < 10 and reimb_per_day > 100:  # Spend <$10/day, get >$100/day back
            weird_patterns.append((reimb_per_day/spend_per_day, days, miles, receipts, expected, spend_per_day, reimb_per_day))
    
    weird_patterns.sort(reverse=True)
    print('  Low daily spending but high daily reimbursement:')
    for ratio, days, miles, receipts, expected, spd, rpd in weird_patterns[:5]:
        print(f'    {days} days, {miles} miles: ${spd:.2f}/day spent → ${rpd:.2f}/day back ({ratio:.1f}x daily ratio)')

if __name__ == "__main__":
    analyze_public_patterns() 