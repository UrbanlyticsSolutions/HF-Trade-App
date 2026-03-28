"""Extract all positions from Questrade"""
import sys
sys.path.insert(0, str(__file__).replace('\\scripts\\get_positions.py', ''))

from clients.questrade_client import create_questrade_client
import json

def main():
    client = create_questrade_client()
    
    # Get accounts
    accounts = client.get_accounts()
    print('Accounts:')
    for acc in accounts:
        print(f"  - {acc.get('type')}: {acc.get('number')} ({acc.get('status')})")
    
    # Get positions for each account
    for acc in accounts:
        acc_id = acc.get('number')
        print(f"\nPositions for account {acc_id}:")
        positions = client.get_account_positions(acc_id)
        if positions:
            for pos in positions:
                print(json.dumps(pos, indent=2))
        else:
            print('  No positions')

if __name__ == "__main__":
    main()
