"""Full API and Database Test for Trading API"""
import httpx
import json

BASE_URL = 'https://ncnbasvptocuwgxvyjmw.supabase.co/functions/v1/trading-api'
API_KEY = 'TRADING_SECRET_2026_CHANGE_ME'

def test_endpoint(name, method, path, json_data=None, require_auth=True):
    print(f'\n{name}')
    print('-' * 40)
    try:
        url = f'{BASE_URL}{path}'
        if require_auth:
            url += f'?key={API_KEY}'
        if method == 'GET':
            r = httpx.get(url, timeout=10)
        else:
            r = httpx.post(url, json=json_data, timeout=10)
        
        print(f'Status: {r.status_code}')
        
        if r.status_code == 200:
            try:
                data = r.json()
                if isinstance(data, list):
                    print(f'Records: {len(data)}')
                    if data and len(data) > 0:
                        print(f'Sample: {json.dumps(data[0], indent=2, default=str)[:300]}')
                else:
                    print(f'Response: {json.dumps(data, indent=2, default=str)[:300]}')
            except:
                print(f'Raw: {r.text[:300]}')
        else:
            print(f'Response: {r.text[:300]}')
        return r.status_code
    except Exception as e:
        print(f'Error: {e}')
        return None

def main():
    print('=' * 50)
    print('TRADING API FULL TEST')
    print('=' * 50)
    
    results = {}
    
    # 1. Test GET /state
    results['state'] = test_endpoint('1. GET /state', 'GET', '/state')
    
    # 2. Test GET /trades
    results['trades'] = test_endpoint('2. GET /trades', 'GET', '/trades')
    
    # 3. Test GET /equity
    results['equity'] = test_endpoint('3. GET /equity', 'GET', '/equity')
    
    # 4. Test unknown endpoint with auth (should 200 with endpoints list per deployed code)
    results['unknown'] = test_endpoint('4. GET /unknown (with auth)', 'GET', '/unknown')
    
    # 5. Test no auth (should 401)
    results['noauth'] = test_endpoint('5. GET /state (no auth, expect 401)', 'GET', '/state', require_auth=False)
    
    # 6. Test POST /sync
    test_data = {
        'state': {
            'test_timestamp': '2026-01-28T12:00:00Z'
        }
    }
    results['sync'] = test_endpoint('6. POST /sync', 'POST', '/sync', test_data)
    
    # Summary
    print('\n' + '=' * 50)
    print('SUMMARY')
    print('=' * 50)
    
    expected = {'state': 200, 'trades': 200, 'equity': 200, 'unknown': 200, 'noauth': 401, 'sync': 200}
    all_pass = True
    
    for name, expected_code in expected.items():
        actual = results.get(name)
        status = 'PASS' if actual == expected_code else 'FAIL'
        if actual != expected_code:
            all_pass = False
        print(f'{name}: {actual} (expected {expected_code}) - {status}')
    
    print('\n' + ('ALL TESTS PASSED!' if all_pass else 'SOME TESTS FAILED'))

if __name__ == '__main__':
    main()
