"""Test simple edge function deployment"""
import httpx

token = 'sbp_437b265321c47ad67470b0d7636e41f11479e70c'
project_ref = 'ncnbasvptocuwgxvyjmw'

# Delete existing function
print('Deleting existing function...')
r = httpx.delete(
    f'https://api.supabase.com/v1/projects/{project_ref}/functions/trading-api',
    headers={'Authorization': f'Bearer {token}'}
)
print(f'Delete status: {r.status_code}')

# Create minimal function
simple_code = 'Deno.serve(() => new Response("Hello from trading-api!"));'

print('Creating simple function...')
r = httpx.post(
    f'https://api.supabase.com/v1/projects/{project_ref}/functions',
    headers={'Authorization': f'Bearer {token}', 'Content-Type': 'application/json'},
    json={'slug': 'trading-api', 'name': 'trading-api', 'body': simple_code, 'verify_jwt': False},
    timeout=60
)
print(f'Create status: {r.status_code}')
print(f'Response: {r.text}')

# Test the function
print('\nTesting function...')
import time
time.sleep(3)
r = httpx.get(f'https://{project_ref}.supabase.co/functions/v1/trading-api', timeout=30)
print(f'Test status: {r.status_code}')
print(f'Test response: {r.text}')
