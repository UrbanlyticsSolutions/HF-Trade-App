import httpx
import json

token = 'sbp_437b265321c47ad67470b0d7636e41f11479e70c'
project_ref = 'ncnbasvptocuwgxvyjmw'

# Get API keys
resp = httpx.get(
    f'https://api.supabase.com/v1/projects/{project_ref}/api-keys',
    headers={'Authorization': f'Bearer {token}'}
)
keys = resp.json()
print("API Keys:")
for key in keys:
    print(f"  {key.get('name')}: {key.get('api_key')[:20]}...")
    if key.get('name') == 'anon':
        anon_key = key.get('api_key')
    if key.get('name') == 'service_role':
        service_key = key.get('api_key')

print(f"\nSupabase URL: https://{project_ref}.supabase.co")
print(f"Anon Key: {anon_key}")
print(f"Service Role Key: {service_key[:30]}...")
