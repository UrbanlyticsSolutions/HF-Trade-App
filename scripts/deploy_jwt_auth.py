"""Deploy JWT-authenticated edge function to Supabase"""
import httpx

TOKEN = 'sbp_437b265321c47ad67470b0d7636e41f11479e70c'
PROJECT = 'ncnbasvptocuwgxvyjmw'

# Read the function code
with open(r'\\dh4300plus-7195\personal_folder\NASCODE\Trade\Live\Paper\supabase\functions\trading-api\index.ts', 'r') as f:
    code = f.read()

# Remove leading newline if present
if code.startswith('\n'):
    code = code[1:]

print("Deploying trading-api with JWT authentication...")
print(f"Code length: {len(code)} bytes")

resp = httpx.patch(
    f'https://api.supabase.com/v1/projects/{PROJECT}/functions/trading-api',
    headers={
        'Authorization': f'Bearer {TOKEN}',
        'Content-Type': 'application/json'
    },
    json={'body': code, 'verify_jwt': False},
    timeout=60
)

print(f'Status: {resp.status_code}')
if resp.status_code == 200:
    print('✅ Deployment successful!')
    print(f'Function URL: https://{PROJECT}.supabase.co/functions/v1/trading-api')
else:
    print(f'❌ Error: {resp.text[:500]}')
