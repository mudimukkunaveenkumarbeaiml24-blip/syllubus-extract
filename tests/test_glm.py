from app.nvidia_client import test_nvidia

print("\nNVIDIA TEST")
print("=" * 50)

try:
    result = test_nvidia()
    print(result)
    print("=" * 50)
    print("SUCCESS")
except Exception as e:
    print("=" * 50)
    print("ERROR:")
    print(e)
    print("=" * 50)