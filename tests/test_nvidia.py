from app.nvidia_client import generate_text

prompt = """
Which number is larger?

9.11 or 9.8

Answer with only the larger number.
"""

result = generate_text(prompt)

print("\nNVIDIA RESPONSE:")
print(result)