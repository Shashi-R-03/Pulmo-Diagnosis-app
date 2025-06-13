import cohere

cohere_api_key = 'DbqJjall1lDsQewj14wfw5uulIM7DYqKF5CsgJDA'  # Replace with your actual API key
co = cohere.Client(cohere_api_key)

response = co.generate(
   model='command',
   prompt="What are the symptoms of pneumonia?",
   max_tokens=100,
   temperature=0.5
)

print(response.generations[0].text.strip())