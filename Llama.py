from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2-xl',device="cuda")
set_seed(42)

inputs = """Out of these movies: The Dark Knight Rises,8.4,2012,"Eight years after the Joker's reign of anarchy, Batman, with the help of the enigmatic Catwoman, is forced from his exile to save Gotham City from the brutal guerrilla terrorist Bane.","Action, Adventure"
Dark Knight,9.0,2008,"When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.","Action, Crime, Drama"
Based on the criteria: Villians, Superheroes, City
I would recommend"""

print(generator(inputs, max_length=200, num_return_sequences=1)[0]["generated_text"])
