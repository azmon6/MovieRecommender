from transformers import pipeline, set_seed
generator = pipeline('text-generation', model='gpt2-xl',device="cuda")
set_seed(42)

inputs = """The Dark Knight-When the menace known as the Joker wreaks havoc and chaos on the people of Gotham, Batman must accept one of the greatest psychological and physical tests of his ability to fight injustice.
Joker-In Gotham City, mentally troubled comedian Arthur Fleck is disregarded and mistreated by society. He then embarks on a downward spiral of revolution and bloody crime. This path brings him face-to-face with his alter-ego: the Joker.
The Dark Knight Rises-Eight years after the Joker's reign of anarchy, Batman, with the help of the enigmatic Catwoman, is forced from his exile to save Gotham City from the brutal guerrilla terrorist Bane.
Batman Begins-After training with his mentor, Batman begins his fight to free crime-ridden Gotham City from corruption.
The Circus-The Tramp finds work and the girl of his dreams at a circus.
Based on the criteria: Batman
I would recommend"""

print(generator(inputs, max_length=250, num_return_sequences=1)[0]["generated_text"])
