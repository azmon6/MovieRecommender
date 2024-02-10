from chatterbot_py import ChatBot
import nltk
import spacy
import random
from randomForest import recommendMovie
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

nlp = spacy.load("en_core_web_sm")

genres_types = ['comedy' , 'drama', 'action' , 'romance', 'horror', 'sci-fi', 'comedy', 'fantasy', 'mistery', 'thriller']

def analyze_input(user_inputs, bot_questions):
    description = []

    user_index = 0

    for input in user_inputs:
        if input.find("No") != -1 or input.lower().find("don't") != -1:
            user_inputs.pop(user_index)
            if user_index > 0:
                bot_questions.pop(user_index - 1)
        user_index = user_index + 1

    tokenized_user_inputs = []
    tokenized_bot_inputs  = []

    tokenized_user_inputs = [nltk.word_tokenize(input) for input in user_inputs] 
    tokenized_bot_inputs = [nltk.word_tokenize(input) for input in bot_questions] 

    tagged_user_inputs = [nltk.pos_tag(words) for words in tokenized_user_inputs]
    tagged_bot_inputs  = [nltk.pos_tag(words) for words in tokenized_bot_inputs]


    for sentence in tagged_bot_inputs:
        for word, tag in sentence:
            if word.lower() in genres_types:
                description.append(word)
            elif tag.startswith('NNP'):
                description.append(word)

    for sentence in tagged_user_inputs:
        for word, tag in sentence:
            if word.lower() in genres_types:
                description.append(word)
            elif tag.startswith('NNP') or tag.startswith('NN') or tag.startswith('CD'):
                description.append(word)

    description = list(set(description))
    description_str = ""
    description_str = " ".join(description)
    return description_str




def is_name(keywords):
    genres = []
    actors = []
    directors = []
    noun = []

    for word in keywords:
        doc = nlp(word)

        return any(ent.label_ == "PERSON" for ent in doc.ents)

def extract_keywords(tagged_inputs):
    keywords = []
    for word, tag in tagged_inputs:
        if tag.startswith('NN') or tag.startswith('JJ') or tag.startswith('NNP'):
            keywords.append(word)
    return keywords


#def generate_questions():
# Define lists of possible components for questions
genres = ["action", "comedy", "drama", "horror", "romance", "sci-fi", "thriller"]
actors = ["Tom Hanks", "Meryl Streep", "Leonardo DiCaprio", "Brad Pitt", "Angelina Jolie"]
directors = ["Steven Spielberg", "Martin Scorsese", "Quentin Tarantino", "Christopher Nolan"]

# Generate a random question
t = 10
conversation = [                 
"What kind of movie woula you like to watch?        "  ,      
"Do you have a specific movie in mind?              "   ,         
"Would you like to watch a movie today?             "    ,                    
"What is your favorite movie?                       "     ,                       
"Who is your favorite actor?"                               
"What is your favorite comedy?",
"Do you have a favorite comedy actor?",
"What is your favorite trhiller?",
"Do you have a favorite trhiller actor?",
"What is your favorite action?",
"Do you have a favorite action actor?",
"What is your favorite sci-fi?",
"Do you have a favorite sci-fi actor?",
"What is your favorite romance?"                                ,
"Do you have a favorite romance actor?",
"What is your favorite comedy?",
"Do you have a favorite comedy actor?                       ",
"What is your favorite horror?",
"Do you have a favorite horror actor?",
"What is your favorite mistery?",
"Do you have a favorite mistery actor?"                             ,
"What is your favorite thriller?",
"Do you have a favorite thriller actor?",
"What is your favorite drama?",
"Do you have a favorite drama actor?",
"Do you have a specific actor in mind?"
]


# Train the chatbot on your movie dataset

user_inputs = []
bot_questions = []
count = 0
current_conversation = conversation

while False:
    try:
        user_input = input()
        bot_input = random.choice(current_conversation)
        print(bot_input)
        current_conversation.pop(current_conversation.index(bot_input))

        user_inputs.append(user_input)
        bot_questions.append(str(bot_input))
        count += 1

        if count == 2:
            print("Can you give a short description what you would like to watch?")
            user_input = input()
            user_inputs.append(user_input)
            description = analyze_input(user_inputs, bot_questions)
            print(description)
            recommendMovie(description)
            count = 0

    except(KeyboardInterrupt, EOFError, SystemExit):
        break