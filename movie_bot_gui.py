import tkinter as tk
import customtkinter
from tkinter import *
from tkinter.ttk import *
import random
import spacy
from PIL import Image, ImageTk
from MovieBot import analyze_input
from randomForest import recommendMovie

default_desc_question = "Can you give a short description what you would like to watch?"

conversation = [                 
"What kind of movie woula you like to watch?        "  ,      
"Do you have a specific movie in mind?              "   ,         
"Would you like to watch a movie today?             "    ,                    
"What is your favorite movie?                       "     ,                       
"Who is your favorite actor?",                               
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

root = Tk()
root.title("MovieBot")

global user_inputs
user_inputs = []
global bot_questions
bot_questions = []
global_count = 0
global current_conversation
current_conversation = conversation

# Create the chatbot's text area
text_area = Text(root, bg="white", width=50, height=50)
text_area.pack()

image = Image.open('Robot1.jpg')
width, height = image.size 
left = 4
top = height / 5
right = 154
bottom = 3 * height / 5

newsize = (120, 180)
image1 = image.resize(newsize)

image1 = ImageTk.PhotoImage(image1)


image_label = tk.Label(root, image = image1)
image_label.place(x=300, y = 400)

# Create the user's input field
input_field = Entry(root, width=50)
input_field.pack()

style = Style()

style.theme_use('alt')

style.configure('W.TButton', font =
               ('calibri', 12, 'bold'),
                foreground = 'black', background = 'turquoise')

# Create the send button
send_button = Button(root, text="Send", style = 'W.TButton', command=lambda: send_message(user_inputs, bot_questions))
send_button.pack(padx=15, side=tk.LEFT)

exit_button = Button(root, text="Exit", style = 'W.TButton', command=lambda: close())
exit_button.pack(padx=30, side=tk.LEFT)

predict_button = Button(root, text="Predict", style = 'W.TButton', command=lambda:predict(user_inputs, bot_questions))
predict_button.pack(padx=20, side=tk.LEFT)

def send_message(user_inputs, bot_questions):
  global global_count

  global_count += 1

  user_input = input_field.get()

  user_inputs.append(user_input)
  # Clear the input field
  input_field.delete(0, END)

  if global_count == 3:
    response = default_desc_question
  else:
  # Generate a response from the chatbot
   response = random.choice(current_conversation)
   current_conversation.pop(current_conversation.index(response))

  bot_questions.append(response)

  # Display the response in the chatbot's text area
  #text_area.insert(END, f"User: {user_input}\n")
  text_area.insert(END, f"User: {user_input}\n")
  text_area.insert(END, f"\nChatbot: {response}\n")

def close():
  root.quit()

def predict(user_inputs, bot_questions):
  description = analyze_input(user_inputs, bot_questions)
  recommendMovie(description)
  #text_area.insert(END, f"Prediction: {description}\n")

root.mainloop()

