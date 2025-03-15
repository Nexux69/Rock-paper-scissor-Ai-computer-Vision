#Initialize Variables
model = None
box_size = None
width = None
attempts = None
computer_move_name = None
final_user_move = None
label_names = None
computer_score = None
user_score = None
rect_color = None
hand_inside = None
total_attempts = None
confidence_threshold = None
smooth_factor = None
de = None
frame = None
ret = None
roi = None
pred = None
move_code = None
user_move = None
prob = None
winner = None
play_again = None

#Import Variables
import os
import cv2
import numpy as np
np.set_printoptions(suppress=True) # prevent numpy exponential
import matplotlib.pyplot as plt
import sys
import io
import base64
import random
plt.clf()
import time
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Dense,MaxPool2D,Dropout,Flatten,Conv2D,GlobalAveragePooling2D,Activation
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from random import choice
from random import shuffle
from scipy import stats as st
from collections import deque

#Load Pre-trained Model

model = load_model("C:\\Users\\DELL\\Desktop\\faiz\\rock_paper_scissor\\rock_paper_scissor\\model.h5")

#Function for Finding Winner
def findout_winner(user_move, Computer_move):
	if user_move == Computer_move:
		return "Tie"
	elif user_move == "rock" and Computer_move == "scissor":
		return "User"
	elif user_move == "rock" and Computer_move == "paper":
		return "Computer"
	elif user_move == "scissor" and Computer_move == "rock":
		return "Computer"
	elif user_move == "scissor" and Computer_move == "paper":
		return "User"
	elif user_move == "paper" and Computer_move == "rock":
		return "User"
	elif user_move == "paper" and Computer_move == "scissor":
		return "Computer"

#Function for showing the Winner of the Game
def show_winner(user_score, computer_score):
	if user_score > computer_score:
		img = cv2.imread("images/youwin.jpg")
	elif user_score < computer_score:
		img = cv2.imread("images/comwins.jpg")
	else:
		img = cv2.imread("images/draw.jpg")

	cv2.putText(img, "Press 'ENTER' to play again, else exit",
				(150, 530), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 0, 255), 3, cv2.LINE_AA)

	cv2.imshow("Rock Paper Scissors", img)
	#If enter is pressed.
	k = cv2.waitKey(0)
	#If the user presses 'ENTER' key then return TRUE, otherwise FALSE
	if k == 13:
		 return True
	else:
		return False

#Function for Showing Computer's Move	
def display_computer_move(computer_move_name, frame):
	icon = cv2.imread( "images/{}.png".format(computer_move_name), 1)
	icon = cv2.resize(icon, (224,224))
	#This is the portion which we are going to replace with the icon image
	roi = frame[0:224, 0:224]
	#Get binary mask from the transparent image, 4th channel is the alpha channel
	mask = icon[:,:,-1]
	#Making the mask completely binary (black & white)
	mask = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)[1]
	#Store the normal bgr image
	icon_bgr = icon[:,:,:3]
	#Now combine the foreground of the icon with background of ROI
	img1_bg = cv2.bitwise_and(roi, roi, mask = cv2.bitwise_not(mask))
	img2_fg = cv2.bitwise_and(icon_bgr, icon_bgr, mask = mask)
	combined = cv2.add(img1_bg, img2_fg)
	frame[0:224, 0:224] = combined
	return frame

#Rock Paper Scissor - Final Application
cap = cv2.VideoCapture(0)
box_size = 234
width = int(cap.get(3))
##Specify the number of attempts you want. This means best of 5.
attempts = 10
#Initially the moves will be `nothing`
computer_move_name = 'nothing'
final_user_move = 'nothing'
label_names = ['nothing', 'paper', 'rock', 'scissor']
#All scores are 0 at the start
computer_score = 0
user_score = 0
#The default color of bounding box is Blue
rect_color = (255, 0, 0)
#This variables remembers if the hand is inside the box or not
hand_inside = False
#At each iteration we will decrease the total_attempts value by 1
total_attempts = attempts
#We will only consider predictions having confidence above this threshold
confidence_threshold = 0.7
#Instead of working on a single prediciton, we will take the mode of 5 predictons by using a deque object
#This way even if we face a false postive, we would easily ignore it
smooth_factor = 5
#Our initial deque list will have 'nothing' repeated 5 times.
de = deque(['nothing'] * 5, maxlen=smooth_factor)

while True:
  ret, frame = cap.read()
  if not ret:
    break
  frame = cv2.flip(frame, 1)

  cv2.namedWindow('Rock Paper Scissors', cv2.WINDOW_NORMAL)

  #Extract the region of image within the user rectangle
  roi = frame[5: box_size-5 , width-box_size + 5: width -5]
  roi = np.array([roi]).astype('float64') / 255.0
  #Predict the move made
  pred = model.predict(roi)
  #Get the index of the predicted class
  move_code = np.argmax((pred[0]))
  #Get the class name of the predicted class
  user_move = label_names[int((move_code + 1) - 1)]
  #Get the confidence of the predicted class
  prob = np.max(pred[0])

  #Get Computer's move and then get the winner.
  #computer_move_name = choice(['rock', 'paper', 'scissor'])
  #winner = findout_winner(final_user_move, computer_move_name)
  #Make sure the probability is above our defined threshold
  if prob >= confidence_threshold:
    #Now add the move to deque list from left
    de.appendleft(user_move)
    #Get the mode i.e. which class has occured more frequently in the last 5 moves.
    try:
      final_user_move = st.mode(de)[0][0]
    except :
      print('Stats Error !!')
      continue

    #If nothing is not true and hand_inside is False then proceed.
    #Basically the hand_inside variable is helping us to not repeatedly predict during the loop
    #So now the user has to take his hands out of the box for every new predicton.
    if final_user_move != 'nothing' and hand_inside == False:
      #Set hand inside to True
      hand_inside = True
      #Get Computer's move and then get the winner
      computer_move_name = choice(['rock', 'paper', 'scissor'])
      winner = findout_winner(final_user_move, computer_move_name)
      #Display the computer's move
      display_computer_move(computer_move_name, frame)
      # Subtract one attempt
      total_attempts = total_attempts - 1
      
      #If winner is computer then it gets points and vice versa.
      #We're also changing the color of rectangle based on who wins the round.
      if winner == 'Computer':
        computer_score = computer_score + 1
        rect_color = (0, 0, 255)
      elif winner == 'User':
        user_score = user_score + 1
        rect_color = (0, 250, 0)
      elif winner == 'Tie':
        rect_color = (255, 250, 255)
	
      #If all the attempts are up then find our the winner
      if total_attempts == 0:
        play_again = show_winner(user_score, computer_score)
        #If the user pressed Enter then restart the game by re initializing all variables
        if play_again:
          user_score = 0
          computer_score = 0
          total_attempts = attempts
        #Otherwise quit the program.
        else:
          break
	
    #Display images when the hand is inside the box even when hand_inside variable is True.
    elif final_user_move != 'nothing' and hand_inside == True:
      display_computer_move(computer_move_name, frame)

    #If class is nothing then hand_inside becomes False
    elif final_user_move == 'nothing':
      hand_inside = False
      rect_color = (255, 0, 0)

  #This is where all annotation is happening. 
  cv2.putText(frame, 'Your Move :  ' + str(final_user_move), (420, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
  cv2.putText(frame, str("Computer's Move :  ") + str(computer_move_name), (2, 270), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
  cv2.putText(frame, 'Your Score :  ' + str(user_score), (420, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
  cv2.putText(frame, str("Computer's Scor :  ") + str(computer_score), (2, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1, cv2.LINE_AA)
  cv2.putText(frame, 'Attempts Left :  ' + str(total_attempts), (190, 400), cv2.FONT_HERSHEY_COMPLEX, 0.7, (255, 0, 0), 1, cv2.LINE_AA)
  cv2.rectangle(frame, (width - box_size, 0), (width, box_size), rect_color, 2)
  
  #Display the image
  cv2.imshow('Rock Paper Scissors', frame)

  #Exit if 'q' is pressed 
  if cv2.waitKey(1) & 0xFF == ord('q'):
  		break

# Relase the camera and destroy all windows
cap.release()
cv2.destroyAllWindows()