import streamlit as st
import pandas as pd

st.title('Instrument Identification using Machine Learning')
st.write('By Project Group 13: Ashley Cain, Ethan Haarer, Keshav Jagannath, Matthew Landis, Yash Saraiya')

# Introduction/Background: 
#   A quick introduction of your topic and mostly literature review of what has been done in this area. 
#   You can briefly explain your dataset and its features here too.
st.header('Introduction')
introduction = 'The goal of this project is to identify the instrument being played in a given audio file. We will be using a dataset of audio files of various instruments being played. We will be using machine learning to train a model to identify the instrument being played in a given audio file. We will be using the Librosa library to extract features from the audio files. We will be using the scikit-learn library to train and test our model. We will be using the matplotlib library to visualize our data.'
st.write(introduction)

# Problem definition: 
#   Why there is a problem here or what is the motivation of the project?
st.header('Problem Definition')
problem_definition = 'The motivation for this project is to be able to identify the instrument being played in a given audio file. This could be used to identify the instrument being played in a song. This could also be used to identify the instrument being played in a recording of a live performance. This could also be used to identify the instrument being played in a recording of a practice session. This could also be used to identify the instrument being played in a recording of a lesson. This could also be used to identify the instrument being played in a recording of a rehearsal.'
st.write(problem_definition)

# Methods: 
#   What algorithms or methods are you going to use to solve the problems. 
#   (Note: Methods may change when you start implementing them which is fine). 
#   Students are encouraged to use existing packages and libraries (i.e. scikit-learn) instead of coding the algorithms from scratch.
st.header('Methods')
methods = """
We will be using the Librosa library to extract features from the audio files. Significant experimentation will have to take place to ensure that the data is in the best form to be trained upon.
For the ML framework, we will be using PyTorch for our multi-class classification model.
We are beginning our work with the creation of a supervised model which will use the ground truths of the dataset to train the convolutional neural net.
If we make signficant progress on the supervised model, we will create an unsupervised model which will use the features extracted from the audio files optimized via a silloutte coefficient to measure the quality of the clusters.
Data visualization will be done using the matplotlib library and the built-in functionality of Streamlit.
Our model will be trained on the NSynth dataset containing over 300,000 samples and organized around the instrument being played.
"""
#Librosa: https://librosa.org/
#PyTorch: https://www.learnpytorch.io/02_pytorch_classification/
st.write(methods)

# Potential results and Discussion:
#   (The results may change while you are working on the project and it is fine; that’s why it is called research). 
#   A good way to talk about potential results is to discuss about what type of quantitative metrics your team plan to use for the project (i.e. ML Metrics).
st.header('Potential Results and Discussion')
potential_results_and_discussion = """
We will be looking for the success rate for which the model can identify the instrument being played in a given audio file. 
This testing will be done on a reserved portion of the dataset that the model has not been trained on.
We are hoping to achieve a success rate of 95% or higher.
"""
st.write(potential_results_and_discussion)

# At least three references (preferably peer reviewed). You need to properly cite the references on your proposal. This part does NOT count towards word limit.
st.header('References')
st.markdown(
"""
- Yoonchang Han, Jaehun Kim, and Kyogu Lee, “Deep convolutional neural networks for predominant instrument recognition in polyphonic music,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 25, no. 1, pp. 208–221, 2016.
-  Peter Li, Jiyuan Qian, and Tian Wang, “Automatic instrument recognition in polyphonic music using convolutional neural networks,” arXiv preprint arXiv:1511.05520, 2015.
"""
)

# Add proposed timeline from start to finish and list each project members’ responsibilities. Fall and Spring semester sample Gantt Chart. This part does NOT count towards word limit.
st.header('Timeline')

# A contribution table with all group members’ names that explicitly provides the contribution of each member in preparing the project task. This part does NOT count towards word limit.
st.header('Contribution Table')

# A checkpoint to make sure you are working on a proper machine learning related project. You are required to have your dataset ready when you submit your proposal. You can change dataset later. However, you are required to provide some reasonings why you need to change the dataset (i.e. dataset is not large enough because it does not provide us a good accuracy comparing to other dataset; we provided accuracy comparison between these two datasets). The reasonings can be added as a section to your future project reports such as midterm report.
st.header('Checkpoint')

# Your group needs to submit a presentation of your proposal. Please provide us a public link which includes a 3 minutes recorded video. I found that OBS Studio and GT subscribed Kaltura are good tools to record your screen. Please make your visuals are clearly visible in your video presentation.
# 3 MINUTE is a hard stop. We will NOT accept submissions which are 3 minutes and one second or above. Conveying the message easily while being concise is not easy and it is a great soft skill for any stage of your life, especially your work life.
st.header('Presentation')
st.video('https://www.youtube.com/watch?v=LXb3EKWsInQ')
st.markdown('[Presentation Link](https://www.youtube.com/watch?v=LXb3EKWsInQ)')

st.write('Graded Word Count: ' + str(len((introduction + problem_definition + methods + potential_results_and_discussion).split())))

