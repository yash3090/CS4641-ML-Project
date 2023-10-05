import streamlit as st
import pandas as pd

st.title('Instrument Identification using Machine Learning')
st.write('By Project Group 13: Ashley Cain, Ethan Haarer, Keshav Jagannath, Matthew Landis, Yash Saraiya')

# Introduction/Background: 
#   A quick introduction of your topic and mostly literature review of what has been done in this area. 
#   You can briefly explain your dataset and its features here too.
st.header('Introduction')
introduction = 'The goal of this project is to identify the instrument being played from a given snippet of audio. This concept has been explored in several research articles as part of bigger audio projects, such as clustering of instrument categories from a multi-instrument recording. We will be using an established dataset of audio files such as NSynth to train a convolutional neural network to fulfill this goal. This dataset also contains classifiers such as "acoustic", "synthetic", and "electronic", as well as "attack" and "decay" time. These are all useful features to have when describing a snippet of audio. NSynth already has its audio files encoded down to integer arrays, and comes with a wide featureset that we will be able to use for training.'
st.write(introduction)

# Problem definition: 
#   Why there is a problem here or what is the motivation of the project?
st.header('Problem Definition')
problem_definition = 'AI classification in the audio realm is a relatively new research field compared to applications such as computer vision. Single instrument classification is an intermediary towards a host of other applications. Companies like "Izotope" are currently interested in developing mixing and mastering virtual assistants based on the ability to classify waveforms into buckets of instrument categories. It would also be useful for individuals to classify instrument details from rehearsal recordings, auditions, or performances for further review.'
st.write(problem_definition)

# Methods: 
#   What algorithms or methods are you going to use to solve the problems. 
#   (Note: Methods may change when you start implementing them which is fine). 
#   Students are encouraged to use existing packages and libraries (i.e. scikit-learn) instead of coding the algorithms from scratch.
st.header('Methods')
methods = """
We will be using the NSynth library and its included feature set in order to train a convolutional neural network. 
We decided upon CNNs by reading different research papers that extract the instruments used from polyphonic audio files and found this to be the most efficient way for our application. 
This particular library has encoded its audio files to the format required. 
For the ML framework, we will be using PyTorch for our multi-class classification model. 
We are beginning with the creation of a supervised model which will use the ground truths to train the convolutional neural network. 
If we make significant progress on the supervised model, we will create an unsupervised model which will use extracted features to define clusters, optimized via a silhouette coefficient. 
Data visualization will be done using the matplotlib library and the built-in functionality of Streamlit. Our model will be trained on the NSynth dataset containing over 300,000 samples and organized around the instrument being played. 
We will first preprocess the data we have from NSynth for our CNN. We will then create the model. After model creation we will train the model and test it on the test set. 
Using the information from test set results we will tune our hyperparameters and extract the most informative features. 
The last two steps will be repeated to fine tune our model and achieve higher accuracies.
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
This testing will be done on a reserved portion of the dataset, and we are hoping to achieve a success rate of 95% or higher. 
When applying the model to real world scenarios, we will be looking at how the model can extract the music from any background noise.
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
st.image("gantchart.png", caption='Gantt Chart', use_column_width=True)
# A contribution table with all group members’ names that explicitly provides the contribution of each member in preparing the project task. This part does NOT count towards word limit.
st.header('Contribution Table')
st.image("contributiontable.png", caption='Contribution Table', use_column_width=True)
# A checkpoint to make sure you are working on a proper machine learning related project. You are required to have your dataset ready when you submit your proposal. You can change dataset later. However, you are required to provide some reasonings why you need to change the dataset (i.e. dataset is not large enough because it does not provide us a good accuracy comparing to other dataset; we provided accuracy comparison between these two datasets). The reasonings can be added as a section to your future project reports such as midterm report.
st.header('Checkpoint')
checkpoint = """
  This is a proper machine learning project because we are using ML libraries and framework such as Librosa and PyTorch in order to create a multi-class classification model, which is a supervised model.
  We will be extracting features of the from the audio files to train the models on test data. Then we will train a convolutional neural network and evaluate the success rate in a quantifiable way in order to evaluate success of the model. These are all key characteristics of a machine learning project. 
"""
st.write(checkpoint)

# Your group needs to submit a presentation of your proposal. Please provide us a public link which includes a 3 minutes recorded video. I found that OBS Studio and GT subscribed Kaltura are good tools to record your screen. Please make your visuals are clearly visible in your video presentation.
# 3 MINUTE is a hard stop. We will NOT accept submissions which are 3 minutes and one second or above. Conveying the message easily while being concise is not easy and it is a great soft skill for any stage of your life, especially your work life.
st.header('Presentation')
st.video('https://www.youtube.com/watch?v=LXb3EKWsInQ')
st.markdown('[Presentation Link](https://www.youtube.com/watch?v=LXb3EKWsInQ)')

st.write('Graded Word Count: ' + str(len((introduction + problem_definition + methods + potential_results_and_discussion).split())))

