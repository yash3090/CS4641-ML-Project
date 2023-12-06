import streamlit as st
import pandas as pd

fin, mid, pro = st.tabs(["Final Report", "Midterm Report", "Proposal"])


# MIDTERM REPORT
fin.title('Instrument Identification using Machine Learning Midterm Report')

fin.write('By Project Group 13: Ashley Cain, Ethan Haarer, Keshav Jagannath, Matthew Landis, Yash Saraiya')

# Introduction/Background: 
fin.header('Introduction')
introduction1 = 'Automatic musical instrument recognition from audio has been widely studied, with deep learning approaches emerging as state-of-the-art. Prior convolutional neural network (CNN) models have shown promising results on polyphonic instrument identification in both classical and synthesized music (Han, Kim, & Lee, 2016; Li, Qian, & Wang, 2015). Computer vision techniques have also been applied for music classification from spectrograms (Ke, Hoiem & Sukthankar, 2005).'
fin.write(introduction1)
introduction2 = 'In this work, we explore deep CNNs for monophonic instrument recognition using the NSynth dataset (NSynth Dataset, n.d.), which provides over 300,000 musical notes from 1000 instruments. We frame this as a multi-class classification problem across 11 instrument categories including keyboard, guitar, bass, and brass. Using 1D and 2D CNN architectures, we classify short solo instrument excerpts based on the raw waveforms and derived spectrograms respectively. We analyze factors like training convergence, accuracy, and confusion patterns. The goal is to develop an optimized deep learning approach to timbre-based instrumentation.'
fin.write(introduction2)

# References:
fin.header('References')
fin.markdown(
"""
- Han, Y., Kim, J., & Lee, K. (2016). Deep convolutional neural networks for predominant instrument recognition in polyphonic music. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 25(1), 208–221. https://doi.org/10.1109/taslp.2016.2623355
- Ke, Y., Hoiem, D. & Sukthankar, R. (2005). Computer vision for music identification. In 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05) (Vol. 1, pp. 597-604). IEEE. https://doi.org/10.1109/CVPR.2005.105
- Li, P., Qian, J., & Wang, T. (2015). Automatic instrument recognition in polyphonic music using convolutional neural networks. arXiv preprint arXiv:1511.05520.
- NSynth Dataset. (n.d.). https://magenta.tensorflow.org/datasets/nsynth
""")

# Problem definition: 
fin.header('Problem Definition')
problem_definition = 'The main problem we are aiming to solve is the ability to identify a musical instrument through an audio file sample. One main issue for many artists and music producers and editors is the inability for them to edit or isolate specific tracks of popular songs that they want to edit, remix, sample or alter to meet their needs. This Ai model could potentially be adapted to identify instruments and their notes and aid artists and production software in isolating instruments and tracks from the rest of the song so that songs may be deconstructed to allow for editing and creating new songs based on preexisting tracks and popular media. This could further be used to generate sheet music based on a song playing, especially if that song is lost media and the original files are lost. This model could then reconstruct that music to be learned and reused in the future.'
fin.write(problem_definition)

# Methods:
fin.header('Methods')
methods1 = 'Before we send the data through our models, we first preprocess them by normalizing the frequency values in each randomly selected audio file from the dataset, mapping the original range of [-1, 1] to [0, 1]. Then we run the PCA algorithm to reduce the features of the dataset to make the training easier. Both of these are primarily done before training in our first model, however for our second iteration of the model, we add the additional step of converting the audio files to a spectrogram representation.'
fin.write(methods1)
methods2 = 'For our batch sizes, we randomly shuffle the training and test points between training to better test accuracy of the model.'
fin.write(methods2)
methods3 = 'The first model we constructed was a convolutional neural network taking in a one-dimensional waveform, built within Tensorflow on Google Colab. This model took in each sample of a note as a single input, and with notes in the n-synth dataset having 16,000 samples per second and each being 4 seconds long, this led to 64,000 total samples per note. This initial network held no hidden layers, and simply had a dimensional reduction of 64,000 to 11, estimating a probability that the presented noise belonged to each classification. We then use one-hot encoding to find the highest probability and output that as the final result for each audio file sent through the model. This allows us to convert the model from a multiclass classification to a binary classification. For testing and changing parameters, we also included the top 3 highest likely categories to ensure the model was training in the correct direction.'
fin.write(methods3)
methods4 = 'Building off of our initial model, we then used Tensorflow to create a much more sophisticated CNN taking in a 2 dimensional input, this time using the spectrograms prepared earlier in the preprocessing stage. This second model now works more like a traditional 2D CNN where each input maps to each pixel of the spectrogram. We’ve also added more layers to this network as well. We’ve added a ReLU layer to add nonlinearity to the model, added dropout to aid in flattening, then we finally use softmax for our classifications. Though this process is more computationally expensive, using 25 million parameters over the original model’s 5 million, this model trains with fewer and smaller batches than our initial model, allowing us to train more efficiently.'
fin.write(methods4)

fin.image("final_report/image3.png", caption='Figure 1: Diagram portraying the layer structure of the CNN. Note that the end of the CNN is 3 unlabelled layers which include two dense layers and a single flatten layer to output the probability of the sound being each of the 11 defined categories.', use_column_width=True)

methods5 = 'We have also sought to optimize the model through implementing SGD with momentum, allowing our model to escape the traps of local minima and have a better chance of locating a higher maxima to achieve a greater accuracy. We have kept the categorical cross entropy, or softmax loss, to calculate loss for the CNN.'
fin.write(methods5)

# Results and Discussion:
fin.header('Results and Discussion')
results_and_discussion1 = 'To evaluate our models, we compared the predicted instrument labels to the ground truth labels in the NSynth test set. The models output a probability distribution over the 11 instrument types. To generate the final predicted label, we selected the instrument class with the highest probability.'
fin.write(results_and_discussion1)

fin.image("final_report/image6.png", caption='Figure 2: Categorical entropy loss over 10 epochs for the 1D Model', use_column_width=True)

results_and_discussion2 = 'Categorical cross-entropy loss was used to optimize the models, measuring the divergence between the predicted class probabilities and the ground truth one-hot labels. The entropy loss plotted across training epochs in figure 2 displays consistent gains in performance over training. Earlier in training, we can see steeper reductions in loss, however, the rate of improvement gradually levels off in subsequent rounds as more subtle features needed for fine-grained classification prove more difficult to extract from the audio signals alone.'
fin.write(results_and_discussion2)

results_and_discussion3 = 'The consistent downward slope shows the 1D architecture continues extracting some additional nuances through the full training duration that incrementally boost classification precision. But the plateauing of this trajectory shows the difficulty that accompanies translating raw audio waveforms into intricate instrument predictions. In addition, this may be a signal of overfitting in our model.'
fin.write(results_and_discussion3)

fin.image("final_report/image5.png", caption='Figure 3: Confusion matrix, recall, precision, and values needed for accuracy calculation for 1D model', use_column_width=True)

fin.image("final_report/image7.png", caption='Figure 4: Performance metrics for 1D Model, taken after training', use_column_width=True)

results_and_discussion4 = 'The accuracy for the model is 88.5%,  meaning that overall the model is correctly classifying instruments in about 9 out of 10 test samples. Looking at the confusion matrix in figure 3, we see most instruments have at least a few correct predictions on the diagonal, indicating the model has the capability to classify those instrument types. In addition, the high specificity of 93.7% suggests that when the model predicts negative classes, it is usually correct.'
fin.write(results_and_discussion4)

results_and_discussion5 = 'However, the confusion matrix also shows that there is significant confusion between certain classes like 1 vs 5, 2 vs 9, and 4 vs 5 vs 6. This is also reflected in the precision, recall and F1 scores (figure 4) that are at 37%, showing that the 1D model struggles to detect positive cases and has lots of false positives. To further improve, additional training data for some of the weaker classes could help differentiation. Targeted data augmentation and architecture adjustments focused on points of confusion can continue advancing accuracy. Potentially, the monitoring of per-class metrics will indicate if there is a need for instrument-specific tuning.'
fin.write(results_and_discussion5)

fin.image("final_report/image1.png", caption='Figure 5: Categorical  entropy loss', use_column_width=True)

results_and_discussion6 = 'The categorical entropy loss over 10 epochs for the 1D and 2D models, seen in figure 5, provides useful insights into the comparative learning trajectories and relative performances of the models. The 1D model exhibits a progressive decline in entropy from 1.0 down to 0.15 over the course of the 10 epochs, signaling steady improvement in classification certainty as the model trains on the data. However, the rate of improvement slows after the first couple epochs.'
fin.write(results_and_discussion6)

fin.image("final_report/Spectrogram.png", caption='Figure 6: An example of the visual spectrogram generated from one of the 1D waveforms', use_column_width=True)

results_and_discussion7 = 'In contrast, the 2D model displays rapid entropy reduction in the initial epochs, quickly stabilizing around 0.197 by the third epoch. The initial steep drop and quick flattening out suggests the 2D representation enables the model to swiftly grasp visual patterns useful for differentiation. This is seen as a result of supplying the spectrogram images as input to the 2D model, as opposed to just the raw audio waveforms. The 2D CNN was able to learn more complex features as a result, improving the ability for faster and more effective learning.  Meanwhile, the consistent downward trend for the 1D model implies potential for further tuning improvements across the full 10 epochs. In addition, the lower final entropy loss by the 1D model by the end of the 10 epochs shows the potential for a better distinction between classes of instruments in comparison to the 2D model, and reasons for this finding will be further explained through the identification of limitations for the training of the 2D model.'
fin.write(results_and_discussion7)

fin.image("final_report/image4.png", caption='Figure 7: Confusion matrix, recall, precision, and values needed for accuracy calculation for 2D model', use_column_width=True)

fin.image("final_report/image2.png", caption='Figure 8: Performance metrics for 2D Model, taken after training', use_column_width=True)

results_and_discussion8 = 'The confusion matrix and metrics indicate the 2D model struggles to accurately classify some instrument types. The overall accuracy is 85.5%, which shows that the majority of test samples are being classified correctly and that the model does a good job at correctly identifying negatives, but there are also low values across precision (20%), recall (20%), and F1-score (20%), shown in Figure 8, showing that the model struggles to identify some specific classes of instruments. In this batch of 10, the model was able to correctly identify 2 true positives, but failed to identify the other 8 instruments, shown in figure 7.'
fin.write(results_and_discussion8)

results_and_discussion9 = 'There are various reasons for these deficiencies in the model’s capabilities. First, the small batch size limits the breadth of training examples, making generalization more difficult. More varied data and larger batches would better match real-world distributions and variety, improving learning. Second, the complexity and subtle detail differentiating instruments makes extracting distinctive features quite challenging. During preprocessing, we reduced the number of features to 300 in order to reduce dimensionality, which  decreased the timbre, potentially causing an instrument to sound very different when put into our model, resulting in an inaccurate classification. Finally, while a 2D representation captures certain visual patterns, the model has insufficient depth to identify finer grained nuances between classes based only on this encoding. More convolutional and dense layers could strengthen feature extraction capabilities.'
fin.write(results_and_discussion9)

results_and_discussion10 = 'Expanding batch size, model depth and parameters to handle the problem complexity, as well as incorporating additional training modalities beyond 2D encoding would help enhance classification capabilities, especially since the nuances of instrument sounds can easily result in misclassification.'
fin.write(results_and_discussion10)

#Additional Limitations
fin.header('Additional Limitations')

limitations = 'In the previous checkpoint, we identified an initiative to reduce the time needed for training by reducing the dimensionality with PCA. However, there were some drawbacks that we found in implementing this for our 2D CNN. The main issue, that was previously discussed, was the reduction of dimensionality. PCA attempts to preserve the highest variance components in a lower dimensional space. But those lower variance patterns still may contain meaningful signals for differentiating instrumental classes. For audio modeling, subtle lower energy signatures could still have essential predictive capabilities for differentiating classes of musical instruments. Eliminating those under the assumptions behind PCA could inadvertently impair model performance by losing difficult to perceive, but critical signals from the inputs.'
fin.write(limitations)

# Conclusion
fin.header('Conclusion')
conclusion = 'This exploration of deep convolutional neural networks shows promising capabilities and challenges in accurate musical instrument recognition. While overall accuracy for both the 1D and 2D model were high, precision and recall deficiencies exposed limitations in classifying specific instruments. This signals the need for expanded and more varied training data, increased model complexity, and targeted tuning on easily confused classes, in order to achieve precise identification. Although real-world robustness is still a goal to work towards, we have seen some promising initial successes from these models.'
fin.write(conclusion)

fin.header('Timeline')
fin.image("gantchart.png", caption='Gantt Chart', use_column_width=True)
fin.markdown('[Timeline Link](https://docs.google.com/spreadsheets/d/14jc8INAYUF7UpRMh5FNqtHCsqTxM9mZr/edit?usp=sharing&ouid=106392410827909927826&rtpof=true&sd=true)')

# A contribution table with all group members’ names that explicitly provides the contribution of each member in preparing the project task. This part does NOT count towards word limit.
fin.header('Contribution Table')
fin.image("final_report/contribution.png", caption='Contribution Table', use_column_width=True)
fin.markdown('[Contribution Link](https://docs.google.com/spreadsheets/d/1ErVX2eNvhlxeY7ajNCB5KscghvE8iW3X/edit?usp=sharing&ouid=106392410827909927826&rtpof=true&sd=true)')

fin.header('Presentation')
fin.video('https://www.youtube.com/watch?v=j2-8w52dzVs')
fin.markdown('[Video Link](https://www.youtube.com/watch?v=j2-8w52dzVs)')
# fin.markdown('[Slides Link](https://docs.google.com/presentation/d/1jdO9zwQNaCHwXmeDMgxfeWpnSxTNvJAb6rdTWqieLiE/edit?usp=sharing)')







# MIDTERM REPORT
mid.title('Instrument Identification using Machine Learning Midterm Report')

mid.write('By Project Group 13: Ashley Cain, Ethan Haarer, Keshav Jagannath, Matthew Landis, Yash Saraiya')

# Introduction/Background: 
mid.header('Introduction')
introduction1 = 'Automatic musical instrument recognition from audio has been widely studied, with deep learning approaches emerging as state-of-the-art. Prior convolutional neural network (CNN) models have shown promising results on polyphonic instrument identification in both classical and synthesized music (Han, Kim, & Lee, 2016; Li, Qian, & Wang, 2015). Computer vision techniques have also been applied for music classification from spectrograms (Ke, Hoiem & Sukthankar, 2005).'
mid.write(introduction1)
introduction2 = 'In this work, we explore deep CNNs for monophonic instrument recognition using the NSynth dataset (NSynth Dataset, n.d.), which provides over 300,000 musical notes from 1000 instruments. We frame this as a multi-class classification problem across 11 instrument categories including keyboard, guitar, bass, and brass. Using 1D and 2D CNN architectures, we classify short solo instrument excerpts based on the raw waveforms and derived spectrograms respectively. We analyze factors like training convergence, accuracy, and confusion patterns. The goal is to develop an optimized deep learning approach to timbre-based instrumentation.'
mid.write(introduction2)

# References:
mid.header('References')
mid.markdown(
"""
- Han, Y., Kim, J., & Lee, K. (2016). Deep convolutional neural networks for predominant instrument recognition in polyphonic music. IEEE/ACM Transactions on Audio, Speech, and Language Processing, 25(1), 208–221. https://doi.org/10.1109/taslp.2016.2623355
- Ke, Y., Hoiem, D. & Sukthankar, R. (2005). Computer vision for music identification. In 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05) (Vol. 1, pp. 597-604). IEEE. https://doi.org/10.1109/CVPR.2005.105
- Li, P., Qian, J., & Wang, T. (2015). Automatic instrument recognition in polyphonic music using convolutional neural networks. arXiv preprint arXiv:1511.05520.
- NSynth Dataset. (n.d.). https://magenta.tensorflow.org/datasets/nsynth
""")

# Problem definition: 
mid.header('Problem Definition')
problem_definition = 'The main problem we are aiming to solve is the ability to identify a musical instrument through an audio file sample. One main issue for many artists and music producers and editors is the inability for them to edit or isolate specific tracks of popular songs that they want to edit, remix, sample or alter to meet their needs. This Ai model could potentially be adapted to identify instruments and their notes and aid artists and production software in isolating instruments and tracks from the rest of the song so that songs may be deconstructed to allow for editing and creating new songs based on preexisting tracks and popular media. This could further be used to generate sheet music based on a song playing, especially if that song is lost media and the original files are lost. This model could then reconstruct that music to be learned and reused in the future.'
mid.write(problem_definition)

# Methods:
mid.header('Methods')
methods1 = 'Before we send the data through our models, we first preprocess them by normalizing the frequency values in each randomly selected audio file from the dataset, mapping the original range of [-1, 1] to [0, 1]. Then we run the PCA algorithm to reduce the features of the dataset to make the training easier. Both of these are primarily done before training in our first model, however for our second iteration of the model, we add the additional step of converting the audio files to a spectrogram representation.'
mid.write(methods1)
methods2 = 'For our batch sizes, we randomly shuffle the training and test points between training to better test accuracy of the model.'
mid.write(methods2)
methods3 = 'The first model we constructed was a convolutional neural network taking in a one-dimensional waveform, built within Tensorflow on Google Colab. This model took in each sample of a note as a single input, and with notes in the n-synth dataset having 16,000 samples per second and each being 4 seconds long, this led to 64,000 total samples per note. This initial network held no hidden layers, and simply had a dimensional reduction of 64,000 to 11, estimating a probability that the presented noise belonged to each classification. We then use one-hot encoding to find the highest probability and output that as the final result for each audio file sent through the model. This allows us to convert the model from a multiclass classification to a binary classification. For testing and changing parameters, we also included the top 3 highest likely categories to ensure the model was training in the correct direction.'
mid.write(methods3)
methods4 = 'Building off of our initial model, we then used Tensorflow to create a much more sophisticated CNN taking in a 2 dimensional input, this time using the spectrograms prepared earlier in the preprocessing stage. This second model now works more like a traditional 2D CNN where each input maps to each pixel of the spectrogram. We’ve also added more layers to this network as well. We’ve added a ReLU layer to add nonlinearity to the model, added dropout to aid in flattening, then we finally use softmax for our classifications. Though this process is more computationally expensive, using 25 million parameters over the original model’s 5 million, this model trains with fewer and smaller batches than our initial model, allowing us to train more efficiently.'
mid.write(methods4)

# Results and Discussion:
mid.header('Results and Discussion')
results_and_discussion1 = 'To evaluate our models, we compared the predicted instrument labels to the ground truth labels in the NSynth test set. The models output a probability distribution over the 11 instrument types. To generate the final predicted label, we selected the instrument class with the highest probability.'
mid.write(results_and_discussion1)

mid.image("Figure1.png", caption='Figure 1: Categorical entropy loss over 10 epochs for the 1D Model', use_column_width=True)

results_and_discussion2 = 'Categorical cross-entropy loss was used to optimize the models, measuring the divergence between the predicted class probabilities and the ground truth one-hot labels. As seen in the training loss curve (Figure 1), the model was able to gradually minimize categorical cross-entropy loss over 10 epochs, indicating the model was steadily learning to produce probabilities closer to the true instrument labels. However, the validation loss leveled off before reaching zero, a sign of potential overfitting. There is room to further reduce loss with additional training data and regularization techniques.'
mid.write(results_and_discussion2)

mid.image("Figure2.png", caption='Figure 2: Confusion matrix, recall, precision, and values needed for accuracy calculation', use_column_width=True)

results_and_discussion3 = "When we evaluate accuracy based on correct identification within the first guess, our 1D CNN model for direct audio sample classification is able to achieve a 83.6% accuracy on the NSynth test set, with a batch size of 10,000. However it should be noted that this evaluation is influenced by the high number of true negative values that we have in the accuracy calculation, since the model will always be successful in correctly refuting a large number of the possible instruments, regardless of whether its single prediction is correct. Considering this fact, it may be more appropriate to evaluate the success of our model by looking at the values for recall and specificity (Figure 2). From these values, we can see that there are still things to be improved in our model."
mid.write(results_and_discussion3)

mid.image("Figure3.png", caption='Figure 3: Categorical entropy loss over 10 epochs for the 1D and 2D Model', use_column_width=True)

results_and_discussion4 = 'Transitioning from the 1D to the 2D convolutional neural network architecture dramatically improved model performance. By supplying spectrogram images as input instead of raw audio waveforms, the 2D CNN could learn more complex features for discriminating between instrument types, improving the ability to learn more effectively, as shown by the loss curve (Figure 2), additional convolutional layers continued to reduce loss over 10 epochs.'
mid.write(results_and_discussion4)

mid.image("Figure4.png", caption='Figure 4: An example of the visual spectrogram generated from one of the 1D waveforms', use_column_width=True)

results_and_discussion5 = 'Spectrograms provide a visual representation of the frequency content of audio signals over time. By transforming the raw 1D waveform into a 2D image encoding the time-frequency spectrum, critical qualities of timbre and harmony are revealed (Figure 4). In particular, the resonant frequencies and overtones unique to each instrument manifest as visual patterns in the spectrogram, which can be used by the model to identify musical patterns in the spectrogram that can be applied to further instrument identification from more qualities, allowing it to become more accurate. '
mid.write(results_and_discussion5)

mid.image("Figure5.png", caption='Figure 5: An example of the internal feature map used to create the spectrogram', use_column_width=True)

results_and_discussion6 = 'Qualitatively, we can see the 2D CNN learned a better representation by visualizing its internal feature maps (Figure 5). The first convolutional layer filters detect low-level patterns in the spectrograms, while deeper layers respond to higher-level features like harmonics and timbre. In direct comparison, the 1D model lacks this hierarchical feature learning.'
mid.write(results_and_discussion6)

#Next Steps
mid.header('Next Steps')
next_steps1 = 'To further improve the accuracy of our 2D CNN model for musical instrument classification, our next steps will focus on refining the neural network architecture and enhancing the spectrogram preprocessing pipeline.'
mid.write(next_steps1)

next_steps2 = 'Currently, our CNN model outputs predictions directly from the convolutional layers to the final 11 instrument categories. We hope to add hidden dense layers between the convolutional feature extraction layers and the output layer, which will allow for more gradual learning of the complex mappings from spectrogram features to instrument labels and enable better generalization.'
mid.write(next_steps2)

next_steps3 = 'We also plan to expand the preprocessing steps applied to the audio data before generating the spectrograms. In order to reduce the time needed for training, we will implement dimensionality reduction with PCA prior to the short-time Fourier transform and spectrogram creation.'
mid.write(next_steps3)

# Conclusion
mid.header('Conclusion')
conclusion = 'In conclusion, converting the audio waveforms to spectrograms and supplying these to a deeper CNN architecture led to improved musical instrument classification. The 2D approach models the frequency characteristics well, while the 1D CNN lacks the capacity to capture timbre details. With further enhancements to the spectrogram-based model, we aim to achieve a higher test accuracy in classifying the dataset.'
mid.write(conclusion)









# PROPOSAL
pro.title('Instrument Identification using Machine Learning Proposal')
pro.write('By Project Group 13: Ashley Cain, Ethan Haarer, Keshav Jagannath, Matthew Landis, Yash Saraiya')

# Introduction/Background: 
#   A quick introduction of your topic and mostly literature review of what has been done in this area. 
#   You can briefly explain your dataset and its features here too.
pro.header('Introduction')
introduction = 'The goal of this project is to identify the instrument being played from a given snippet of audio. This concept has been explored in several research articles as part of bigger audio projects, such as clustering of instrument categories from a multi-instrument recording. We will be using an established dataset of audio files such as NSynth to train a convolutional neural network to fulfill this goal. This dataset also contains classifiers such as "acoustic", "synthetic", and "electronic", as well as "attack" and "decay" time. These are all useful features to have when describing a snippet of audio. NSynth already has its audio files encoded down to integer arrays, and comes with a wide featureset that we will be able to use for training.'
pro.write(introduction)

# Problem definition: 
#   Why there is a problem here or what is the motivation of the project?
pro.header('Problem Definition')
problem_definition = 'AI classification in the audio realm is a relatively new research field compared to applications such as computer vision. Single instrument classification is an intermediary towards a host of other applications. Companies like "Izotope" are currently interested in developing mixing and mastering virtual assistants based on the ability to classify waveforms into buckets of instrument categories. It would also be useful for individuals to classify instrument details from rehearsal recordings, auditions, or performances for further review.'
pro.write(problem_definition)

# Methods: 
#   What algorithms or methods are you going to use to solve the problems. 
#   (Note: Methods may change when you start implementing them which is fine). 
#   Students are encouraged to use existing packages and libraries (i.e. scikit-learn) instead of coding the algorithms from scratch.
pro.header('Methods')
methods = """
We are using the NSynth library and its included feature set in order to train a convolutional neural network. 
We decided upon CNNs by reading different research papers that extract the instruments used from polyphonic audio files and found this as the most efficient way for our application. 
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
pro.write(methods)

# Potential results and Discussion:
#   (The results may change while you are working on the project and it is fine; that’s why it is called research). 
#   A good way to talk about potential results is to discuss about what type of quantitative metrics your team plan to use for the project (i.e. ML Metrics).
pro.header('Potential Results and Discussion')
potential_results_and_discussion = """
We will be looking for the success rate for which the model can identify the instrument being played in a given audio file. 
This testing will be done on a reserved portion of the dataset, and we are hoping to achieve a success rate of 95% or higher. 
When applying the model to real world scenarios, we will be looking at how the model can extract the music from any background noise.
"""
pro.write(potential_results_and_discussion)

# At least three references (preferably peer reviewed). You need to properly cite the references on your proposal. This part does NOT count towards word limit.
pro.header('References')
pro.markdown(
"""
- Yoonchang Han, Jaehun Kim, and Kyogu Lee, “Deep convolutional neural networks for predominant instrument recognition in polyphonic music,” IEEE/ACM Transactions on Audio, Speech, and Language Processing, vol. 25, no. 1, pp. 208–221, 2016.
- Peter Li, Jiyuan Qian, and Tian Wang, “Automatic instrument recognition in polyphonic music using convolutional neural networks,” arXiv preprint arXiv:1511.05520, 2015.
- Yan Ke, D. Hoiem and R. Sukthankar, "Computer vision for music identification," 2005 IEEE Computer Society Conference on Computer Vision and Pattern Recognition (CVPR'05), San Diego, CA, USA, 2005, pp. 597-604 vol. 1, doi: 10.1109/CVPR.2005.105.
""")

# Add proposed timeline from start to finish and list each project members’ responsibilities. Fall and Spring semester sample Gantt Chart. This part does NOT count towards word limit.
pro.header('Timeline')
pro.image("gantchart.png", caption='Gantt Chart', use_column_width=True)
pro.markdown('[Timeline Link](https://docs.google.com/spreadsheets/d/14jc8INAYUF7UpRMh5FNqtHCsqTxM9mZr/edit?usp=sharing&ouid=106392410827909927826&rtpof=true&sd=true)')

# A contribution table with all group members’ names that explicitly provides the contribution of each member in preparing the project task. This part does NOT count towards word limit.
pro.header('Contribution Table')
pro.image("contributiontable.png", caption='Contribution Table', use_column_width=True)
pro.markdown('[Contribution Link](https://docs.google.com/spreadsheets/d/1ErVX2eNvhlxeY7ajNCB5KscghvE8iW3X/edit?usp=sharing&ouid=106392410827909927826&rtpof=true&sd=true)')

# A checkpoint to make sure you are working on a proper machine learning related project. You are required to have your dataset ready when you submit your proposal. You can change dataset later. However, you are required to provide some reasonings why you need to change the dataset (i.e. dataset is not large enough because it does not provide us a good accuracy comparing to other dataset; we provided accuracy comparison between these two datasets). The reasonings can be added as a section to your future project reports such as midterm report.
pro.header('Checkpoint')
checkpoint = """
  This is a proper machine learning project because we are using ML libraries and framework such as Librosa and PyTorch in order to create a multi-class classification model, which is a supervised model.
  We will be extracting features of the from the audio files to train the models on test data. Then we will train a convolutional neural network and evaluate the success rate in a quantifiable way in order to evaluate success of the model. These are all key characteristics of a machine learning project. 
"""
pro.write(checkpoint)

# Your group needs to submit a presentation of your proposal. Please provide us a public link which includes a 3 minutes recorded video. I found that OBS Studio and GT subscribed Kaltura are good tools to record your screen. Please make your visuals are clearly visible in your video presentation.
# 3 MINUTE is a hard stop. We will NOT accept submissions which are 3 minutes and one second or above. Conveying the message easily while being concise is not easy and it is a great soft skill for any stage of your life, especially your work life.
pro.header('Presentation')
pro.video('https://youtu.be/HjY5fkp9aiw')
pro.markdown('[Video Link](https://youtu.be/HjY5fkp9aiw)')
pro.markdown('[Slides Link](https://docs.google.com/presentation/d/1jdO9zwQNaCHwXmeDMgxfeWpnSxTNvJAb6rdTWqieLiE/edit?usp=sharing)')

# pro.write('Graded Word Count: ' + str(len((introduction + problem_definition + methods + potential_results_and_discussion).split())))

