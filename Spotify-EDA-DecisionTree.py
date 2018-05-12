# file converted using
# jupyter nbconvert Spotify-EDA-DecisionTree.ipynb --to=python


# coding: utf-8

# In[57]:


# install dependencies listed in the file requirements.txt 
# (extracted with pip3 freeze > requirements.txt and edited to remove redundancies)
import subprocess
subprocess.call(["pip3", "install", "-r", "requirements.txt"])

# contains DataFrame, Series (if not installed, run pip3 install pandas)
import pandas as pd 
import numpy as np # scientific computer package - contains Array, matrix and linear algebra

# sklearn is the machine learning package (if not installed, run pip3 install scipy, then sklearn)
from sklearn import tree 
from sklearn.tree import DecisionTreeClassifier 
from sklearn.model_selection import train_test_split # method to easily split large data set

# makes nice charts (if not installed, run pip3 install matplotlib)
from matplotlib import pyplot as plt 
# visualization library, makes charts even nicer (if not installed, run pip3 install seaborn)
import seaborn as sns 

import io

# this is needed because misc.imread is deprecated
import imageio

# below needs this to run on terminal:  brew install graphviz
from sklearn.tree import export_graphviz
import graphviz

import pydotplus

# this needs:  pip3 install pillow
from scipy import misc

get_ipython().run_line_magic('matplotlib', 'inline')


# # Spotify song attributes EDA
# - Import data
# - EDA to visualize data and observe structure 
# - Train a classifier (Decision Tree) 
# - Predict target using the trained classifier

# In[2]:


data = pd.read_csv( 'data/data.csv' )


# In[3]:


type( data )


# In[4]:


data.describe() # display statistics from the data


# In[5]:


data.head() # first 5 rows


# In[6]:


data.info() # columns (features), datatypes, size of data
# see more about each feature at https://developer.spotify.com/web-api/get-audio-features/
# or rather https://web.archive.org/web/20170322073729/https://developer.spotify.com/web-api/get-audio-features/


# In[7]:


train, test = train_test_split( data, test_size = 0.15 ) # use 15% of data for testing


# In[8]:


print( "Training size: {}; Test size: {}".format( len( train ), len( test ) ) )


# In[9]:


train.shape # shows number of rows and columns


# In[10]:


# custom color palette
green_red = [ "#195B00", "red" ]
palette = sns.color_palette( green_red )
sns.set_palette( palette )
sns.set_style( "white" )


# In[11]:


# query data to get tempo for rows where the target is 1 (user liked song)
positive_tempo = data[ data[ 'target' ] == 1 ][ 'tempo' ] 
# query data to get tempo for rows where the target is 0 (user disliked song)
negative_tempo = data[ data[ 'target' ] == 0 ][ 'tempo' ] 

# create graph with size (x,y) and title
fig = plt.figure( figsize = ( 15, 6 ) )
plt.title( "Song Tempo Like / Dislike Distribution" )

# histogram:  alpha make it translucent so you can see both positive/negative, divide data in 30 groups
positive_tempo.hist( alpha = 0.7, bins = 30, label = 'positive' )
negative_tempo.hist( alpha = 0.7, bins = 30, label = 'negative' )

plt.legend( loc = "upper right" )


# In[12]:



# do the same for other features
pos_dance = data[ data[ 'target' ] == 1 ][ 'danceability' ] 
neg_dance = data[ data[ 'target' ] == 0 ][ 'danceability' ] 

pos_duration = data[ data[ 'target' ] == 1 ][ 'duration_ms' ] 
neg_duration = data[ data[ 'target' ] == 0 ][ 'duration_ms' ] 

pos_energy = data[ data[ 'target' ] == 1 ][ 'energy' ] 
neg_energy = data[ data[ 'target' ] == 0 ][ 'energy' ] 

pos_instrumentalness = data[ data[ 'target' ] == 1 ][ 'instrumentalness' ] 
neg_instrumentalness = data[ data[ 'target' ] == 0 ][ 'instrumentalness' ] 

pos_loud = data[ data[ 'target' ] == 1 ][ 'loudness' ] 
neg_loud = data[ data[ 'target' ] == 0 ][ 'loudness' ] 

pos_speechy = data[ data[ 'target' ] == 1 ][ 'speechiness' ] 
neg_speechy = data[ data[ 'target' ] == 0 ][ 'speechiness' ] 

pos_key = data[ data[ 'target' ] == 1 ][ 'key' ] 
neg_key = data[ data[ 'target' ] == 0 ][ 'key' ] 

pos_acoustic = data[ data[ 'target' ] == 1 ][ 'acousticness' ] 
neg_acoustic = data[ data[ 'target' ] == 0 ][ 'acousticness' ] 

pos_valence = data[ data[ 'target' ] == 1 ][ 'valence' ] 
neg_valence = data[ data[ 'target' ] == 0 ][ 'valence' ] 

# create another graph to reveal bias and structure of the data
fig2 = plt.figure( figsize = ( 15, 15 ) )

# Danceability:  add smaller graph inside the big graph
ax3 = fig2.add_subplot( 331 )
ax3.set_xlabel( 'Danceability' )
ax3.set_ylabel( 'Count' )
ax3.set_title( 'Song Danceability Like Distribution' )
pos_dance.hist( alpha = 0.5, bins = 30 )
neg_dance.hist( alpha = 0.5, bins = 30 )
# Duration
ax4 = fig2.add_subplot( 332 )
ax4.set_xlabel( 'Duration' )
ax4.set_ylabel( 'Count' )
ax4.set_title( 'Song Duration Like Distribution' )
pos_duration.hist( alpha = 0.5, bins = 30 )
neg_duration.hist( alpha = 0.5, bins = 30 )
# Energy
ax5 = fig2.add_subplot( 333 )
ax5.set_xlabel( 'Energy' )
ax5.set_ylabel( 'Count' )
ax5.set_title( 'Song Energy Like Distribution' )
pos_energy.hist( alpha = 0.5, bins = 30 )
neg_energy.hist( alpha = 0.5, bins = 30 )
# Instrumentalness
ax6 = fig2.add_subplot( 334 )
ax6.set_xlabel( 'Instrumentalness' )
ax6.set_ylabel( 'Count' )
ax6.set_title( 'Song Instrumentalness Like Distribution' )
pos_instrumentalness.hist( alpha = 0.5, bins = 30 )
neg_instrumentalness.hist( alpha = 0.5, bins = 30 )
# Loudness
ax7 = fig2.add_subplot( 335 )
ax7.set_xlabel( 'Loudness' )
ax7.set_ylabel( 'Count' )
ax7.set_title( 'Song Loudness Like Distribution' )
pos_loud.hist( alpha = 0.5, bins = 30 )
neg_loud.hist( alpha = 0.5, bins = 30 )
# Speechiness
ax8 = fig2.add_subplot( 336 )
ax8.set_xlabel( 'Speechiness' )
ax8.set_ylabel( 'Count' )
ax8.set_title( 'Song Speechiness Like Distribution' )
pos_speechy.hist( alpha = 0.5, bins = 30 )
neg_speechy.hist( alpha = 0.5, bins = 30 )
# Key
ax6 = fig2.add_subplot( 337 )
ax6.set_xlabel( 'Key' )
ax6.set_ylabel( 'Count' )
ax6.set_title( 'Song Key Like Distribution' )
pos_key.hist( alpha = 0.5, bins = 30 )
neg_key.hist( alpha = 0.5, bins = 30 )
# Acousticness
ax7 = fig2.add_subplot( 338 )
ax7.set_xlabel( 'Acousticness' )
ax7.set_ylabel( 'Count' )
ax7.set_title( 'Song Acousticness Like Distribution' )
pos_acoustic.hist( alpha = 0.5, bins = 30 )
neg_acoustic.hist( alpha = 0.5, bins = 30 )
# Valence
ax8 = fig2.add_subplot( 339 )
ax8.set_xlabel( 'Valence' )
ax8.set_ylabel( 'Count' )
ax8.set_title( 'Song Valence Like Distribution' )
pos_valence.hist( alpha = 0.5, bins = 30 )
neg_valence.hist( alpha = 0.5, bins = 30 )



# In[50]:


c = DecisionTreeClassifier( min_samples_split = 100 )

# Accuracy using Decision Tree:  67.0 % for min samples = 150
# Accuracy using Decision Tree:  72.9 % for min samples = 100
# Accuracy using Decision Tree:  70.0 % for min samples = 50

features = [ "danceability", "loudness", "valence", "energy", "instrumentalness", "acousticness", "key", "speechiness", "duration_ms" ]
X_train = train[ features ]
Y_train = train[ "target" ]
X_test = test[ features ]
Y_test = test[ "target" ]


# In[51]:


dt = c.fit( X_train, Y_train )


# In[52]:


def show_tree( tree, features, path ):
    f = io.StringIO() 
    export_graphviz( tree, out_file=f, feature_names=features ) 
    pydotplus.graph_from_dot_data( f.getvalue() ).write_png( path )
    #img = misc.imread( path ) 
    img = imageio.imread( path )
    plt.rcParams[ "figure.figsize" ] = ( 20, 20 ) 
    plt.imshow( img )


# In[53]:


show_tree( dt, features, 'dec_tree_01.png' )


# In[56]:


Y_pred = c.predict( X_test )
from sklearn.metrics import accuracy_score
score = accuracy_score( Y_test, Y_pred ) * 100
print( "Accuracy using Decision Tree: ", round( score, 1 ), "%" )

