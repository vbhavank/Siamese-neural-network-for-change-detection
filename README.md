# siamese_rit_change

   `python3 -m pip install --index-url https://test.pypi.org/simple/ siamese-rit-change`
   
   `python3`
   
   `import siamese_rit_change`
   

We present a patch-based algorithm for detecting structural changes in satellite imagery using a Siamese neural network. The two channels of our Siamese network are based on the VGG16 architecture with shared weights. Changes between the target and reference images are detected with a fully connected decision network that was trained on DIRSIG simulated samples and achieved a high detection rate. Alternatively, a change detection approach based on Euclidean distance between deep convolutional features achieved very good results with minimal supervision.

Dependencies required
   1)Tensorflow
   2)Keras with tensorflow background
   3)Numpy
   4)Keras.utils
   5)numpy_utils
   6)Python 2.7

Data
   Few sample data in is present in image pairs
   Unzip the file
   Names starting with AChip has a corresponding ANeg these are the the pairs
   for example
      AChip1,ANeg1 becomes a pair
      AChip2.ANeg2 becomes a pair

Testing 
   Siamese_predict.py is used for testing
   open command line and type
   python Siamese_predict.py
   It will ask for 1st image chip choose the image pairs as described above 
   Do the same for 2nd image chip
Output will be in command line Change or No change
