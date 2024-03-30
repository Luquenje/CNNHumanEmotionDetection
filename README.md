# CNNHumanEmotionDetection

## Python Environment
### The dependencies used are in a Anaconda environment yaml file conda_python_env.yaml
### You can import the Anaconda environment or install the dependencies yourself.

## TrainEmotionDetectorVGG16.py & TrainEmotionDetectorResNet50.py
Trains the respective CNN model.
If you want to train using GPU can try uncommenting lines 13-18 provided you have installed the CUDA libraries.

## TestEmotionDetectorVGG16.py & TestEmotionDetectorResNet50.py
Loads a sample video and finds faces using CV2's haarcascade face detection algorithm.
After finding the faces, it uses the model to predict the emotion and displays the predicted result.
Takes in a command line argument in the form of: python TestEmotionDetectorVGG16.py arg
If arg == Webcam, the program opens the default webcam and uses that to test the emotion detector model.
If arg == (some_filepath), the program tries to open a video file, preferably mp4, and uses that to test the emotion detector model.

## EvaluateEmotionDetectorVGG16.py & EvaluateEmotionDetectorResNet50.py
Evaluates the respective models and displays the stats.
