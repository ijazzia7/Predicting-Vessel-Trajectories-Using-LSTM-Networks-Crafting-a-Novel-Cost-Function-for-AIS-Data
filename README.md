# Vessel Trajectory Prediction Using AIS Data with LSTM

## Project Overview
This project focuses on predicting future ship trajectories using Automatic Identification System (AIS) data. AIS provides real-time information about vessel position, speed, course, and navigational status, making it a critical component of modern maritime navigation. However, accurately forecasting future vessel movement remains challenging due to environmental dynamics, irregular sampling, and complex ship behaviors.
To address this, the project uses a Long Short-Term Memory (LSTM) neural network trained on historical AIS sequences to predict future latitude and longitude positions. A novel trajectory-constraining loss function is introduced to ensure geographically realistic predictions.

