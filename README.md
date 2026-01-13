# Vessel Trajectory Prediction Using AIS Data with LSTM

## Project Overview

This project focuses on predicting future ship trajectories using Automatic Identification System (AIS) data. AIS provides real-time information about vessel position, speed, course, and navigational status, making it a critical component of modern maritime navigation. However, accurately forecasting future vessel movement remains challenging due to environmental dynamics, irregular sampling, and complex ship behaviors.
To address this, the project uses a Long Short-Term Memory (LSTM) neural network trained on historical AIS sequences to predict future latitude and longitude positions. A novel trajectory-constraining loss function is introduced to ensure geographically realistic predictions.

## Problem Statement

Predicting vessel trajectories from AIS data is difficult due to:

- Highly dynamic maritime environments
- Irregular and noisy AIS transmissions
- Complex temporal dependencies in ship movement
- Lack of geographic constraints in standard prediction models

Many existing approaches fail to enforce realistic movement boundaries, leading to physically implausible trajectory predictions.

## Objectives

The main goals of this project are:

- Predict future ship positions using historical AIS sequences
- Model long-term temporal dependencies using LSTM networks
- Introduce a custom loss function to constrain predictions within realistic geographic bounds
- Improve prediction stability and reliability for maritime navigation use cases

## Dataset Description

The dataset consists of AIS records collected from a Danish port and includes:

- Latitude and longitude
- Speed Over Ground (SOG)
- Course Over Ground (COG)
- Heading
- Ship dimensions and identifiers
- Navigational status

Original AIS timestamps were irregular, ranging from seconds to several seconds apart.

## Data Preprocessing

AIS data is inherently noisy and inconsistent. Extensive preprocessing was applied to make the dataset suitable for sequence-based learning.

### Key preprocessing steps

- Resampled data to uniform one-minute intervals
- Removed duplicate and corrupted records
- Filled missing navigational parameters using interpolation and forward or backward filling
- Grouped and filled static ship attributes using MMSI identifiers
- Filtered only Class A vessels to focus on large commercial ships
- Removed journeys shorter than the required sequence length
- Converted timestamps into hour and minute features
- Encoded categorical variables using numerical labels
  These steps significantly improved data quality and model stability.

## Feature Selection

An ablation study was conducted to identify the most impactful features. The final feature set includes:

- Latitude and longitude
- Speed Over Ground (SOG)
- Course Over Ground (COG)
- Heading
- Navigational status
- Ship type
- Ship length and beam
- Structural offsets A, B, C, D

This combination allows the model to learn both spatial movement patterns and contextual vessel behavior.

## Model Architecture

The predictive model is based on an LSTM architecture designed for sequential AIS data.

### Architecture highlights

- LSTM layer to capture long-term temporal dependencies
- Intermediate fully connected layers for nonlinear feature refinement
- Output layer predicting future position coordinates

The inclusion of intermediate dense layers improves the model’s ability to learn complex motion patterns compared to standard LSTM implementations.

## Custom Trajectory-Constraining Loss

To prevent unrealistic predictions, a custom loss function was introduced.

### Core idea

Convert latitude and longitude targets into east and north displacements in meters
Apply a spatial penalty when predictions fall outside a predefined geographic buffer
Combine standard Mean Squared Error with a distance-based penalty

### Loss composition

- 90 percent standard MSE loss
- 10 percent spatial constraint penalty
- This approach improves geographic plausibility and reduces large trajectory deviations.

## Training Procedure

- AIS data is grouped by ship and sliced into overlapping sequences of 95 time steps
- Sequences are generated using a custom data generator
- Model is trained using backpropagation with gradient stabilization
- Periodic evaluation is performed on validation data to monitor generalization

## Evaluation Metrics

Model performance is evaluated using:

- Mean Squared Error (MSE)
- Spatial constraint penalty
- Trajectory visualizations comparing predicted and actual paths

Both quantitative and qualitative evaluations are used to assess prediction accuracy.

## Results

Due to computational and hardware limitations, training was performed on a very small subset of the full dataset, representing only one day of AIS data from an 18-year archive.

While results are not state-of-the-art, they demonstrate:

- The feasibility of LSTM-based trajectory prediction
- The effectiveness of the custom geographic loss function
- Strong potential for improvement with larger datasets and better resources

## Conclusion

This project demonstrates that LSTM networks, combined with spatially constrained loss functions, can produce more realistic vessel trajectory predictions from AIS data. Although limited by data and computational resources, the approach provides a strong foundation for future research in maritime predictive analytics.

## Future Work

Potential extensions include:

- Training on multi-year AIS datasets
- Incorporating weather and traffic data
- Exploring hybrid models combining statistical and deep learning approaches
- Comparing performance against state-of-the-art trajectory prediction models
