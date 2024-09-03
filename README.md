# Network_Digital_Twins part II
This repository extends Repository [Video_server](https://github.com/johnsengendo/Video_server) and [Web_server](https://github.com/johnsengendo/Web_server), where we aim to replicate a pcap file into a digital twin.

The process involves the following steps:
- Generating the pcap file.
- Analyzing the pcap file to extract meaningful data.
- Predicting Traffic: Performing traffic predictions based on the analyzed pcap file data.
- Replicating in a Digital Twin: Applying the predictions to a digital twin to generate another pcap file, which should colorate with the original pcap file.

Repository structure:

- data/: Contains pcap files and related data.
- analysis/: Scripts and notebooks for analyzing pcap files.
- predictions/: Predictive models and scripts for traffic prediction.
- digital_twin/: Implementation of the digital twin replication.
- 
  ### Time-Series Prediction with CNN

This script [time_series_cnn_prediction](https://github.com/johnsengendo/DigitalTwins_part2/blob/main/predictions/time_series_cnn_prediction.py)leverages Convolutional Neural Networks (CNNs) for predicting time-series data, specifically focusing on packet-per-second metrics.  
It preprocesses the data, trains multiple models with different window sizes and forecast horizons, and visualizes the results.  
Key dependencies include `TensorFlow`, `Keras`, and `Scikit-learn`.

![alt text](https://github.com/johnsengendo/DigitalTwins_part2/blob/main/Images/Image.jpg)
