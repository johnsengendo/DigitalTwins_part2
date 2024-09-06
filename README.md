# Network_Digital_Twins part II.
This repository extends Repository [Video_server](https://github.com/johnsengendo/Video_server) and [Web_server](https://github.com/johnsengendo/Web_server), where we aim to replicate a pcap file into a digital twin.

The process involves the following steps:
- Generating the pcap file.
- Analyzing the pcap file to extract meaningful data.
- Predicting Traffic: Performing traffic predictions based on the analyzed pcap file data.
- Replicating in a Digital Twin: Applying the predictions to a digital twin to generate another pcap file, which should colorate with the original pcap file.

Repository structure:

- data/: Contains pcap files and etracted data from the pcap files.
- analysis/:  Contains scripts and notebooks for processing pcap files to extract out useful features.
- predictions/: Pre-trained models and scripts for traffic prediction.
- digital_twin/: Implementation of the digital twin replication.

### Pysical Twin with Video Streaming application with Dynamic Network conditions.

The Pysical_Twin folder contains the general network setup where the [topology script](https://github.com/johnsengendo/DigitalTwins_part2/blob/main/Physical_Twin/network-topology-script.py) sets up a video streaming application using Mininet and Containernet, simulating dynamic network conditions by varying bandwidth and delay of the bottleneck link. The setup includes server and client containers for video streaming, with network properties (BW and Delay) changing every 120 seconds.

Key Features:
- **Dynamic network simulation:** Involves adjusting bandwidth and delay to create variability in the network and as well to collect more data. Below is a table of different values used:
![data](https://github.com/johnsengendo/DigitalTwins_part2/blob/main/Images/Screenshot%202024-09-04%20103539.png)
- **Automated testing:** Option to enable autotest mode for automated topology testing.
- **Integration with Mininet:** Utilizing Mininet and ComNetsEmu for network simulation and container management.
- Pcap_file captured: Beloww is a pcap file captured after streaming the a vedio application for 30 minutes.
![data](https://github.com/johnsengendo/DigitalTwins_part2/blob/main/Images/Screenshot%202024-09-04%20112810.png)

- Extracted features (packets_per_second) from the Pcap_file.
![data](https://github.com/johnsengendo/DigitalTwins_part2/blob/main/Images/Screenshot%202024-09-04%20113955.png)

## Digital Twin Network simulation.

The [Digital_Twin](https://github.com/johnsengendo/DigitalTwins_part2/blob/main/Digital_Twin/digital_twin.py) emulates a "digital twin" network that mirrors the traffic patterns observed in a physical twin.
### Time-Series Prediction with CNN

Within the predictions folder, different algorithims are tried out to generate predictions one of the promising is a [time_series_cnn_prediction](https://github.com/johnsengendo/DigitalTwins_part2/blob/main/predictions/time_series_cnn_prediction.py) that leverages Convolutional Neural Networks (CNNs) for predicting time-series data, specifically focusing on packet-per-second metrics.  
It preprocesses the data, trains multiple models with different window sizes and forecast horizons, and visualizes the results.  
Key dependencies include `TensorFlow`, `Keras`, and `Scikit-learn`

## Results showing the different results.
![alt text](https://github.com/johnsengendo/DigitalTwins_part2/blob/main/Images/Screenshot%202024-09-06%20131235.png)
## Below as well are curves showing the predictions curves vs the true values.
## Window_size = 360 seconds and ahead = 360 seconds
![alt text](https://github.com/johnsengendo/DigitalTwins_part2/blob/main/Images/Screenshot%202024-09-06%20130413.png)

## Window_size = 300 seconds and ahead = 60 seconds
![alt text](https://github.com/johnsengendo/DigitalTwins_part2/blob/main/Images/Screenshot%202024-09-06%20130638.png)
## Window_size = 120 seconds and ahead = 4 seconds

![alt text](https://github.com/johnsengendo/DigitalTwins_part2/blob/main/Images/Screenshot%202024-09-06%20130750.png)
## System design.
![alt text](https://github.com/johnsengendo/DigitalTwins_part2/blob/main/Images/Image.jpg)
