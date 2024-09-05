# Network_Digital_Twins part II.
This repository extends Repository [Video_server](https://github.com/johnsengendo/Video_server) and [Web_server](https://github.com/johnsengendo/Web_server), where we aim to replicate a pcap file into a digital twin.

The process involves the following steps:
- Generating the pcap file.
- Analyzing the pcap file to extract meaningful data.
- Predicting Traffic: Performing traffic predictions based on the analyzed pcap file data.
- Replicating in a Digital Twin: Applying the predictions to a digital twin to generate another pcap file, which should colorate with the original pcap file.

Repository structure:

- data/: Contains pcap files and related data.
- analysis/: Scripts and notebooks for processing pcap files to extract out useful features.
- predictions/: Predictive models and scripts for traffic prediction.
- digital_twin/: Implementation of the digital twin replication.
## Pysical Twin with Video Streaming application with Dynamic Network conditions.

The Pysical_Twin folder contains the general network setup where the [topology script](https://github.com/johnsengendo/DigitalTwins_part2/blob/main/Physical_Twin/network-topology-script.py)  sets up a video streaming application using Mininet and Containernet, simulating dynamic network conditions by varying bandwidth and delay of the bottleneck link in real-time. The setup includes server and client containers for video streaming, with network properties changing every 120 seconds.

Key Features:
- **Dynamic network simulation:** Adjusts bandwidth and delay to create variability in the network and as well to collect more data. Below is a table of different parameters used:
![data](https://github.com/johnsengendo/DigitalTwins_part2/blob/main/Images/Screenshot%202024-09-04%20103539.png)
- **Automated testing:** Option to enable autotest mode for automated topology testing.
- **Integration with Mininet:** Utilizes Mininet and ComNetsEmu for network simulation and container management.
- Pcap_file captured
![data](https://github.com/johnsengendo/DigitalTwins_part2/blob/main/Images/Screenshot%202024-09-04%20112810.png)

- Extracted features from the Pcap_file
![data](https://github.com/johnsengendo/DigitalTwins_part2/blob/main/Images/Screenshot%202024-09-04%20113955.png)

## Digital Twin Network simulation.

The [Digital_Twin](https://github.com/johnsengendo/DigitalTwins_part2/blob/main/Digital_Twin/digital_twin.py) emulates a "digital twin" network that mirrors the traffic patterns observed in a physical twin.
### Time-Series Prediction with CNN

Within the predictions folder, different algorithims a tried out to generate predictions one of the promising is a [time_series_cnn_prediction](https://github.com/johnsengendo/DigitalTwins_part2/blob/main/predictions/time_series_cnn_prediction.py) that leverages Convolutional Neural Networks (CNNs) for predicting time-series data, specifically focusing on packet-per-second metrics.  
It preprocesses the data, trains multiple models with different window sizes and forecast horizons, and visualizes the results.  
Key dependencies include `TensorFlow`, `Keras`, and `Scikit-learn`

### Overview:
- **Physical twin data integration**: Traffic data captured from a physical twin is analysed and stored in a CSV file (`predictions_with_bandwidth.csv`). This data reflects the trasferred packets/Second in the physical twin.
- **Digital twin setup**: The script sets up a Mininet-based "digital twin" that replicates the physical twin's network environment. The topology includes two hosts connected by two switches, designed to mirror the network infrastructure of the physical twin.
- **Traffic emulation**: The captured network traffic from the physical train is replayed in the digital twin network.
- **Traffic capture and analysis**: Network traffic in the digital twin is captured using `tcpdump` and saved to a pcap file. This allows for a detailed comparison and analysis of how closely the digital twin replicates the physical twin's network behavior.
## Results showing the different results.
![alt text](https://github.com/johnsengendo/DigitalTwins_part2/blob/main/results/Screenshot%202024-09-04%20092646.png)

## System design.
![alt text](https://github.com/johnsengendo/DigitalTwins_part2/blob/main/Images/Image.jpg)
