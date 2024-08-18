import pandas as pd
import time
from comnetsemu.net import Containernet
from comnetsemu.node import Host
from comnetsemu.link import TCLink
from mininet.log import setLogLevel, info
from mininet.cli import CLI
import os

# Loading the CSV data
csv_file = "your_csv_file.csv"  # Replace with your actual CSV file name

# Ensuring the script and CSV file are in the same directory
current_directory = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_directory, csv_file)

# Loading the CSV data
data = pd.read_csv(csv_path)

# Setting up the Comnetsemu network
setLogLevel("info")

# Creating a network
net = Containernet()

# Adding two switches
s1 = net.addSwitch("s1")
s2 = net.addSwitch("s2")

# Adding two hosts
h1 = net.addHost("h1", ip="10.0.0.1/24")
h2 = net.addHost("h2", ip="10.0.0.2/24")

# Adding links between the hosts and the switches
net.addLink(h1, s1, cls=TCLink, bw=10)  # 10 Mbps link
net.addLink(s1, s2, cls=TCLink, bw=10)  # 10 Mbps link
net.addLink(s2, h2, cls=TCLink, bw=10)  # 10 Mbps link

# Starting the network
net.start()

# Defining the pcap file path in the same directory as the script
pcap_file = os.path.join(current_directory, "h1_capture.pcap")

# Starting tcpdump on h1 to capture traffic on its interface and store it in the specified directory
h1.cmd(f"tcpdump -i h1-eth0 -w {pcap_file} &")

# Starting the iperf server on h2
h2.cmd("iperf3 -s &")

# Iterating over each row in the CSV file
for index, row in data.iterrows():
    bandwidth_bps = row['Bandwidth_bps']
    protocol = row['Protocol']

    # Converting bandwidth from bps to Mbps for iperf (1 Mbps = 10^6 bps)
    bandwidth_mbps = bandwidth_bps / 1e6

    # Adjusting the protocol if necessary (TCP or UDP)
    if protocol == 'TCP':
        protocol_option = ''
    else:
        protocol_option = '-u'

    # Running the iperf client on h1 with the specified bandwidth and protocol
    iperf_command = f"iperf3 -c 10.0.0.2 {protocol_option} -b {bandwidth_mbps}M -t 1 -i 1 > iperf_output_{index}.txt &"
    h1.cmd(iperf_command)
    
    # Waiting for 1 second before running the next iperf command
    time.sleep(1)

# Stopping tcpdump after the test is completed
h1.cmd("pkill tcpdump")

# Optionally, dropping into the CLI for further testing or interaction
CLI(net)

# Stopping the network
net.stop()