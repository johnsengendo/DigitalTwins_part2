import pandas as pd
import time
from mininet.net import Mininet
from mininet.node import OVSSwitch
from mininet.link import Link
from mininet.log import setLogLevel, info
from mininet.cli import CLI
import os

# Loading the CSV data
csv_file = "predictions_with_bandwidth.csv"

# Ensuring the script and CSV file are in the same directory
current_directory = os.path.dirname(os.path.abspath(__file__))
csv_path = os.path.join(current_directory, csv_file)

# Loading the CSV data
data = pd.read_csv(csv_path)

# Setting up the Mininet network
setLogLevel("info")

# Creating a network
net = Mininet(switch=OVSSwitch)

# Adding two switches
s1 = net.addSwitch("s1")
s2 = net.addSwitch("s2")

# Adding two hosts
h1 = net.addHost("h1", ip="10.0.0.1/24")
h2 = net.addHost("h2", ip="10.0.0.2/24")

# Adding links between the hosts and the switches
net.addLink(h1, s1)
net.addLink(s1, s2)
net.addLink(s2, h2)

# Starting the network
net.start()

# Setting bandwidth limits using tc
h1.cmd("tc qdisc add dev h1-eth0 root tbf rate 10Mbit burst 15k latency 1ms")
s1.cmd("tc qdisc add dev s1-eth1 root tbf rate 10Mbit burst 15k latency 1ms")
s1.cmd("tc qdisc add dev s1-eth2 root tbf rate 10Mbit burst 15k latency 1ms")
s2.cmd("tc qdisc add dev s2-eth1 root tbf rate 10Mbit burst 15k latency 1ms")
s2.cmd("tc qdisc add dev s2-eth2 root tbf rate 10Mbit burst 15k latency 1ms")
h2.cmd("tc qdisc add dev h2-eth0 root tbf rate 10Mbit burst 15k latency 1ms")

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