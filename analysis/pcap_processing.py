
'''
Installing necessary Python libraries and system packages:
1. Installing the pyshark and pandas Python libraries.
2. Installing the nest_asyncio library, which allows nested use of asyncio event loops.
3. Updating the package list on the system.
4. Installing the tshark network protocol analyzer.
'''

!pip install pyshark pandas
!pip install nest_asyncio
!apt-get update
!apt-get install -y tshark

# Importing the packages
import pyshark
import pandas as pd
import nest_asyncio

# Mounting to Google Drive for accessing and saving data
from google.colab import drive
drive.mount('/content/drive')

'''
Performing necessary setup steps for processing a PCAP file:
1. Enabling nested asyncio to allow nested use of asyncio event loops.
2. Defining the path to the PCAP file stored in the drive.
'''
nest_asyncio.apply()
pcap_file_path = '/content/drive/My Drive/pcap/server_2.pcap'

# Initializing list to store extracted timestamps
timestamps = []

'''
Capturing packets and extracting timestamps:
1. Capturing packets from the PCAP file using pyshark.
2. Iterating over each packet and extracting the timestamp.
3. Appending the extracted timestamp to the list.
4. Closing the packet capture file.
'''

cap = pyshark.FileCapture(pcap_file_path)
for packet in cap:
    timestamps.append(int(packet.sniff_time.timestamp()))
cap.close()

# Creating a DataFrame to calculate packets per second
df = pd.DataFrame({'timestamp': timestamps})

# Calculating packets per second
packets_per_sec = df.groupby('timestamp').size().reset_index(name='packets_per_sec')

'''
Saving the packets per second data to a CSV file and displaying the first few rows.
'''

csv_file_path = '/content/drive/My Drive/pcap/packets_per_sec_analysis.csv'
packets_per_sec.to_csv(csv_file_path, index=False)

# Displaying the first few rows of the packets per second data
packets_per_sec.head()

# @title timestamp vs packets_per_sec

from matplotlib import pyplot as plt
import seaborn as sns
def _plot_series(series, series_name, series_index=0):
  palette = list(sns.palettes.mpl_palette('Dark2'))
  xs = series['timestamp']
  ys = series['packets_per_sec']

  plt.plot(xs, ys, label=series_name, color=palette[series_index % len(palette)])

fig, ax = plt.subplots(figsize=(10, 5.2), layout='constrained')
df_sorted = packets_per_sec.sort_values('timestamp', ascending=True)
_plot_series(df_sorted, '')
sns.despine(fig=fig, ax=ax)
plt.xlabel('timestamp')
_ = plt.ylabel('packets_per_sec')