# Demo for Unified Compression

This repository contains the demo for the paper [Learned Compression in Adaptive Point Cloud Streaming](https://dl.acm.org/doi/10.1145/3712676.3719266).

## Hardware
This demo requires the following Hardware:
- 2 Jetson AGX Orin developer kits
- 1 Occulus Quest 2 VR Headset + Link Cable
- 1 ZED2i Stereo Camera 

Cables and peripherals.

## Project Structure 
The project is structured as follows
```
    .
    ├── sender                   # Sender modules
    |	├── capturer             
    |	├── encoder             
    |	├── mediaserver             
    |	└── docker-compose.yaml
    ├── receiver                 # Receiver modules
    |	├── client             
    |	├── decoder             
    |	├── visualization             
    |	└── docker-compose.yaml
    ├── shared                   # Shared modules
    ├── evaluation               # Plot generation and logs
    ├── dependencies             # Dependencies
    ├── tests             	 # Test scripts for development
    ├── LICENSE
    └── README.md
```

## Installation

For setup, collect the required dependencies:
```
mkdir dependencies && cd dependencies
git clone --branch unified_demo https://github.com/ikt-luh/Unified-Point-Cloud-Compression.git
```


### Sender
First, we need to open the port for the media server:

```
sudo iptables -A INPUT -p tcp --dport 4000 -j ACCEPT
```

On the sender device, make sure a ZED Camera is plugged in, then run
```
cd sender
docker-compose up --build
```

### Receiver 


Starting the server:
```
cd receiver
docker-compose up --build
```


The visualization should now be available via https://localhost:5173/ on the receiver device.
Furthermore, the Dashboard is available via https://localhost:5000/ 

### Running the VR headset

For running the Occulus Quest 2 VR headset, you can use ADB. The implementation for this is currently under development and will be available in the coming weeks.

On the receiver device, run
```
sudo apt-get install android-tools-adb
```
You will have to update udev rules for the headset. For this, run 
```
lsusb
```
and get the ID of the Oculus VR, Inc. Quest 2 headset. This should look like "AAAA:BBBB".
Then, add the following line to /etc/udev/rules.d/51-android.rules (Replacing AAAA and BBBB with the Device ID from lsusb)
```
SUBSYSTEM=="usb", ATTR{idVendor}=="AAAA", ATTR{idProduct}=="BBBB", MODE="0666", GROUP="plugdev"
``` 

Finally, update udev rules
```
sudo udevadm control --reload-rules
```
and unplug/replug the device


Now, the device should be listed by
```
adb devices
```

In the Quest 2 Headset, allow connection. (Automatic pop-up)


If it is, use reverse port forwarding to make the browser accesible
```
adb reverse tcp:5173 tcp:5173
adb reverse tcp:8765 tcp:8765
adb forward tcp:8765 tcp:8765
```


## Notes
### Adding the docker group
By default, you might not be in the docker group, unable to run containers. To solve this:
```
sudo groupadd docker
sudo usermod -aG docker $USER
newgrp docker
```
You might need to reboot the device.

### Setting the power mode
For maximum performance, we want both jetson's to run in Power Mode 0 (MAXN)
```
sudo /usr/sbin/nvpmodel -m 0
```

### Monitoring ressource utilization
[jetson_stats](https://github.com/rbonghi/jetson_stats) can be used to monitor ressource utilization on Jetson Devices.
Install via:
```
sudo pip3 install -U jetson-stats
```
and run with
```
jtop
```

### Citation
If you find this work helpfull, consider citing:
```
@inproceedings{10.1145/3712676.3719266,
	author = {Rudolph, Michael and Rizk, Amr},
	title = {Learned Compression in Adaptive Point Cloud Streaming: Opportunities, Challenges and Limitations},
	year = {2025},
	doi = {10.1145/3712676.3719266},
	booktitle = {Proceedings of the 16th ACM Multimedia Systems Conference},
	pages = {328–334},
	numpages = {7},
	series = {MMSys '25}
}
```
