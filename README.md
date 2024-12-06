# Demo for Unified Compression

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
    ├── dependencies             # Dependencies
    ├── LICENSE
    └── README.md
```

## Installation

For setup, collect the required dependencies:
```
	mkdir dependencies && cd dependencies
	git clone git@gitlab.uni-hannover.de:ve-intern/research/unified-compression.git
```


### Sender
On the sender device, make sure a ZED Camera is plugged in, then run
```
cd sender
docker-compose up --build
```

### Receiver 
On the receiver side, we need to connect the Occulus Quest 2 before running:

First, check if the headset is accesible via adb:
```
adb devices
```

If it is, use reverse port forwarding to make the browser accesible
```
adb reverse tcp:5173 tcp:5173
adb reverse tcp:8765 tcp:8765
adb reverse tcp:8765 tcp:8765
```

Finally, starting the server:
```
cd receiver
docker-compose up --build
```


The visualization should now be available via https://localhost:5173/ on the receiver device **and** via the Occulus Headset 

## Usage
### Rendering 

Current rendering is setup via

Build:
```
docker build -t threejs-app src/rendering/
```

Run:
```
docker run -p 5173:5173 -v $(pwd)/src/rendering:/app/src/rendering threejs-app
```

### Running the VR headset

For running the Occulus Quest 2 VR headset, we use adb. It is required to install Sidequest for authorization.

First, check if the headset is accesible via adb:
```
adb devices
```

If it is, use reverse port forwarding to make the browser accesible
```
adb reverse tcp:5173 tcp:5173
adb reverse tcp:8765 tcp:8765
adb reverse tcp:8765 tcp:8765
```
