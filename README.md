# Demo for Unified Compression


## Installation


## JETSON Building
### Sender
```
cd sender
docker-compose up --build
```

### Receiver 

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
