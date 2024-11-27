# Demo for Unified Compression


## Installation


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
