import * as THREE from 'three';
import { VRButton } from 'three/addons/webxr/VRButton.js';

// Initialize Scene, Camera, Renderer
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xdddddd);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 5000);
camera.position.set(0, 1, 3); // Position the camera for VR

const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
renderer.xr.enabled = true; // Enable WebXR for VR functionality
document.body.appendChild(renderer.domElement);
document.body.appendChild(VRButton.createButton(renderer));

// Movement variables
const movementSpeed = 0.05; // Adjust the speed of movement
const direction = new THREE.Vector3(); // To calculate movement direction
const quaternion = new THREE.Quaternion(); // To rotate based on the headset's orientation

 // Set up WebSocket to receive point cloud data
 const ws = new WebSocket('ws://localhost:8765');
 ws.binaryType = "arraybuffer"; // Receive data as an ArrayBuffer
 let dynamicGeometry = null;


 ws.onopen = () => {
     console.log('Connected to WebSocket server');
 };

 ws.onmessage = (event) => {
    const buffer = event.data; // Received binary data

    // Constants based on known data structure
    const pointSize = 4 * 3; // Each point: 3 float32 (4 bytes each)
    const colorSize = 1 * 3; // Each color: 3 uint8 (1 byte each)
    const bytesPerPoint = pointSize + colorSize;

    // Calculate number of points
    const pointCount = buffer.byteLength / bytesPerPoint;

    if (!Number.isInteger(pointCount)) {
        console.error("Data size does not match point cloud structure. Check your server data.");
        return;
    }

    // Extract points and colors directly from the buffer
    const float32Array = new Float32Array(buffer, 0, pointCount * 3); // Points
    const uint8Array = new Uint8Array(buffer, pointCount * pointSize, pointCount * 3); // Colors
    const normalizedColors = new Float32Array(uint8Array.length);
    for (let i = 0; i < uint8Array.length; i++) {
        normalizedColors[i] = uint8Array[i] / 255; // Normalize to [0, 1]
    }

    // Apply scaling to fit the point cloud in the camera's view
    const scaleFactor = 0.001;  // Scale down the point cloud
    for (let i = 0; i < float32Array.length; i++) {
        float32Array[i] *= scaleFactor; // Scale coordinates
    }

    // Create or update geometry
    if (!dynamicGeometry) {
        dynamicGeometry = new THREE.BufferGeometry();
        const material = new THREE.PointsMaterial({
            size: 0.005,
            vertexColors: true,
        });
        const points = new THREE.Points(dynamicGeometry, material);
        scene.add(points);
    }

    // Update attributes dynamically
    dynamicGeometry.setAttribute('position', new THREE.Float32BufferAttribute(float32Array, 3));
    dynamicGeometry.setAttribute('color', new THREE.Float32BufferAttribute(normalizedColors, 3));

    // Notify Three.js of the changes
    dynamicGeometry.attributes.position.needsUpdate = true;
    dynamicGeometry.attributes.color.needsUpdate = true;

    // Optionally center the cloud
    centerPointCloud();
};



// Center the point cloud in front of the user
function centerPointCloud() {
    if (!dynamicGeometry) return;

    // Compute bounding box and center
    dynamicGeometry.computeBoundingBox();
    const box = dynamicGeometry.boundingBox;
    const center = new THREE.Vector3();
    box.getCenter(center);

    // Translate geometry to center it
    dynamicGeometry.translate(-center.x, -center.y, -center.z);
}
// Add VR Controller Input
function setupController() {
    //const controller = renderer.xr.getController(0);
    //scene.add(controller);

    //// Listen for controller button press
    //controller.addEventListener('selectstart', () => {
    //    centerPointCloud();
    //});
}

// Handle joystick input for movement
function handleMovement() {
    //const session = renderer.xr.getSession();
    //if (!session) return;

    //// Loop through input sources (e.g., controllers)
    //session.inputSources.forEach((inputSource) => {
    //    if (inputSource.gamepad) {
    //        const axes = inputSource.gamepad.axes; // Get joystick axes

    //        // Move forward/backward (axes[1]) and left/right (axes[0])
    //        direction.set(axes[0] * movementSpeed, 0, -axes[1] * movementSpeed);
    //        direction.applyQuaternion(camera.quaternion); // Rotate direction by camera's orientation

    //        // Update camera position
    //        camera.position.add(direction);
    //    }
    //});
}


// Animate and Render Scene
function animate() {
    renderer.setAnimationLoop(() => {
        handleMovement(); // Handle joystick input
        renderer.render(scene, camera);
    });
}

// Start Preloading Models, Setup Controllers, and Animation
setupController();
animate();
