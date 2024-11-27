import * as THREE from 'three';
import { PLYLoader } from 'three/examples/jsm/loaders/PLYLoader';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls';
import { VRButton } from 'three/addons/webxr/VRButton.js';

// Initialize Scene, Camera, Renderer
const scene = new THREE.Scene();
scene.background = new THREE.Color(0xdddddd);

const camera = new THREE.PerspectiveCamera(45, window.innerWidth / window.innerHeight, 0.1, 5000);
const renderer = new THREE.WebGLRenderer();
renderer.setSize(window.innerWidth, window.innerHeight);
document.body.appendChild(VRButton.createButton(renderer));
renderer.xr.enabled = false;
document.body.appendChild(renderer.domElement);

// Initialize Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(512, 512, 512); // Set default target
controls.update();

// Define start frame and number of frames
const startFrame = 1480; // Example start frame
const numFrames = 60; // Example number of frames to include
const filePrefix = 'test_data/redandblack_vox10_';
const fileSuffix = '.ply';

// Dynamically generate file names based on the range
const plyFiles = Array.from({ length: numFrames }, (_, i) => `${filePrefix}${startFrame + i}${fileSuffix}`);


const loader = new PLYLoader();
const models = []; // Array to hold preloaded models
let currentModel = null;
let currentIndex = 0;
let direction = 1

// Preload all models
function preloadModels() {
    plyFiles.forEach((file, index) => {
        loader.load(
            file,
            (geometry) => {
                const material = new THREE.PointsMaterial({
                    size: 5.0,
                    vertexColors: true 
                });
                const points = new THREE.Points(geometry, material);
                models.push(points);

                // Add the first model to the scene initially
                if (index === 0) {
                    currentModel = points;
                    scene.add(points);
                }
            },
            (xhr) => {
                console.log(`Loading ${file}: ${(xhr.loaded / xhr.total) * 100}% loaded`);
            },
            (error) => {
                console.error(`Error loading ${file}:`, error);
            }
        );
    });
}

function switchModel() {
    if (models.length === 0) return; // Ensure models are preloaded

    // Remove the current model
    if (currentModel) {
        scene.remove(currentModel);
    }

    // Update index based on the current direction
    currentIndex += direction;

    // If at the end of the list, reverse the direction
    if (currentIndex > models.length) {
        currentIndex = models.length - 1; // Stay within bounds
        direction = -1; // Switch to backward
    }
    // If at the start of the list, reverse the direction
    else if (currentIndex < 0) {
        currentIndex = 0; // Stay within bounds
        direction = 1; // Switch to forward
    }

    // Set the new current model and add it to the scene
    currentModel = models[currentIndex];
    scene.add(currentModel);
}

// Animate and Render Scene
function animate() {
    renderer.setAnimationLoop(() => {
        // Switch the model every 1/30th of a second (30 FPS)
        const frameDuration = 1000 / 30; // 30 FPS = ~33ms per frame

        setTimeout(() => {
            switchModel();
        }, frameDuration);

        renderer.render(scene, camera);
    });
}

// Start Preloading Models and Animation
preloadModels();
animate();
