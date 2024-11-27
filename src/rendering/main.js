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
renderer.xr.enabled = true;
document.body.appendChild(renderer.domElement);

// Initialize Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.target.set(512, 512, 512); // Set default target
controls.update();

// List of .ply file URLs
const plyFiles = [
    'test_data/redandblack_vox10_1450.ply',
    'test_data/redandblack_vox10_1451.ply',
    'test_data/redandblack_vox10_1452.ply',
];

const loader = new PLYLoader();
let currentModel = null;
let currentIndex = 0;

function loadNextModel() {
    if (currentModel) {
        scene.remove(currentModel); // Remove the previous model
    }

    loader.load(
        plyFiles[currentIndex],
        (geometry) => {
            const material = new THREE.PointsMaterial({
                size: 2.0,
                vertexColors: true 
            });

            currentModel = new THREE.Points(geometry, material);
            scene.add(currentModel);

            // Advance to the next model or loop back to the first
            currentIndex = (currentIndex + 1) % plyFiles.length;
            //setTimeout(loadNextModel, 5000);
        },
        (xhr) => {
            console.log(`Loading ${plyFiles[currentIndex]}: ${(xhr.loaded / xhr.total) * 100}% loaded`);
        },
        (error) => {
            console.error(`Error loading ${plyFiles[currentIndex]}:`, error);
        }
    );
}

// Start Loading Models
loadNextModel();

// Animate and Render Scene
function animate() {
    requestAnimationFrame(animate);

    renderer.render(scene, camera);
}

renderer.setAnimationLoop(animate);
